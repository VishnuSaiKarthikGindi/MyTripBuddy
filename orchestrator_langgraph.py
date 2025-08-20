from __future__ import annotations

from typing import Optional, Dict, Any, List, TypedDict
import os
import re
import json

from langgraph.graph import StateGraph, START, END
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from SearchPOITool import search_tool as tripadvisor_search_tool
from weather_tool import make_weather_tool
from google_maps_tool import make_route_tool, get_route
from vector_search import VectorSearch, make_vector_search_tool

try:
    from amadeus_tool import get_amadeus_tools
except Exception:
    get_amadeus_tools = None  # type: ignore


class GraphState(TypedDict):
    question: str
    generation: Optional[str]
    route: Optional[str]


def _build_llm() -> ChatOpenAI:
    model = os.getenv("ORCHESTRATOR_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("ORCHESTRATOR_TEMP", "0"))
    # ChatOpenAI from langchain_openai reads OPENAI_API_KEY
    return ChatOpenAI(model=model, temperature=temperature)


def _route_with_llm(llm: ChatOpenAI, question: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a router. Choose the best datasource for a user question.\n"
            "Return ONLY a JSON object with a single key 'datasource' whose value is one of: \n"
            "['tripadvisor', 'weather', 'route', 'vector', 'amadeus'].\n"
            "Use: tripadvisor for accommodations/attractions/restaurants/places.\n"
            "Use: weather for weather/temperature/forecast queries.\n"
            "Use: route for driving directions (origin to destination).\n"
            "Use: vector for FAQs/knowledge/document-based queries.\n"
            "Use: amadeus for flight/airline/tickets/fares.\n"
            "If unclear, default to 'vector'."
        )),
        ("human", "{question}")
    ])
    msg = prompt | llm
    result = msg.invoke({"question": question})
    try:
        data = json.loads(result.content)
        choice = str(data.get("datasource", "vector")).lower()
    except Exception:
        text = (result.content or "").strip().lower()
        if any(k in text for k in ["tripadvisor", "hotel", "attraction", "restaurant"]):
            choice = "tripadvisor"
        elif "weather" in text or "forecast" in text:
            choice = "weather"
        elif "route" in text or "direction" in text:
            choice = "route"
        elif "amadeus" in text or "flight" in text:
            choice = "amadeus"
        else:
            choice = "vector"
    if choice not in {"tripadvisor", "weather", "route", "vector", "amadeus"}:
        choice = "vector"
    return choice


def _parse_origin_dest(question: str) -> Optional[tuple[str, str]]:
    q = question.strip()
    patterns = [
        r"from\s+(?P<origin>.+?)\s+to\s+(?P<dest>.+)$",
        r"between\s+(?P<origin>.+?)\s+and\s+(?P<dest>.+)$",
        r"^(?P<origin>.+?)\s+to\s+(?P<dest>.+)$",
    ]
    for p in patterns:
        m = re.search(p, q, re.IGNORECASE)
        if m:
            return m.group("origin"), m.group("dest")
    return None


def _extract_route_with_llm(llm: ChatOpenAI, question: str) -> Optional[tuple[str, str]]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Extract driving route endpoints. Return ONLY JSON: {\"origin\": string, \"destination\": string}.\n"
            "If either endpoint is missing, return an empty JSON {}."
        )),
        ("human", "{question}")
    ])
    msg = prompt | llm
    res = msg.invoke({"question": question})
    try:
        data = json.loads(res.content)
        origin = data.get("origin")
        dest = data.get("destination")
        if origin and dest:
            return origin, dest
    except Exception:
        pass
    return None


def build_orchestrator_app(
    include_amadeus: bool = True,
    vector_index_name: str = "mytripbuddy",
) -> tuple[Any, Dict[str, Any]]:
    # LLM for routing and extraction
    llm = _build_llm()

    # Tools
    tools: Dict[str, Any] = {}
    tools["tripadvisor"] = tripadvisor_search_tool
    tools["weather"] = make_weather_tool()
    tools["route_tool"] = make_route_tool()

    # Vector search tool
    vector_tool: Optional[Tool] = None
    try:
        vs = VectorSearch(index_name=vector_index_name)
        vector_tool = make_vector_search_tool(vs)
        tools["vector_tool"] = vector_tool
    except Exception:
        tools["vector_tool"] = None

    # Amadeus (optional)
    amadeus_tools: List[Tool] = []
    if include_amadeus and get_amadeus_tools is not None:
        try:
            amadeus_tools = get_amadeus_tools()
        except Exception:
            amadeus_tools = []

    # Graph
    graph = StateGraph(GraphState)

    def router_node(state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        route = _route_with_llm(llm, question)
        return {"route": route}

    def run_tripadvisor(state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        tool = tools["tripadvisor"]
        try:
            result = tool.run(question)  # type: ignore
            if isinstance(result, (dict, list)):
                result = json.dumps(result, indent=2)
            return {"generation": str(result)}
        except Exception as e:
            return {"generation": f"TripAdvisor error: {e}"}

    def run_weather(state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        tool = tools["weather"]
        try:
            return {"generation": str(tool.run(question))}  # type: ignore
        except Exception as e:
            return {"generation": f"Weather error: {e}"}

    def run_route(state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        od = _parse_origin_dest(question)
        if not od:
            od = _extract_route_with_llm(llm, question)
        if not od:
            return {"generation": "Please provide route as: 'from ORIGIN to DESTINATION'"}
        origin, dest = od
        try:
            return {"generation": get_route(origin, dest)}
        except Exception as e:
            return {"generation": f"Route error: {e}"}

    def run_vector(state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        tool = tools.get("vector_tool")
        if tool is None:
            return {"generation": "Vector search not configured."}
        try:
            return {"generation": str(tool.run(question))}  # type: ignore
        except Exception as e:
            return {"generation": f"Vector error: {e}"}

    def run_amadeus(state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        if not amadeus_tools:
            return {"generation": "Amadeus tools not configured."}
        for t in amadeus_tools:
            try:
                return {"generation": str(t.run(question))}
            except Exception:
                continue
        return {"generation": "Could not process with Amadeus tools."}

    # Register nodes
    graph.add_node("router", router_node)
    graph.add_node("tripadvisor", run_tripadvisor)
    graph.add_node("weather", run_weather)
    graph.add_node("route", run_route)
    graph.add_node("vector", run_vector)
    graph.add_node("amadeus", run_amadeus)

    def route_edges(state: GraphState) -> str:
        route = state.get("route") or "vector"
        if route not in {"tripadvisor", "weather", "route", "vector", "amadeus"}:
            route = "vector"
        return route

    graph.add_conditional_edges(
        START,
        lambda s: "router",
        {"router": "router"},
    )
    graph.add_conditional_edges(
        "router",
        route_edges,
        {
            "tripadvisor": "tripadvisor",
            "weather": "weather",
            "route": "route",
            "vector": "vector",
            "amadeus": "amadeus",
        },
    )
    graph.add_edge("tripadvisor", END)
    graph.add_edge("weather", END)
    graph.add_edge("route", END)
    graph.add_edge("vector", END)
    graph.add_edge("amadeus", END)

    app = graph.compile()
    context = {"llm": llm, "tools": tools}
    return app, context


def run_query(question: str) -> str:
    app, _ = build_orchestrator_app()
    state: GraphState = {"question": question, "generation": None, "route": None}
    result: Dict[str, Any] = {}
    for step in app.stream(state):  # stream to run through graph
        result.update(step)
    gen = result.get("generation")
    return gen or ""


if __name__ == "__main__":
    print(run_query("top attractions in Paris within 5 km"))
