### Project Description

MyTripBuddy is a multi-agent travel and information assistant designed to handle diverse queries related to travel and FAQs. It integrates multiple tools and capabilities into a seamless system, allowing users to ask questions about:

1. Travel Planning:
   - Accommodations, attractions, and restaurants via the TripAdvisor API using a custom-built LangChain Tool.
   - Flight ticket information using the Amadeus Toolkit.
   - Weather forecasts through OpenWeatherMap API.
   - Driving routes generated with Google Maps Directions API.

2. Knowledge Retrieval:
   - A vectorstore retriever acts as another agent for answering FAQ-style or knowledge-based queries, such as technical concepts or indexed topics from Pinecone vectorstore.

The system employs a routing mechanism to intelligently direct user questions to the appropriate agent:
- Tools Agent: For travel and location-related queries.
- Vectorstore Agent: For knowledge retrieval from pre-indexed documents.

By combining LLM-driven agents with powerful APIs, the project provides a robust, user-friendly platform for answering complex questions and assisting with travel plans in real-time.

### Setup

- Set environment variables:
  - `TRIPADVISOR_API_KEY`
  - `GOOGLE_MAPS_API_KEY`
  - `OPENAI_API_KEY`
  - `PINECONE_API_KEY`

- Install dependencies:
  - `pip install -r requirements.txt`

### Usage

- TripAdvisor Tool example (Python):
```python
from SearchPOITool import search_tool
print(search_tool.run("top attractions in Paris within 5 km"))
```