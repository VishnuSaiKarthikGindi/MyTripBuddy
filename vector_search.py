from __future__ import annotations

from typing import List, Optional
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain.tools import Tool
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


class VectorSearch:
    """
    High-level wrapper around Pinecone + LangChain to index web pages and perform
    similarity search via a retriever.
    """

    def __init__(
        self,
        pinecone_api_key: Optional[str] = None,
        index_name: str = "mytripbuddy",
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        dimension: int = 1536,
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> None:
        self.index_name = index_name

        pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is not set.")

        openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        existing_index_names = [meta["name"] for meta in pc.list_indexes()]
        if self.index_name not in existing_index_names:
            pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
        index = pc.Index(self.index_name)

        # Initialize embeddings and vector store
        embeddings = OpenAIEmbeddings(model=embedding_model, api_key=openai_api_key)
        self.vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

    def _split_documents(self, urls: List[str]):
        loaders = [WebBaseLoader(url) for url in urls]
        docs_nested = [loader.load() for loader in loaders]
        docs = [doc for sub in docs_nested for doc in sub]

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500,
            chunk_overlap=50,
        )
        return splitter.split_documents(docs)

    def add_urls(self, urls: List[str]) -> int:
        """
        Load, split and upsert documents from the provided URLs into the vector store.

        Returns the number of chunks inserted.
        """
        doc_chunks = self._split_documents(urls)
        if not doc_chunks:
            return 0
        self.vectorstore.add_documents(doc_chunks)
        return len(doc_chunks)

    def get_retriever(self, k: int = 4):
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def similarity_search(self, query: str, k: int = 4):
        retriever = self.get_retriever(k=k)
        return retriever.invoke(query)


def make_vector_search_tool(vs: VectorSearch, k: int = 4) -> Tool:
    """
    Create a LangChain Tool that performs retrieval over the VectorSearch index
    and returns concatenated text snippets.
    """
    retriever = vs.get_retriever(k=k)

    def _run(query: str) -> str:
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant documents found."
        return "\n\n".join(doc.page_content for doc in docs)

    return Tool(
        name="VectorSearch",
        description=(
            "Retrieve relevant passages from indexed FAQs and web documents. "
            "Use this for FAQ-style or knowledge-based queries."
        ),
        func=_run,
    )

