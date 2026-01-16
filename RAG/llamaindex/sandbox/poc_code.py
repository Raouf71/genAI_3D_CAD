import os
from llama_cloud_services import LlamaParse
from copy import deepcopy
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode, Document

from llama_index.graph_stores.neo4j import Neo4jPGStore

from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)
from llama_index.core import PropertyGraphIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# vector index
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine

# kg index
from llama_index.core.indices.property_graph import VectorContextRetriever

import nest_asyncio
nest_asyncio.apply()

######################################################## Credentials
# API access to llama-cloud
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-WROpF69GBXDRmw9jP8oiN1lwU6iDTbJs0kRY0nj3ReVfXtuY"
os.environ["OPENAI_API_KEY"] = "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"

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

llm = OpenAI(model="gpt-4o")
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = llm
Settings.embed_model = embed_model


######################################################## Parse
docs = LlamaParse(
    parse_mode="parse_page_with_agent",
    model="openai-gpt-4-1-mini",
    high_res_ocr=True,
    adaptive_long_table=True,
    outlined_table_extraction=True,
    output_tables_as_HTML=True,
).load_data("../data/lager.pdf")

######################################################## Split

def get_sub_docs(docs):
    """Split docs into pages, by separator."""
    sub_docs = []
    for doc in docs:
        doc_chunks = doc.text.split("\n---\n")
        for doc_chunk in doc_chunks:
            sub_doc = Document(
                text=doc_chunk,
                metadata=deepcopy(doc.metadata),
            )
            sub_docs.append(sub_doc)

    return sub_docs

# this will split into pages
sub_docs = get_sub_docs(docs)

######################################################## Run vector search

base_index = VectorStoreIndex.from_documents(sub_docs, embed_model=embed_model)
vector_retriever = base_index.as_retriever(similarity_top_k=10)
base_query_engine = RetrieverQueryEngine(vector_retriever)

response = base_query_engine.query(
    "Worum geht es in dem Dokument? Antworte in 2-3 Sätzen."
)
print(str(response))

######################################################## Init KG + Index + Extract entities/relations

graph_store = Neo4jPGStore(
    username="neo4j",
    password="graph1312",
    url="bolt://localhost:7687",
)
vec_store = None

index = PropertyGraphIndex.from_documents(
    sub_docs,
    embed_model=OpenAIEmbedding(model_name="text-embedding-3-small"),
    kg_extractors=[
        ImplicitPathExtractor(),
        SimpleLLMPathExtractor(
            llm=OpenAI(model="gpt-3.5-turbo", temperature=0.3),
            num_workers=4,
            max_paths_per_chunk=10,
        ),
    ],
    property_graph_store=graph_store,
    show_progress=True,
)

######################################################## Run KG search

kg_retriever = VectorContextRetriever(
    index.property_graph_store,
    embed_model=OpenAIEmbedding(model_name="text-embedding-3-small"),
    similarity_top_k=5,
    path_depth=1,
    # include_text=False,
    include_text=True,
)

nodes = kg_retriever.retrieve(
    "Gib mir die ganze Reihe für den Zylinderrollenlager DIN5412 mit d=55"
)

# Custom retriever
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from typing import List

######################################################## Run both vector and KG search

class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both KG vector search and direct vector search."""

    def __init__(self, kg_retriever, vector_retriever):
        self._kg_retriever = kg_retriever
        self._vector_retriever = vector_retriever

    def _retrieve(self, query_bundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        kg_nodes = self._kg_retriever.retrieve(query_bundle)
        vector_nodes = self._vector_retriever.retrieve(query_bundle)

        unique_nodes = {n.node_id: n for n in kg_nodes}
        unique_nodes.update({n.node_id: n for n in vector_nodes})
        return list(unique_nodes.values())
custom_retriever = CustomRetriever(kg_retriever, vector_retriever)

nodes = custom_retriever.retrieve(
    "Gib mir die ganze Reihe für den Zylinderrollenlager DIN5412 mit d=55"
)

######################################################## Run agent

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RetrieverQueryEngine

kg_query_engine = RetrieverQueryEngine(custom_retriever)
kg_query_tool = QueryEngineTool(
    query_engine=kg_query_engine,
    metadata=ToolMetadata(
        name="query_tool",
        description="Provides information about row table lookups",
    ),
)

# from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context

agent = ReActAgent(
    tools=[kg_query_tool],
    llm=llm,
    verbose=True,
    allow_parallel_tool_calls=False,
)

# context to hold this session/state
ctx = Context(agent)


# positive test case
handler = agent.run(
    "Listen Sie mir bitte die komplette Reihe (mit allen Spaltenwerten) für den Axialrillenkugellager DIN 711 wo d_w=30",
    ctx=ctx)
from llama_index.core.agent.workflow import ToolCallResult, AgentStream

async for ev in handler.stream_events():
    # if isinstance(ev, ToolCallResult):
    #     print(f"\nCall {ev.tool_name} with {ev.tool_kwargs}\nReturned: {ev.tool_output}")
    if isinstance(ev, AgentStream):
        print(f"{ev.delta}", end="", flush=True)

response = await handler