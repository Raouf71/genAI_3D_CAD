import asyncio
import os
import logging
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
import ollama
from lightrag.llm.ollama import ollama_model_complete, ollama_embed, _ollama_model_if_cache
from typing import Union, AsyncIterator
from lightrag import LightRAG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def count_entities_and_relations(response):
    entities = response.count("entity<|#|>")  # Count entities
    relations = response.count("relation<|#|>")  # Count relationships
    print(f"Entities: {entities}, Relationships: {relations}")
    return entities, relations

async def main():
    config = RAGAnythingConfig(
        working_dir="./rag_storage_text_only",
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True
    )

    async def ollama_llm_model(prompt, system_prompt=None, history_messages=[], **kwargs):
        print("\033[36m==================================[ollama_llm_model]============================================%s\033[0m")
        # print("\033[36mPROMPT: %s\033[0m" % prompt)

        if history_messages is None:
            history_messages = []

        # Detect LightRAG answer calls (they have ---Context--- and Document Chunks)
        if system_prompt and "---Context---" in system_prompt:
            # Build a single big user prompt that includes context + query
            merged_prompt = (
                "You are given a document in the sections below.\n"
                "Use ONLY that content to answer the user's question.\n\n"
                + system_prompt
                + "\n\nUser query: "
                + (prompt or "")
            )
            # Replace the short system prompt with something simple
            system_prompt = (
                "You are a helpful assistant that MUST answer "
                "only using the document text included in the user message."
            )
            prompt = merged_prompt

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for m in (history_messages or []):
            messages.append(m)

        messages.append({"role": "user", "content": prompt})

        response = ollama.chat(
            model="llama3.2-vision:11b-instruct-q4_K_M",
            messages=messages,
        )
        result = response["message"]["content"]

        entities, relations = count_entities_and_relations(result)
        print(f"Entities: {entities}, Relationships: {relations}")
        # print("\033[33mRaw model response: %s\033[0m" % result)
        print("==============================")
        return result

    rag = RAGAnything(
        config=config,
        lightrag_kwargs = { "llm_model_kwargs": {"options": {"num_ctx": 32768}}},
        llm_model_func=ollama_llm_model,
        # vision_model_func=vision_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            func=lambda texts: ollama_embed(texts, embed_model="nomic-embed-text")
        )
    )

    # Dokument verarbeiten
    await rag.process_document_complete(
        file_path="C:/Users/grabe_stud/RAG/RAG-Anything/docs/txt-only.pdf",
        output_dir="./output_text_only",
        parse_method="auto"
    )

    # Textanfrage
    text_result = await rag.aquery(
        "Summarize page in 5-6 sentences",
        # mode="local", 
        mode="hybrid", 
        enable_rerank=False,
        # vlm_enhanced=True,
    )
    print("\033[31mText query result: %s\033[0m" % text_result)

if __name__ == "__main__":
    asyncio.run(main())
