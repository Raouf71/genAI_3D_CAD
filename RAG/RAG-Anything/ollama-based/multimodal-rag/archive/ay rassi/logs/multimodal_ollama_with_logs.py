import asyncio
import os
import logging
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
import ollama
from lightrag.llm.ollama import  ollama_embed
from lightrag.llm.ollama import ollama_model_complete, ollama_embed, _ollama_model_if_cache
from typing import Union, AsyncIterator
from lightrag import LightRAG
from ollama import AsyncClient
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a document-grounded assistant. Use ONLY the provided context from the PDF and its extracted images, tables, and equations. If the answer is missing from the context, do not guess and do not add external information. Respond with: 'Cannot find answer in the provided document'."""
DATA_URL_RE = re.compile(r"^data:image\/[a-zA-Z0-9.+-]+;base64,(.+)$", re.DOTALL)

VISION_MODEL_NAME = "llama3.2-vision:11b-instruct-q4_K_M"
# VISION_MODEL_NAME = "ingu627/Qwen2.5-VL-7B-Instruct-Q5_K_M"
EMBEDDING_MODEL_NAME = "nomic-embed-text"
# EMBEDDING_MODEL_NAME = "bge-m3"
# EMBEDDING_MODEL_NAME = "mxbai-embed-large"

def _extract_base64_from_data_url(url: str) -> str:
    m = DATA_URL_RE.match(url.strip())
    return m.group(1) if m else url.strip()

def _normalize_rag_messages_for_ollama(
    messages,
    system_prompt=None,
    history_messages=None,
    prompt=None,
):
    normalized = []
    # Optional extra system_prompt, mostly for non-VLM paths
    if system_prompt:
        normalized.append({"role": "system", "content": system_prompt})
    if history_messages:
        normalized.extend(history_messages)
    for msg in messages or []:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        images = []
        # Case 1: content is already simple string
        if isinstance(content, str):
            text = content
        # Case 2: OpenAI-style multimodal content list
        elif isinstance(content, list):
            text_chunks = []
            for part in content:
                ptype = part.get("type")
                if ptype == "text":
                    text_chunks.append(part.get("text", ""))
                elif ptype == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url:
                        images.append(_extract_base64_from_data_url(url))
            text = "".join(text_chunks)
        else:
            # Fallback: stringify
            text = str(content)
        base_msg = {"role": role, "content": text}
        if images:
            base_msg["images"] = images
        normalized.append(base_msg)
    # If there is an extra plain prompt, append as final user turn
    if prompt:
        normalized.append({"role": "user", "content": prompt})
    return normalized

async def main():
    config = RAGAnythingConfig(
        working_dir="./rag_storage_multimodall",
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
        context_window=5,  # Include 5 pages of context
        context_mode="chunk",  # Process content by chunk
        max_context_tokens=2000,  # Keep the context size manageable
        include_headers=True,  # Include headers for technical content
        include_captions=True,  # Include captions for images/tables
        context_filter_content_types=["text", "table", "image"],  # Include relevant content types
    )

    async def ollama_llm_model(
        prompt: str,
        system_prompt: str = None,
        history_messages=None,
        **kwargs,
    ):
        if history_messages is None:
            history_messages = []
        model_name = kwargs.get("model", VISION_MODEL_NAME)
        ollama_options = kwargs.get("options", None)
        # Merging context + query into a single user message if system_prompt contains "---Context---"
        if system_prompt and "---Context---" in system_prompt:
            merged_prompt = (
                "You are given a document in the sections below.\n"
                "Use ONLY that content to answer the user's question.\n\n"
                + system_prompt
                + "\n\nUser query: "
                + (prompt or "")
            )
            system_prompt = (
                "You are a helpful assistant that MUST answer "
                "only using the document text included in the user message."
            )
            prompt = merged_prompt
        # Build message list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            normalized_history = _normalize_rag_messages_for_ollama(history_messages)
            messages.extend(normalized_history)
        messages.append({"role": "user", "content": prompt or ""})

        # Log the prompt and system prompt
        logger.info(f"\033[34mSystem Prompt: {system_prompt}\033[0m")
        logger.info(f"\033[34mUser Prompt: {prompt}\033[0m")

        response = ollama.chat(
            model=model_name,
            messages=messages,
            **({"options": ollama_options} if ollama_options else {}),
        )

        result = response["message"]["content"]
        return result

    async def vision_model_func(
        prompt: str,
        system_prompt: str = None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ):
        if history_messages is None:
            history_messages = []
        ollama_options = kwargs.get("options", None)

        # Check if messages are already provided
        if messages:
            final_messages = _normalize_rag_messages_for_ollama(
                messages=messages,
                system_prompt=None,
                history_messages=None,
                prompt=None,
            )

            response = ollama.chat(
                model=VISION_MODEL_NAME,
                messages=final_messages,
                **({"options": ollama_options} if ollama_options else {}),
            )
            result = response["message"]["content"]
            print("\033[31m[vision_model_func] response (messages-path) received.\033[0m")
            return result

        # Multimodal fallback (text + image data)
        if image_data is not None:
            if system_prompt is None:
                system_prompt = SYSTEM_PROMPT
            if isinstance(image_data, (list, tuple)):
                images = list(image_data)
            else:
                images = [image_data]
            system_message = {"role": "system", "content": system_prompt}
            user_message = {
                "role": "user",
                "content": prompt or "",
                "images": images,
            }
            final_messages = [system_message] + (history_messages or []) + [user_message]
            response = ollama.chat(
                model=VISION_MODEL_NAME,
                messages=final_messages,
                **({"options": ollama_options} if ollama_options else {}),
            )
            result = response["message"]["content"]
            print("\033[31m[vision_model_func] response (multimodal-path) received.\033[0m")
            return result

        # Fallback to text model if no image data
        print("\033[33m=================== txt-only fallback =============================\033[0m")
        try:
            result = await ollama_llm_model(
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )
            print("\033[31m[vision_model_func] txt-only result received.\033[0m")
            return result
        except Exception as e:
            logger.error(f"[vision_model_func] txt-only fallback error: {e}", exc_info=True)
            return f"Error in vision_model_func txt-only path: {str(e)}"

    rag = RAGAnything(
        config=config,
        lightrag_kwargs={"llm_model_kwargs": {"options": {"num_ctx": 32768}}},
        llm_model_func=ollama_llm_model,
        vision_model_func=vision_model_func,
        embedding_func=EmbeddingFunc(
            # embedding_dim=1024,
            embedding_dim=768,
            func=lambda texts: ollama_embed(texts, embed_model=EMBEDDING_MODEL_NAME)
        )
    )

    # Process document
    await rag.process_document_complete(
        file_path="C:/Users/grabe_stud/RAG/RAG-Anything/docs/gear.pdf",
        output_dir="./output_multimodall",
        parse_method="auto",
        
        # MinerU special parameters - all supported kwargs:
        lang="en",                   # Document language for OCR optimization (e.g., "ch", "en", "ja")
        # device="cuda:0",             # Inference device: "cpu", "cuda", "cuda:0", "npu", "mps"
        # start_page=0,                # Starting page number (0-based, for PDF)
        # end_page=10,                 # Ending page number (0-based, for PDF)
        formula=True,                # Enable formula parsing
        table=True,                  # Enable table parsing
        # backend="pipeline",          # Parsing backend: pipeline|vlm-transformers|vlm-sglang-engine|vlm-sglang-client.
        # source="huggingface",        # Model source: "huggingface", "modelscope", "local"
        # vlm_url="http://127.0.0.1:3000" # Service address when using backend=vlm-sglang-client

        # Standard RAGAnything parameters
        display_stats=True,          # Display content statistics
        # split_by_character=None,     # Optional character to split text by
        # doc_id=None                  # Optional document ID
    )

    # Text query with improvements
    text_result = await rag.aquery(
        "From the table with dimensions for polyacetal spur gears, please extract the full row for ZZ=65 teeth",
        mode="naive",
        vlm_enhanced=False,
        enable_rerank=False,
    )
    print("\033[31mText query result: %s\033[0m" % text_result)

if __name__ == "__main__":
    asyncio.run(main())
