import asyncio
import logging
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
import ollama
from lightrag.llm.ollama import ollama_model_complete, ollama_embed, _ollama_model_if_cache
import re
from lightrag.rerank import jina_rerank

# Ensure logging is set up
logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_URL_RE = re.compile(r"^data:image\/[a-zA-Z0-9.+-]+;base64,(.+)$", re.DOTALL)
TD_PATTERN = re.compile(r"<td[^>]*>(.*?)</td>", flags=re.IGNORECASE | re.DOTALL)
VISION_MODEL_NAME = "llama3.2-vision:11b-instruct-q4_K_M"
TEXT_MODEL_NAME = "llama3.2-vision:11b-instruct-q4_K_M"
EMBEDDING_MODEL_NAME = "nomic-embed-text"

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
        working_dir="./rag_storage_ollama_debugger",
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
        content_format="minerU",        # Default content format for context extraction     
        context_mode="chunk",
        max_context_tokens=16000,       # enough for all relevant context(full table chunk etc.) and < num_ctx of model
        context_window=0,               # just that page
        include_headers = True,         # Include headers for technical content
        include_captions = True,        # Include captions for images/tables
        context_filter_content_types = ["text", "table"]      
    )

    async def ollama_llm_model(
        prompt: str,
        system_prompt: str = None,
        history_messages=None,
        **kwargs,
    ):
        if history_messages is None:
            history_messages = []
        model_name = kwargs.get("model", TEXT_MODEL_NAME)
        ollama_options = kwargs.get("options", None)

        # --- LightRAG / RAGAnything context-detection shortcut ---
        # If system_prompt already contains the big "---Context---" block,
        # we merge context + query into a single user message.
        if system_prompt and "---Context---" in system_prompt:
            print("\033[35m[ollama_llm_model] DETECTED ---Context--- block in system_prompt (query-phase call)\033[0m")

            merged_prompt = f"""
            You are a document-grounded assistant working on technical PDFs for mechanical components
            (e.g. spur gears, shafts, bearings).

            You are given CONTEXT extracted from the PDF. CONTEXT may include:
            - OCR'd tables with dimensions and properties
            - Text paragraphs (materials, fits, tolerances, notes)
            - Images of 2D technical drawings portraying geometry
            - Descriptions of formulas and equations

            GLOBAL RULES (MUST OBEY)
            - Use ONLY information that appears in the CONTEXT.
            - You are NOT allowed to invent, modify, or "correct" any numeric value.
            - Do NOT use external standards, formulas, or domain knowledge.
            - Do NOT interpolate, smooth, or approximate values.
            - If you output a numeric value, it MUST literally appear somewhere in the CONTEXT
            (same digits, commas/dots, units, spacing).
            - If the information needed to answer the question is missing or incomplete, answer EXACTLY:
            CANNOT_FIND_ANSWER_IN_DOCUMENT
            and do not guess.

            OUTPUT FORMAT FOR GENERAL QUESTIONS
            - When listing dimensions, use bullets:
            - <symbol or header name> = <value> <unit> — <short description>

            CONTEXT START
            {system_prompt}
            CONTEXT END

            User query:
            {prompt or ""}
            """

            # --- Log the Context Being Passed ---
            # logger.debug(f"[ollama_llm_model] Merged system prompt (with CONTEXT):\n{merged_prompt}")

            system_prompt = (
                "You are a deterministic, document-grounded assistant. "
                "You must treat the CONTEXT as the only source of truth and you must never invent, "
                "modify, or reformat numeric values. When the STRICT TABLE ROW COPY rules apply, "
                "you MUST follow them exactly."
            )
            prompt = merged_prompt

        # --- Build message list ---
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        # Normalize any RAG-style history messages (list content, images, etc.)
        if history_messages:
            normalized_history = _normalize_rag_messages_for_ollama(history_messages)
            messages.extend(normalized_history)
        # Current user turn
        messages.append({"role": "user", "content": prompt or ""})

        # --- Log the Final Message Being Sent to LLM ---
        logger.debug(f"[ollama_llm_model] Final message being sent to LLM:\n{messages}")

        # --- Call Ollama (text-only path here) ---
        use_json_format = False
        if system_prompt:
            if "expert data analyst. Provide detailed table analysis" in system_prompt:
                use_json_format = True
            if "expert content analyst specializing in discarded content" in system_prompt:
                use_json_format = True

        response = ollama.chat(
            model=model_name,
            messages=messages,
            format="json" if use_json_format else None,
            **({"options": ollama_options} if ollama_options else {}),
        )
        result = response["message"]["content"]
        print("\033[34m[ollama_llm_model] raw model response:\n%s\033[0m" % result)
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

        # Allow passing Ollama options via kwargs
        ollama_options = kwargs.get("options", None)
        # 1) Direct messages path: RAGAnything already built multimodal messages
        if messages:
            final_messages = _normalize_rag_messages_for_ollama(
                messages=messages,
                system_prompt=None,
                history_messages=None,
                prompt=None,  # prompt already embedded into messages
            )

            # --- Log the final message for vision model ---
            logger.debug(f"[vision_model_func] Final multimodal message being sent to vision model:\n{final_messages}")

            response = ollama.chat(
                model=VISION_MODEL_NAME,
                messages=final_messages,
                **({"options": ollama_options} if ollama_options else {}),
            )
            result = response["message"]["content"]
            print("\033[34m[vision_model_func] <if messages> raw model response:\n%s\033[0m" % result)
            return result

        # 2) Multimodal path: direct text + image_data (if you ever use it)
        if image_data is not None:
            if system_prompt is None:
                system_prompt = SYSTEM_PROMPT  # reuse your global SYSTEM_PROMPT

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

            # --- Log the final multimodal message with images ---
            logger.debug(f"[vision_model_func] Final multimodal message with images being sent to vision model:\n{final_messages}")

            response = ollama.chat(
                model=VISION_MODEL_NAME,
                messages=final_messages,
                format="json",  # image analyzer is always expected to return JSON
                **({"options": ollama_options} if ollama_options else {}),
            )
            result = response["message"]["content"]
            print("\033[34m[vision_model_func] <if image_data is not None> raw model response:\n%s\033[0m" % result)
            return result

        # 3) Fallback: pure text – delegate to your text LLM wrapper
        try:
            logger.debug("[vision_model_func] Text-only fallback: delegating to ollama_llm_model")
            result = await ollama_llm_model(
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )
            return result
        except Exception as e:
            logger.error(f"[vision_model_func] txt-only fallback error: {e}", exc_info=True)
            return f"Error in vision_model_func txt-only path: {str(e)}"
      
    rag = RAGAnything(
        config=config,
        lightrag_kwargs = {
                            "rerank_model_func": jina_rerank,  # Reranker
                            # "enable_llm_cache": False,
                           "llm_model_kwargs": 
                                {"options": {
                                        "num_ctx": 32768,
                                        "temperature": 0.05,   # almost deterministic
                                        "top_p": 0.9,          # reduce tail sampling
                                    }
                                }
                           },

        llm_model_func=ollama_llm_model,
        vision_model_func=vision_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(texts, embed_model=EMBEDDING_MODEL_NAME)
        )
    )

    # --- Log the embedding context before processing the document ---
    logger.debug("[embedding context] Embedding context for document:")
    # (Assuming the `rag.process_document_complete` will internally create the embeddings)
    # Here we would add debug logs based on your internal implementation

    # Process document
    await rag.process_document_complete(
        file_path="C:/Users/grabe_stud/llm_cad_projekt/RAG/llm_3d_cad/RAG-Anything/docs/bevel_gear.pdf",
        output_dir="./output_ollama_debugger",
        parse_method="auto",
        formula=True,                # Enable formula parsing
        table=True,                  # Enable table parsing
        display_stats=True,          # Display content statistics
    )

    # User query
    text_result = await rag.aquery(
        "From the table with dimensions for bevel gears made of zinc (ZnAl4Cu1), please extract the full table row where Art.-Nr. = KD3516-1:1ZN",
        vlm_enhanced=False,
        enable_rerank=True,
        chunk_top_k=9,      
        mode="naive",
    )

    # --- Log the embedding context for the user query ---
    logger.debug("[embedding context] Embedding context for user query:")
    # Here we could inspect the actual embedding context used during the query as well.

    # Print result
    print("\033[31mText query result:\n%s\033[0m" % text_result)


if __name__ == "__main__":
    import os; 
    
    asyncio.run(main())
