import asyncio
import logging
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc, logger
import ollama
from lightrag.llm.ollama import ollama_model_complete, ollama_embed, _ollama_model_if_cache
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Get the logger from the lightrag module
logger_lightrag = logging.getLogger("lightrag")
# Enable propagation to the root logger
logger_lightrag.propagate = True
# Set logging level to DEBUG
logger_lightrag.setLevel(logging.DEBUG)

# SYSTEM_PROMPT = "You are a document-grounded assistant. Use ONLY the provided context from the PDF and its extracted images, tables and equations. If you cannot find the requested value as literal text in the context, you MUST answer exactly NOT_IN_CONTEXT for that value and you MUST NOT approximate, infer, or calculate it"
DATA_URL_RE = re.compile(r"^data:image\/[a-zA-Z0-9.+-]+;base64,(.+)$", re.DOTALL)
TD_PATTERN = re.compile(r"<td[^>]*>(.*?)</td>", flags=re.IGNORECASE | re.DOTALL)
VISION_MODEL_NAME = "llama3.2-vision:11b-instruct-q4_K_M"
TEXT_MODEL_NAME = "llama3.2-vision:11b-instruct-q4_K_M"
# TEXT_MODEL_NAME = "qwen2.5:7b-instruct"
EMBEDDING_MODEL_NAME = "nomic-embed-text"
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
        working_dir="./rag_storage_ollama_diff_m",
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
        content_format="minerU",
        context_mode="chunk",                 # keep chunk mode for deterministic retrieval behavior
        max_context_tokens=28000,             # leave headroom for system/tools/user prompt under num_ctx=32768
        context_window=0,                     # include the hit page + 1 neighbor page (covers 2 pages well)
        include_headers=True,
        include_captions=True,
        context_filter_content_types = ["text"]     # for chunk-based context      
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

            # ===================== RETRIEVAL DEBUG START =====================
            original_sp = system_prompt

            # 1) Try to isolate the CONTEXT section inside the system_prompt
            ctx_start_idx = original_sp.find("---Context---")
            if ctx_start_idx != -1:
                ctx_section = original_sp[ctx_start_idx:]
            else:
                ctx_section = original_sp  # fallback, should not happen if we are in this branch

            # 2) Optional: try to cut off anything after a possible "User question" / "Task" section
            #    This depends on LightRAG's template, so we do it best-effort and non-destructively.
            for marker in ["---User Question---", "---Task---", "---Goal---"]:
                m_idx = ctx_section.find(marker)
                if m_idx > 0:
                    # keep up to the marker (context only)
                    ctx_section = ctx_section[:m_idx]
                    break

            # 3) Try to detect "chunks" inside the context text for nicer debug printing.
            #    We don't change logic; this is purely visualization.
            #    We look for lines starting with something like "Chunk 1:", "[1]", etc.
            chunk_candidates = []
            # a) Pattern for "Chunk 1:", "Chunk 2:", etc.
            matches = re.findall(
                r'(Chunk\s+\d+:[\s\S]*?)(?=(?:\nChunk\s+\d+:|$))',
                ctx_section
            )
            if matches:
                chunk_candidates = matches
            else:
                # b) Pattern for "[1] ", "[2] " style numbering
                matches = re.findall(
                    r'(\[\d+\][\s\S]*?)(?=(?:\n\[\d+\]\s|$))',
                    ctx_section
                )
                if matches:
                    chunk_candidates = matches

            if chunk_candidates:
                print("\033[34m[retrieval-debug] ===== RETRIEVED CHUNKS (from context) =====\033[0m")
                for i, ch in enumerate(chunk_candidates):
                    print("\033[34m[retrieval-debug] CHUNK_%d:\n%s\033[0m" % (i, ch))
            else:
                print("\033[34m[retrieval-debug] No explicit chunk markers detected, printing raw context only.\033[0m")

            # 4) Print FULL context section in yellow (this is what the model actually sees)
            print("\033[33m[retrieval-debug] ===== FULL CONTEXT SECTION (as sent by LightRAG) =====\033[0m")
            print("\033[33m%s\033[0m" % ctx_section)
            # 🔍 NEW: check how many tokens the final context has
            # tokens = rag.tokenizer.encode(ctx_section)
            # print(
            #     f"\033[33m[retrieval-debug] Context tokens: {len(tokens)} / limit: {rag.max_total_tokens}\033[0m"
            # )
            print("\033[33m[retrieval-debug] ===== END CONTEXT SECTION =====\033[0m")
            # ===================== RETRIEVAL DEBUG END =======================

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
        print(f"[ollama_llm_model] Final message being sent to LLM:\n{messages}")
        
        # --- Call Ollama (text-only path here) ---
        # Decide if this call must return strict JSON (for table/discarded analyzers)
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

        # DEBUG: show full raw model response
        print("\033[34m[ollama_llm_model] raw model response:\n%s\033[0m" % result)
        print("==============================")
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
            response = ollama.chat(
                model=VISION_MODEL_NAME,
                messages=final_messages,
                **({"options": ollama_options} if ollama_options else {}),
            )
            result = response["message"]["content"]
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

            response = ollama.chat(
                model=VISION_MODEL_NAME,
                messages=final_messages,
                format="json",  # image analyzer is always expected to return JSON
                **({"options": ollama_options} if ollama_options else {}),
            )
            result = response["message"]["content"]
            return result

        # 3) Fallback: pure text – delegate to your text LLM wrapper
        try:
            print("\033[33m[vision_model_func] txt-only fallback -> delegating to ollama_llm_model\033[0m")
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
                            "enable_llm_cache": True,
                            "llm_model_kwargs": 
                                {"options": {
                                        "num_ctx": 32768,
                                        "temperature": 0.05,   # almost deterministic
                                        "top_p": 0.9,          # reduce tail sampling
                                        # "repeat_penalty": 1.05 # mild, to avoid weird loops
                                    }
                                }
                           },

        llm_model_func=ollama_llm_model,
        vision_model_func=vision_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            # max_token_size=8192,
            func=lambda texts: ollama_embed(texts, embed_model=EMBEDDING_MODEL_NAME)
        )
    )

    # Process document
    await rag.process_folder_complete(
        folder_path="./docs",
        output_dir="./output_ollama_diff_m",
        recursive=True,
        parse_method="auto",
        display_stats=True,          # Display content statistics
    )

    # specify module
    text_result = await rag.aquery(
        query = (
            "From the table with dimensions for polyacetal spur gears of module 1,0, "
            f"please extract the full row for Z2=12 teeth"
        ),
        vlm_enhanced=False,
        enable_rerank=False,
        mode="naive",
    )

    print("\033[31mText query result:\n%s\033[0m" % text_result)


if __name__ == "__main__":
    import os; 
    
    asyncio.run(main())
