import asyncio
import os
import logging
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
import ollama
from lightrag.llm.ollama import ollama_model_complete, ollama_embed, _ollama_model_if_cache
from typing import Union, AsyncIterator
from lightrag import LightRAG
from ollama import AsyncClient
import re
from lightrag.rerank import cohere_rerank, jina_rerank

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a document-grounded assistant. Use ONLY the provided context from the PDF and its extracted images, tables and equations. If you cannot find the requested value as literal text in the context, you MUST answer exactly NOT_IN_CONTEXT for that value and you MUST NOT approximate, infer, or calculate it"
DATA_URL_RE = re.compile(r"^data:image\/[a-zA-Z0-9.+-]+;base64,(.+)$", re.DOTALL)
VISION_MODEL_NAME = "llama3.2-vision:11b-instruct-q4_K_M"
TEXT_MODEL_NAME = "qwen2.5:32b-instruct-q4_K_M"
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
        working_dir="./rag_storage_multimodalclager_ragtest",
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=False,

        context_mode="page",
        context_window=0,          # just that page
        max_context_tokens=3000,   # allow bigger context if needed
        # context_mode = "chunk",         # Process content by chunk
        # context_window = 5,             # Include 5 pages of context
        # max_context_tokens = 2000,      # Keep the context size manageable
        
        include_headers = True,         # Include headers for technical content
        include_captions = True,        # Include captions for images/tables
        context_filter_content_types = ["text", "table", "image"]  # Include relevant content types
    )

    async def ollama_llm_model(
        prompt: str,
        system_prompt: str = None,
        history_messages=None,
        **kwargs,
    ):
        # DEBUG: high-level call info (retrieval relevant)
        print("\033[36m[ollama_llm_model] ===== CALL =====\033[0m")
        sp_preview = (system_prompt or "")
        pr_preview = (prompt or "")
        print("\033[34m[ollama_llm_model] incoming system_prompt :\n%s\033[0m" % sp_preview)
        print("\033[34m[ollama_llm_model] incoming user prompt :\n%s\033[0m" % pr_preview)

        if history_messages is None:
            history_messages = []

        # Optional: model + options override
        # NOTE: use TEXT_MODEL_NAME as default for all text-only calls
        model_name = kwargs.get("model", TEXT_MODEL_NAME)
        ollama_options = kwargs.get("options", None)

        # DEBUG: basic meta
        print("\033[34m[ollama_llm_model] model_name=%s, has_options=%s, history_len=%d\033[0m"
              % (model_name, bool(ollama_options), len(history_messages or [])))

        # --- LightRAG / RAGAnything context-detection shortcut ---
        # If system_prompt already contains the big "---Context---" block,
        # we merge context + query into a single user message.
        if system_prompt and "---Context---" in system_prompt:
            print("\033[35m[ollama_llm_model] DETECTED ---Context--- block in system_prompt (query-phase call)\033[0m")

            # DEBUG: scan the provided CONTEXT for table lines containing "ZZ"
            ctx_text = system_prompt
            zz_lines = [ln for ln in ctx_text.splitlines() if "ZZ" in ln or "Z Z" in ln]
            print("\033[36m[ollama_llm_model] CONTEXT lines containing 'ZZ':\033[0m")
            for line in zz_lines:
                print("\033[36m  %s\033[0m" % line)

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
            - Do NOT use external standards, formulas, or domain knowledge to invent new numbers.
            - Do NOT "correct", interpolate, or smooth values.
            - If you output a numeric value, it MUST literally appear in the CONTEXT
              (same digits, commas/dots, units).
            - If the information needed to answer the question is missing or incomplete, answer EXACTLY:
              CANNOT_FIND_ANSWER_IN_DOCUMENT
              and briefly state which piece of information is missing.

            TASK
            - Read the user query below.
            - Search the CONTEXT (tables, drawings, text, equations) for information relevant to the query.
            - You may combine information from tables, drawings, equations, and text,
              but NEVER create new numerical values.
            - Prefer precise table entries and dimension annotations from drawings over vague text.

            OUTPUT
            - Answer in Markdown.
            - When listing dimensions, use bullets:
              - <symbol or name> = <value> <unit> — <short description>
            - If the user asks for a specific table row (e.g. “ZZ = 25”),
              copy that row EXACTLY (header + row) and then list the columns as bullets.
            - If any cell is empty or “-”, output `X` for that cell.

            CONTEXT START
            {system_prompt}
            CONTEXT END

            User query:
            {prompt or ""}
            """
            system_prompt = (
                "You are a document-grounded assistant. "
                "Follow the instructions inside the user message and NEVER invent numbers."
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

        # DEBUG: inspect final messages before calling Ollama (retrieval-relevant)
        print("\033[34m[ollama_llm_model] final messages count=%d\033[0m" % len(messages))
        if messages:
            fm = messages[0]
            fm_text = (fm.get("content", "") or "")
            print("\033[34m[ollama_llm_model] first message role=%s, content:\n%s\033[0m"
                  % (fm.get("role"), fm_text))
        if len(messages) > 1:
            lm = messages[-1]
            lm_text = (lm.get("content", "") or "")
            print("\033[34m[ollama_llm_model] last (user) message content:\n%s\033[0m"
                  % lm_text)

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

        # Keep this function quiet-ish: we care now mostly about retrieval, not multimodal JSON guts
        print("\033[36m[vision_model_func] CALL (multimodal preprocessing)\033[0m")

        # Allow passing Ollama options via kwargs
        ollama_options = kwargs.get("options", None)

        # ------------------------------------------------------------------
        # 1) Direct messages path: RAGAnything already built multimodal messages
        # ------------------------------------------------------------------
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
        llm_model_func=ollama_llm_model,
        vision_model_func=vision_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            # max_token_size=8192,
            func=lambda texts: ollama_embed(texts, embed_model=EMBEDDING_MODEL_NAME)
        ),
        
    )
    # Dokument verarbeiten
    await rag.process_document_complete(
        file_path="C:/Users/grabe_stud/llm_cad_projekt/RAG/llm_3d_cad/RAG-Anything/docs/gear.pdf",
        # file_path="C:/Users/grabe_stud/llm_cad_projekt/RAG/llm_3d_cad/RAG-Anything/docs/lager.pdf",
        output_dir="./output_multimodal_ragtest",
        parse_method="auto",
        # MinerU special parameters - all supported kwargs:
        # lang="en",                   # Document language for OCR optimization (e.g., "ch", "en", "ja")
        formula=False,                # Enable formula parsing
        table=True,                  # Enable table parsing
        display_stats=True,          # Display content statistics
    )
    # Textanfrage
    text_result = await rag.aquery(
        "From the table with dimensions for polyacetal spur gears, please extract the full row for ZZ=25 teeth",
        # "Für einen Kegelrollenlager nach DIN720, extrahiere alle erforderlichen Abmessungen für einen Durchmesser d=40",
        # mode="hybrid",
        vlm_enhanced=False,
        enable_rerank=True,
        chunk_top_k=20,        # make sure enough raw chunks come in
        mode="hybrid",
    )

    print("\033[31mText query result:\n%s\033[0m" % text_result)

if __name__ == "__main__":
    asyncio.run(main())
