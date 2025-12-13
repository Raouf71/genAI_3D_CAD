import asyncio
import logging
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
import ollama
from lightrag.llm.ollama import ollama_model_complete, ollama_embed, _ollama_model_if_cache
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        working_dir="./rag_storage_sandbox_table_lookup",
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
        # context_filter_content_types = ["text"]     # for chunk-based context      
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

            SPECIAL CASE: STRICT TABLE ROW COPY (MUST FOLLOW EXACTLY)
            - If the user asks for a specific table row to be "copied", "returned", or "extracted"
            (e.g. "full row for Z/ZZ/Z2=25 teeth", "give me the row where Z/ZZ/Z2 = 25", "give me the row where Art.-Nr. = KD1016-1:1ZN", etc.):

            1) Scan the CONTEXT for any block that starts with the literal prefix:
                "TABLE_AS_JSON:" followed by a JSON object.

            2) For each such block:
                - Ignore any prose around it.
                - Mentally parse the JSON that follows "TABLE_AS_JSON:".
                - Inside that JSON, look under:
                - "table" -> "columns": list of column/header names.
                - "table" -> "rows": list of row objects (each row is a JSON object).

            3) From the user query, identify:
                - The requested column name (the token before "=", e.g. "Z2" in "Z2=65" or "Art.-Nr." in "Art.-Nr.= KD1016-1:1ZN").
                - The requested value (the token after "=", e.g. "65" in "Z2=65" or "KD1016-1:1ZN" in "Art.-Nr.= KD1016-1:1ZN").

                Match the requested column name to one of the entries in "columns"
                (case-insensitive). If there is no exact match, return NO_EXACT_KEY_MATCH.

            4) Search ONLY within "table.rows" of the TABLE_AS_JSON blocks:
                - Find all rows where row[column_name] EXACTLY equals the requested value
                as a string (same digits and punctuation, e.g. "65" vs "65,0").
                - Do NOT pick “closest” or “similar” values. Exact match only.

            5) If you find one or more matching rows:
                - OUTPUT ONLY the JSON for the matching row object(s).
                - Preserve all keys and string values exactly as they appear in the table
                (same spelling, digits, commas, dots, and units).
                - You may output a single JSON object or a JSON array of row objects.
                - Do NOT wrap the output in Markdown, tables, code fences, or any extra text.
                - Do NOT explain the result. Just output the raw JSON object(s).

            6) If you do NOT find any matching row in ANY TABLE_AS_JSON block:
                - Output EXACTLY: NOT_IN_CONTEXT
                - Do NOT construct a plausible row, do NOT guess, and do NOT reuse another row
                as a template.

            GENERAL TASK (NON-STRICT CASES)
            - For all other questions:
            - Read the user query.
            - Search the CONTEXT (tables, drawings, text, equations) for information relevant to the query.
            - You may combine information from tables, drawings, equations, and text,
                but NEVER create new numerical values.
            - Prefer precise table entries and dimension annotations from drawings over vague text.

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
                            "enable_llm_cache": False,
                           "llm_model_kwargs": 
                                {"options": {
                                        "num_ctx": 32768,
                                        "temperature": 0.05,   # almost deterministic
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

    # Process document
    await rag.process_document_complete(
        file_path="C:/Users/grabe_stud/llm_cad_projekt/RAG/llm_3d_cad/RAG-Anything/docs/sheep.jpg",
        output_dir="./output_sandbox_table_lookup",
        parse_method="auto",
        formula=True,                # Enable formula parsing
        table=True,                  # Enable table parsing
        display_stats=True,          # Display content statistics
    )

    multimodal_result = await rag.aquery_with_multimodal(
        # "Describe the table in 2-3 sentences",
        "From the following table please extract the full row for Art.-Nr. = KD2016-1:1ZN",
        multimodal_content=[
            {
                "type": "table",
                "table_data": {
                    "columns": [
                    "M",
                    "z",
                    "B[mm]|",
                    "ON[mm]",
                    "OTK[mm]",
                    "OKK[mm]",
                    "E[mm]",
                    "NL[mm]",
                    "ZB",
                    "Q",
                    "L|[mm]|",
                    "G[g]",
                    "DM**[Ncm]",
                    "Art.-Nr."
                    ],
                    "rows": [
                    {
                        "M": "1,0",
                        "z": "16",
                        "B[mm]|": "6",
                        "ON[mm]": "12",
                        "OTK[mm]": "16",
                        "OKK[mm]": "17,7",
                        "E[mm]": "17,9",
                        "NL[mm]": "7.5",
                        "ZB": "4.5",
                        "Q": "13",
                        "L|[mm]|": "13",
                        "G[g]": "7",
                        "DM**[Ncm]": "21,82",
                        "Art.-Nr.": "KD1016-1:1ZN"
                    },
                    {
                        "M": "1,5",
                        "z": "16",
                        "B[mm]|": "8",
                        "ON[mm]": "19",
                        "OTK[mm]": "24",
                        "OKK[mm]": "26",
                        "E[mm]": "25,2",
                        "NL[mm]": "10,7",
                        "ZB": "6,9",
                        "Q": "17",
                        "L|[mm]|": "18,6",
                        "G[g]": "27",
                        "DM**[Ncm]": "73,13",
                        "Art.-Nr.": "KD1516-1:1ZN"
                    },
                    {
                        "M": "2,0",
                        "z": "16",
                        "B[mm]|": "10",
                        "ON[mm]": "23",
                        "OTK[mm]": "32",
                        "OKK[mm]": "34,8",
                        "E[mm]": "30",
                        "NL[mm]": "10",
                        "ZB": "9,6",
                        "Q": "19,2",
                        "L|[mm]|": "21,3",
                        "G[g]": "52",
                        "DM**[Ncm]": "185,77",
                        "Art.-Nr.": "KD2016-1:1ZN"
                    },
                    {
                        "M": "2.5",
                        "z": "16",
                        "B[mm]|": "12",
                        "ON[mm]": "26",
                        "OTK[mm]": "40",
                        "OKK[mm]": "43.3",
                        "E[mm]": "36,2",
                        "NL[mm]": "12",
                        "ZB": "12,3",
                        "Q": "23",
                        "L|[mm]|": "25,5",
                        "G[g]": "88",
                        "DM**[Ncm]": "357,06",
                        "Art.-Nr.": "KD2516-1:1ZN"
                    },
                    {
                        "M": "3,0",
                        "z": "16",
                        "B[mm]|": "14",
                        "ON[mm]": "30",
                        "OTK[mm]": "48",
                        "OKK[mm]": "52,3",
                        "E[mm]": "42,5",
                        "NL[mm]": "13",
                        "ZB": "14",
                        "Q": "26",
                        "L|[mm]|": "29,3",
                        "G[g]": "146",
                        "DM**[Ncm]": "576,86",
                        "Art.-Nr.": "KD3016-1:1ZN"
                    },
                    {
                        "M": "3.5",
                        "z": "16",
                        "B[mm]|": "16",
                        "ON[mm]": "34",
                        "OTK[mm]": "56",
                        "OKK[mm]": "61,4",
                        "E[mm]": "49,2",
                        "NL[mm]": "14",
                        "ZB": "15,5",
                        "Q": "29,2",
                        "L|[mm]|": "33,2",
                        "G[g]": "228",
                        "DM**[Ncm]": "898,94",
                        "Art.-Nr.": "KD3516-1:1ZN"
                    }
                    ]
                },
                # "table_data": 
                # "table_data": """Method,Accuracy,Processing_Time
                #             RAGAnything,95.2%,120ms
                #             Traditional_RAG,87.3%,180ms
                #             Baseline,82.1%,200ms""",
                "table_caption": "gear dimensions",
            }
        ],
        vlm_enhanced=False,
        enable_rerank=False,
        mode="hybrid",
    )
    logger.info(f"Multimodal result Answer: {multimodal_result}")
    

if __name__ == "__main__":
    import os; 
    
    asyncio.run(main())
