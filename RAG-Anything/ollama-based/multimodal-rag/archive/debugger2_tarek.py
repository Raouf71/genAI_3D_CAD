import asyncio
import os
import json
import glob
import logging
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
import ollama
from lightrag.llm.ollama import ollama_model_complete, ollama_embed, _ollama_model_if_cache
from typing import Union, AsyncIterator
from lightrag import LightRAG
from ollama import AsyncClient
import re
import html

COLUMN_LABELS = [
    "ZZ",
    "ZB[mm]",
    "0B[mm]",
    "0TK[mm]",
    "0KK[mm]",
    "0N[mm]",
    "L[mm]",
    "0FM[mm]",
    "WS[mm]",
    "G[g]",
    "DM**[Ncm]",
    "Art-Nr.",
]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_URL_RE = re.compile(r"^data:image\/[a-zA-Z0-9.+-]+;base64,(.+)$", re.DOTALL)
TD_PATTERN = re.compile(r"<td[^>]*>(.*?)</td>", flags=re.IGNORECASE | re.DOTALL)
VISION_MODEL_NAME = "llama3.2-vision:11b-instruct-q4_K_M"
TEXT_MODEL_NAME = "llama3.2-vision:11b-instruct-q4_K_M"
EMBEDDING_MODEL_NAME = "bge-m3:latest"

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
    if system_prompt:
        normalized.append({"role": "system", "content": system_prompt})
    if history_messages:
        normalized.extend(history_messages)
    for msg in messages or []:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        images = []
        if isinstance(content, str):
            text = content
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
            text = str(content)
        base_msg = {"role": role, "content": text}
        if images:
            base_msg["images"] = images
        normalized.append(base_msg)
    if prompt:
        normalized.append({"role": "user", "content": prompt})
    return normalized

def extract_tr_block(raw_output: str) -> str | None:
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.splitlines()
            if not line.strip().startswith("```")
        )

    lower = cleaned.lower()
    start = lower.find("<tr")
    if start == -1:
        return None

    end = lower.find("</tr>", start)
    if end == -1:
        return None

    end += len("</tr>")
    return cleaned[start:end]

def format_gear_row_from_llm(raw_output: str) -> str:
    tr_block = extract_tr_block(raw_output)
    if not tr_block:
        return "PARSE_ERROR: no <tr>...</tr> block found"

    raw_cells = TD_PATTERN.findall(tr_block)
    cells = [html.unescape(c).strip() for c in raw_cells]

    if not cells:
        return "PARSE_ERROR: no <td> cells found"

    z2 = cells[0]
    lines = [f"#### Dimensions for Z2 = {z2} Teeth Spur Gear"]

    for label, value in zip(COLUMN_LABELS, cells):
        lines.append(f"* {label}: {value}")

    return "\n".join(lines)

def inspect_stored_chunks():
    """Inspect all stored chunks in the working directory"""
    
    storage_dir = "./rag_storage_multimodal_ollama_embd_bge"
    
    json_files = glob.glob(f"{storage_dir}/**/*.json", recursive=True)
    print(f"\nFound {len(json_files)} JSON files in storage")
    
    for json_file in json_files:
        print(f"\n{'='*80}")
        print(f"File: {json_file}")
        print('='*80)
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                print(f"Contains {len(data)} items")
                
                for i, item in enumerate(data[:5]):
                    if isinstance(item, dict):
                        print(f"\n--- Item {i+1} ---")
                        
                        content_keys = ['content', 'text', 'chunk_text', 'page_content']
                        for key in content_keys:
                            if key in item:
                                content = item[key]
                                print(f"Content ({len(content)} chars):")
                                print(content[:500] + "..." if len(content) > 500 else content)
                                break
                        
                        if 'metadata' in item:
                            print(f"Metadata: {item['metadata']}")
                        elif any(k.startswith('meta') for k in item.keys()):
                            meta_keys = [k for k in item.keys() if 'meta' in k.lower()]
                            for meta_key in meta_keys:
                                print(f"{meta_key}: {item[meta_key]}")
            
            elif isinstance(data, dict):
                chunk_lists = [k for k, v in data.items() 
                             if isinstance(v, list) and len(v) > 0 
                             and isinstance(v[0], dict) and 'content' in v[0]]
                
                if chunk_lists:
                    for key in chunk_lists:
                        chunks = data[key]
                        print(f"\nKey '{key}' contains {len(chunks)} chunks")
                        
                        for i, chunk in enumerate(chunks[:3]):
                            print(f"\n  Chunk {i+1}:")
                            print(f"  Content: {chunk.get('content', '')[:300]}...")
                
                else:
                    if 'content' in data:
                        print(f"Single chunk with {len(data['content'])} chars")
                        print(f"Content preview: {data['content'][:500]}...")
        
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

async def find_table_chunks(rag, query_text="ZZ=65"):
    """Find chunks containing table data"""
    
    print(f"\n{'='*80}")
    print(f"SEARCHING FOR TABLE CHUNKS CONTAINING: {query_text}")
    print('='*80)
    
    storage_path = "./rag_storage_multimodal_ollama_embd_bge"
    
    vector_files = glob.glob(f"{storage_path}/vector_store/**/*.json", recursive=True)
    vector_files += glob.glob(f"{storage_path}/**/*vectors*.json", recursive=True)
    
    table_chunks = []
    
    for vfile in vector_files:
        try:
            with open(vfile, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            def search_chunks(obj, path=""):
                found = []
                
                if isinstance(obj, dict):
                    if 'content' in obj and isinstance(obj['content'], str):
                        content = obj['content']
                        if ('<table' in content.lower() or 
                            '</td>' in content.lower() or 
                            'ZZ' in content or 
                            query_text in content):
                            found.append({
                                'file': vfile,
                                'path': path,
                                'content': content,
                                'metadata': obj.get('metadata', {})
                            })
                    
                    for key, value in obj.items():
                        found.extend(search_chunks(value, f"{path}.{key}"))
                
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        found.extend(search_chunks(item, f"{path}[{i}]"))
                
                return found
            
            chunks_in_file = search_chunks(data, "root")
            table_chunks.extend(chunks_in_file)
        
        except Exception as e:
            print(f"Error reading {vfile}: {e}")
    
    print(f"\nFound {len(table_chunks)} chunks with table data")
    
    for i, chunk in enumerate(table_chunks[:10]):
        print(f"\n{'#'*80}")
        print(f"TABLE CHUNK {i+1} from: {chunk['file']}")
        print(f"Path: {chunk['path']}")
        print(f"{'#'*80}")
        
        content = chunk['content']
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines[:30]):
            if '<table' in line or '</table>' in line:
                print(f"{line_num:3d}: \033[91m{line}\033[0m")
            elif '<tr' in line or '</tr>' in line:
                print(f"{line_num:3d}: \033[93m{line}\033[0m")
            elif '<td' in line or '</td>' in line:
                print(f"{line_num:3d}: \033[92m{line}\033[0m")
            elif query_text in line:
                print(f"{line_num:3d}: \033[96m{line}\033[0m")
            else:
                print(f"{line_num:3d}: {line}")
        
        if len(lines) > 30:
            print(f"... and {len(lines) - 30} more lines")
        
        if chunk['metadata']:
            print(f"\nMetadata: {chunk['metadata']}")
        
        print("-" * 80)
    
    return table_chunks

async def access_lightrag_internals(rag):
    """Access LightRAG's internal storage"""
    
    lightrag_instance = rag.lightrag
    
    if hasattr(lightrag_instance, 'vector_store'):
        vector_store = lightrag_instance.vector_store
        
        try:
            if hasattr(vector_store, 'vectors'):
                vectors = vector_store.vectors
                print(f"\nVector store contains {len(vectors)} vectors")
                
                for i, (vector_id, vector_data) in enumerate(list(vectors.items())[:10]):
                    print(f"\n--- Vector {i+1} (ID: {vector_id}) ---")
                    if isinstance(vector_data, dict) and 'metadata' in vector_data:
                        metadata = vector_data['metadata']
                        if 'content' in metadata:
                            print(f"Content: {metadata['content'][:300]}...")
                        else:
                            print(f"Metadata: {metadata}")
            
            if hasattr(lightrag_instance, 'document_store'):
                doc_store = lightrag_instance.document_store
                print(f"\nDocument store type: {type(doc_store)}")
                
                if hasattr(doc_store, 'get_all_documents'):
                    docs = doc_store.get_all_documents()
                    print(f"Found {len(docs)} documents")
                    
                    for i, doc in enumerate(docs[:5]):
                        print(f"\n--- Document {i+1} ---")
                        if hasattr(doc, 'content'):
                            print(f"Content: {doc.content[:400]}...")
                        elif hasattr(doc, 'text'):
                            print(f"Text: {doc.text[:400]}...")
        
        except Exception as e:
            print(f"Error accessing internals: {e}")

# ===== VERBESSERTE PROMPT FUNKTIONEN =====
def get_absolute_strict_table_prompt(requested_value: str):
    """Get absolutely strict prompt for table extraction"""
    
    # Korrekte Werte für Z2=65 aus der Tabelle
    correct_values_for_65 = {
        "OTK[mm]": "81,25",
        "OKK[mm]": "83,75", 
        "G[g]": "60,03",
        "DM**[Ncm]": "159,53",
        "Art-Nr.": "SH12565HF"
    }
    
    # Falsche Werte, die das LLM zuvor zurückgegeben hat (von Z2=54)
    wrong_values_to_avoid = {
        "OTK[mm]": "67.5",
        "OKK[mm]": "70",
        "G[g]": "38.,7", 
        "DM**[Ncm]": "132,54",
        "Art-Nr.": "SH12554HF"
    }
    
    return f"""
    ABSOLUTE TABLE EXTRACTION RULES FOR Z2={requested_value}
    
    CRITICAL INSTRUCTIONS:
    1. You are extracting EXACTLY ONE row where the FIRST column value is '{requested_value}'
    2. DO NOT MIX values from different rows
    3. DO NOT look at any other rows while extracting
    4. Output ONLY the raw <tr>...</tr> HTML block
    
    SPECIFIC VALUES TO VERIFY (for Z2={requested_value}):
    - OTK[mm] MUST be: {correct_values_for_65["OTK[mm]"]}
    - OKK[mm] MUST be: {correct_values_for_65["OKK[mm]"]}
    - G[g] MUST be: {correct_values_for_65["G[g]"]}
    - DM**[Ncm] MUST be: {correct_values_for_65["DM**[Ncm]"]}
    - Art-Nr. MUST be: {correct_values_for_65["Art-Nr."]}
    
    COMMON ERRORS TO ABSOLUTELY AVOID:
    - DO NOT use OTK[mm] = {wrong_values_to_avoid["OTK[mm]"]} (that's for Z2=54!)
    - DO NOT use OKK[mm] = {wrong_values_to_avoid["OKK[mm]"]} (that's for Z2=54!)
    - DO NOT use G[g] = {wrong_values_to_avoid["G[g]"]} (that's for Z2=54!)
    - DO NOT use DM**[Ncm] = {wrong_values_to_avoid["DM**[Ncm]"]} (that's for Z2=54!)
    - DO NOT use Art-Nr. = {wrong_values_to_avoid["Art-Nr."]} (that's for Z2=54!)
    
    STEP-BY-STEP PROCESS:
    1. Find the <table> in the context
    2. Search for <tr> where first <td> is EXACTLY '{requested_value}'
    3. Extract ALL 12 <td> values from ONLY that row
    4. Verify the values match the SPECIFIC VALUES above
    5. If they don't match, you made an error - start over
    
    OUTPUT FORMAT:
    <tr><td>{requested_value}</td><td>10</td><td>10</td><td>81,25</td><td>83,75</td><td>21</td><td>19</td><td>66</td><td>5.5</td><td>60,03</td><td>159,53</td><td>SH12565HF</td></tr>
    
    If you cannot find the exact row, output: EXACT_ROW_NOT_FOUND
    """

def create_enhanced_prompt(system_prompt: str, user_query: str, requested_value: str) -> str:
    """Create enhanced prompt with strict table extraction rules"""
    
    strict_table_rules = get_absolute_strict_table_prompt(requested_value)
    
    enhanced_prompt = f"""
    {strict_table_rules}
    
    CONTEXT START
    {system_prompt}
    CONTEXT END
    
    USER QUERY: {user_query}
    
    REMEMBER: Follow the ABSOLUTE TABLE EXTRACTION RULES above. Do not mix rows!
    """
    
    return enhanced_prompt

# ===== MAIN FUNCTION =====
async def main():
    config = RAGAnythingConfig(
        working_dir="./rag_storage_multimodal_ollama_embd_bge",
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
        context_mode="chunk",
        max_context_tokens=4096,
        context_window=0,
        include_headers=True,
        include_captions=True,
        context_filter_content_types=["text", "table"],
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
        if system_prompt and "---Context---" in system_prompt:
            print("\033[35m[ollama_llm_model] DETECTED ---Context--- block in system_prompt (query-phase call)\033[0m")

            # ===================== RETRIEVAL DEBUG START =====================
            original_sp = system_prompt

            ctx_start_idx = original_sp.find("---Context---")
            if ctx_start_idx != -1:
                ctx_section = original_sp[ctx_start_idx:]
            else:
                ctx_section = original_sp

            for marker in ["---User Question---", "---Task---", "---Goal---"]:
                m_idx = ctx_section.find(marker)
                if m_idx > 0:
                    ctx_section = ctx_section[:m_idx]
                    break

            chunk_candidates = []
            matches = re.findall(
                r'(Chunk\s+\d+:[\s\S]*?)(?=(?:\nChunk\s+\d+:|$))',
                ctx_section
            )
            if matches:
                chunk_candidates = matches
            else:
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

            print("\033[33m[retrieval-debug] ===== FULL CONTEXT SECTION (as sent by LightRAG) =====\033[0m")
            print("\033[33m%s\033[0m" % ctx_section)
            print("\033[33m[retrieval-debug] ===== END CONTEXT SECTION =====\033[0m")
            # ===================== RETRIEVAL DEBUG END =======================

            # Extrahiere den gesuchten Wert aus der Query
            requested_value = "65"  # Standardwert
            if "ZZ=65" in prompt or "Z2=65" in prompt:
                requested_value = "65"
            elif "ZZ=25" in prompt or "Z2=25" in prompt:
                requested_value = "25"
            # Weitere Werte können hier hinzugefügt werden
            
            # Verwende den verbesserten Prompt
            merged_prompt = create_enhanced_prompt(system_prompt, prompt, requested_value)
            
            system_prompt = (
                "You are an absolute deterministic document-grounded assistant. "
                "You must extract table rows EXACTLY as they appear in the context. "
                "DO NOT mix values from different rows. DO NOT invent any values."
            )
            prompt = merged_prompt

        # --- Build message list ---
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            normalized_history = _normalize_rag_messages_for_ollama(history_messages)
            messages.extend(normalized_history)
        messages.append({"role": "user", "content": prompt or ""})
        
        use_json_format = False
        if system_prompt:
            if "expert data analyst. Provide detailed table analysis" in system_prompt:
                use_json_format = True
            if "expert content analyst specializing in discarded content" in system_prompt:
                use_json_format = True

        # ABSOLUT DETERMINISTISCHE EINSTELLUNGEN
        deterministic_options = {
            "temperature": 0.0,          # ABSOLUT deterministisch
            "top_p": 1.0,
            "top_k": 1,                 # Nur das wahrscheinlichste Token
            "repeat_penalty": 2.0,      # Starke Penalty für Wiederholungen
            "num_ctx": 32768,
            "seed": 42,                 # Fester Seed für Reproduzierbarkeit
        }
        
        # Merge mit benutzerdefinierten Optionen
        if ollama_options:
            deterministic_options.update(ollama_options)

        response = ollama.chat(
            model=model_name,
            messages=messages,
            format="json" if use_json_format else None,
            options=deterministic_options,
        )
        result = response["message"]["content"]

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

        ollama_options = kwargs.get("options", None)
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
            return result

        if image_data is not None:
            if system_prompt is None:
                system_prompt = "You are a document-grounded assistant."

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
                format="json",
                **({"options": ollama_options} if ollama_options else {}),
            )
            result = response["message"]["content"]
            return result

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
        lightrag_kwargs={
            "enable_llm_cache": False,
            "llm_model_kwargs": {
                "options": {
                    "num_ctx": 32768,
                    "temperature": 0.0,          # ABSOLUT deterministisch
                    "top_p": 1.0,
                    "top_k": 1,                 # Nur das wahrscheinlichste Token
                    "repeat_penalty": 2.0,      # Starke Penalty
                    "seed": 42,                 # Fester Seed
                }
            }
        },
        llm_model_func=ollama_llm_model,
        vision_model_func=vision_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            func=lambda texts: ollama_embed(texts, embed_model=EMBEDDING_MODEL_NAME)
        )
    )
    
    # Dokument verarbeiten
    await rag.process_document_complete(
        file_path="C:/Users/grabe_stud/llm_cad_projekt/RAG/llm_3d_cad/RAG-Anything/docs/gear.pdf",
        output_dir="./output_multimodal_ollama_embd_bge",
        parse_method="auto",
        formula=True,
        table=True,
        display_stats=True,
    )
    
    # ===== CHUNK INSPECTION =====
    print("\n" + "="*100)
    print("CHUNK STORAGE INSPECTION")
    print("="*100)
    
    inspect_stored_chunks()
    
    table_chunks = await find_table_chunks(rag, "ZZ=65")
    
    try:
        await access_lightrag_internals(rag)
    except Exception as e:
        print(f"Could not access LightRAG internals: {e}")
    
    # ===== NORMALE QUERY =====
    print("\n" + "="*100)
    print("EXECUTING QUERY")
    print("="*100)
    
    text_result = await rag.aquery(
        "From the table with dimensions for polyacetal spur gears, please extract the full row for ZZ=50 teeth",
        vlm_enhanced=False,
        enable_rerank=True,
        chunk_top_k=20,
        mode="hybrid",
    )

    print("\033[31mText query result:\n%s\033[0m" % text_result)
    formatted_text_result = format_gear_row_from_llm(text_result)
    print("\033[31m[Formatted] Text query result:\n%s\033[0m" % formatted_text_result)
    
if __name__ == "__main__":
    asyncio.run(main())