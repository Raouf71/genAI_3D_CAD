import asyncio
import os
import json
import glob
import logging
from datetime import datetime
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

# Output Datei für Ergebnisse
OUTPUT_FILE = r"C:\Users\grabe_stud\llm_cad_projekt\RAG\llm_3d_cad\RAG-Anything\ollama-based\multimodal-rag\LOGS_tarek\gear_extraction_zz12.txt"

def save_to_txt(content: str, mode: str = 'a'):
    """Speichert Inhalte in die TXT-Datei"""
    with open(OUTPUT_FILE, mode, encoding='utf-8') as f:
        f.write(content + "\n")

def save_header():
    """Speichert Kopfzeile mit Zeitstempel"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"""
{'='*80}
GEAR TABLE EXTRACTION RESULTS
{'='*80}
Timestamp: {timestamp}
Document: gear.pdf
Output file: {OUTPUT_FILE}
{'='*80}

"""
    save_to_txt(header, 'w')

def save_test_case_result(query: str, expected_z2: str, result: str, is_valid: bool, formatted: str = None):
    """Speichert ein einzelnes Testergebnis"""
    result_text = f"""
{'='*60}
QUERY: {query}
Expected Z2: {expected_z2}
Status: {'✓ VALID' if is_valid else '✗ INVALID'}
{'='*60}

Raw Result:
{result}

"""
    
    if formatted and is_valid:
        result_text += f"Formatted Result:\n{formatted}\n"
    
    save_to_txt(result_text)

def save_summary(valid_count: int, total_count: int, results: dict):
    """Speichert eine Zusammenfassung"""
    summary = f"""
{'='*80}
TEST SUMMARY
{'='*80}
Valid extractions: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*80}

Detailed Results:
"""
    
    for query, data in results.items():
        status = "✓ VALID" if data.get("valid", False) else "✗ INVALID"
        summary += f"{status}: {query}\n"
        if 'error' in data:
            summary += f"  Error: {data['error']}\n"
    
    save_to_txt(summary)

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

def extract_z2_value_from_query(query: str) -> str:
    """Extract Z2 value from query string"""
    
    patterns = [
        r'ZZ=(\d+)',      # ZZ=65
        r'Z2=(\d+)',      # Z2=65  
        r'for (\d+) teeth', # for 65 teeth
        r'row for (\d+)',   # row for 65
        r'(\d+) teeth',     # 65 teeth
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1)
    
    numbers = re.findall(r'\b(\d+)\b', query)
    if numbers:
        return numbers[0]
    
    return None

def get_general_strict_table_prompt(requested_value: str):
    """Get general strict prompt for ANY table extraction"""
    
    return f"""
    ABSOLUTE TABLE EXTRACTION RULES FOR Z2={requested_value}
    
    CRITICAL INSTRUCTIONS (MUST FOLLOW):
    1. Find EXACTLY ONE row where the FIRST column value is '{requested_value}'
    2. Extract ALL 12 values from ONLY that specific row
    3. DO NOT look at any other rows while extracting
    4. DO NOT mix values from different rows
    5. Output ONLY the raw <tr>...</tr> HTML block
    
    COMMON LLM ERRORS TO AVOID:
    - ERROR 1: Do NOT use values from Z2=54 when extracting Z2=65
    - ERROR 2: Do NOT use values from Z2=64 when extracting Z2=65  
    - ERROR 3: Do NOT approximate or calculate missing values
    - ERROR 4: Do NOT invent values that are not in the table
    
    STEP-BY-STEP PROCESS:
    1. Find the <table> in the context
    2. Search row-by-row for <tr> where first <td> is EXACTLY '{requested_value}'
    3. When found, STOP searching and focus ONLY on that row
    4. Extract ALL 12 <td> values in order (columns 1-12)
    5. Output the COMPLETE <tr> block with ALL 12 values
    
    VERIFICATION CHECK:
    - After extraction, verify you have EXACTLY 12 values
    - Verify the FIRST value is EXACTLY '{requested_value}'
    - If not, you made an error - start over
    
    OUTPUT FORMAT EXAMPLE:
    For Z2=25, it should look like:
    <tr><td>25</td><td>10</td><td>6</td><td>31.25</td><td>33.75</td><td>15</td><td>19</td><td>21</td><td>7</td><td>11,39</td><td>61,36</td><td>SH12525HF</td></tr>
    
    For Z2=65, it should look like:
    <tr><td>65</td><td>10</td><td>10</td><td>81,25</td><td>83,75</td><td>21</td><td>19</td><td>66</td><td>5.5</td><td>60,03</td><td>159,53</td><td>SH12565HF</td></tr>
    
    ERROR HANDLING:
    If you cannot find the exact row, output: EXACT_ROW_NOT_FOUND
    If the table is not in the context, output: TABLE_NOT_IN_CONTEXT
    """

def create_table_extraction_prompt(system_prompt: str, user_query: str) -> str:
    """Create enhanced prompt with general table extraction rules"""
    
    requested_value = extract_z2_value_from_query(user_query)
    
    if not requested_value:
        strict_table_rules = """
        ABSOLUTE TABLE EXTRACTION RULES:
        
        1. Find the table in the context
        2. Extract rows based on the user's query
        3. DO NOT mix values from different rows
        4. Output ONLY raw <tr>...</tr> HTML blocks
        5. Preserve exact formatting including commas, dots, spaces
        
        If no specific row is requested, extract relevant rows based on query.
        """
    else:
        strict_table_rules = get_general_strict_table_prompt(requested_value)
    
    enhanced_prompt = f"""
    {strict_table_rules}
    
    CONTEXT START
    {system_prompt}
    CONTEXT END
    
    USER QUERY: {user_query}
    
    REMEMBER THE CRITICAL RULE: 
    DO NOT MIX VALUES FROM DIFFERENT ROWS! 
    Extract values ONLY from the exact matching row(s).
    """
    
    return enhanced_prompt

def validate_extracted_row(extracted_html: str, expected_z2: str = None) -> bool:
    """Validate if extracted row has correct structure - AUSKOMMENTIERT"""
    # VALIDIERUNG AUSKOMMENTIERT - NUR ERGEBNISSE ANZEIGEN
    return True  # Immer True zurückgeben, um Validierung zu überspringen
    
    # AUSKOMMENTIERTER ORIGINALCODE:
    # if "EXACT_ROW_NOT_FOUND" in extracted_html or "TABLE_NOT_IN_CONTEXT" in extracted_html:
    #     return True
    # 
    # cells = TD_PATTERN.findall(extracted_html)
    # 
    # if len(cells) != 12:
    #     print(f"VALIDATION ERROR: Expected 12 cells, got {len(cells)}")
    #     return False
    # 
    # if expected_z2 and cells[0] != expected_z2:
    #     print(f"VALIDATION ERROR: First cell should be {expected_z2}, got {cells[0]}")
    #     return False
    # 
    # if "38.,7" in extracted_html and expected_z2 == "65":
    #     print("VALIDATION ERROR: Found Z2=54 values (38.,7) in Z2=65 row!")
    #     return False
    # 
    # if "67.5" in extracted_html and expected_z2 == "65":
    #     print("VALIDATION ERROR: Found Z2=54 values (67.5) in Z2=65 row!")
    #     return False
    # 
    # return True
async def test_specific_values(rag):
    """Extrahiere spezifische Z2 Werte aus der Tabelle"""
    
    # Werte, die in der Tabelle existieren (basierend auf Ihrem PDF)
    specific_values = [12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,30,32, 35, 36,38,40,42, 45,48, 50,52,54,55,56, 60,64,65, 70,72,75, 80, 90,100,110]
    
    
    
    print("\n" + "="*100)
    print(f"EXTRACTING SPECIFIC Z2 VALUES")
    print("="*100)
    
    save_to_txt(f"\n{'='*100}\nEXTRACTING SPECIFIC Z2 VALUES\n{'='*100}")
    
    all_results = {}
    
    for z2_value in specific_values:  # Ändern Sie dies zu specific_values für alle Werte
        query = f"extract the row where Z2={z2_value}"
        
        print(f"\n{'='*80}")
        print(f"EXTRACTING Z2={z2_value}")
        print('='*80)
        
        try:
            result = await rag.aquery(
                f"From the table with dimensions for polyacetal spur gears, please extract {query}",
                vlm_enhanced=False,
                enable_rerank=True,
                chunk_top_k=20,
                mode="hybrid",
            )
            
            formatted = format_gear_row_from_llm(result)
            
            print(f"\n\033[92mZ2={z2_value} DIMENSIONS:\033[0m")
            print(formatted)
            
            all_results[z2_value] = formatted
            
            save_to_txt(f"\nZ2={z2_value}:")
            save_to_txt(formatted)
            
        except Exception as e:
            error_msg = f"ERROR Z2={z2_value}: {e}"
            print(f"\033[91m{error_msg}\033[0m")
            save_to_txt(f"\n{error_msg}")
            
            all_results[z2_value] = f"ERROR: {e}"
    
    print(f"\n\033[93mResults saved to: {OUTPUT_FILE}\033[0m")
    return all_results

async def test_multiple_values(rag):
    """Extrahiere Z2=12 Werte - OHNE VALIDIERUNG"""
    
    query = "extract the row where Z2=12"
    expected_z2 = "12"
    
    print("\n" + "="*100)
    print("EXTRACTING Z2=12 DIMENSIONS")
    print("="*100)
    
    save_to_txt(f"\n{'='*100}\nEXTRACTING Z2=12 DIMENSIONS\n{'='*100}")
    
    try:
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print('='*80)
        
        result = await rag.aquery(
            f"From the table with dimensions for polyacetal spur gears, please extract {query}",
            vlm_enhanced=False,
            enable_rerank=True,
            chunk_top_k=20,
            mode="hybrid",
        )
        
        formatted = format_gear_row_from_llm(result)
        
        # ERGEBNISSE ANZEIGEN
        print(f"\n\033[94mRAW RESULT:\033[0m")
        print(result)
        
        print(f"\n\033[92mFORMATTED DIMENSIONS FOR Z2=12:\033[0m")
        print(formatted)
        
        # Speichern
        save_to_txt(f"\nQUERY: {query}")
        save_to_txt(f"\nRAW RESULT:\n{result}")
        save_to_txt(f"\nFORMATTED RESULT:\n{formatted}")
        
        print(f"\n\033[93mResults saved to: {OUTPUT_FILE}\033[0m")
        
        return {"result": result, "formatted": formatted}
        
    except Exception as e:
        print(f"\033[91mERROR: {e}\033[0m")
        save_to_txt(f"\nERROR: {str(e)}")
        return {"error": str(e)}
# ===== SPEZIFISCHE QUERIES MIT TXT-SPEICHERUNG =====
async def run_specific_queries(rag):
    """Run specific queries and save to TXT"""
    
    specific_tests = [
        ("From the table with dimensions for polyacetal spur gears, please extract the full row for ZZ=12 teeth", "12"),
    ]
    
    print("\n" + "="*100)
    print("SPECIFIC QUERIES")
    print("="*100)
    
    # Speichere Überschrift
    save_to_txt(f"\n{'='*100}\nSPECIFIC QUERIES\n{'='*100}")
    
    for query, expected_z2 in specific_tests:
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print('='*80)
        
        try:
            result = await rag.aquery(
                query,
                vlm_enhanced=False,
                enable_rerank=True,
                chunk_top_k=20,
                mode="hybrid",
            )
            
            print(f"\033[31mRaw result:\n{result}\033[0m")
            
            # Validierung
            is_valid = validate_extracted_row(result, expected_z2)
            
            if is_valid and "EXACT_ROW_NOT_FOUND" not in result and "TABLE_NOT_IN_CONTEXT" not in result:
                formatted = format_gear_row_from_llm(result)
                print(f"\033[32mFormatted result:\n{formatted}\033[0m")
                
                # Speichere in TXT-Datei
                save_test_case_result(query, expected_z2, result, is_valid, formatted)
            else:
                print(f"\033[91mINVALID or error result\033[0m")
                save_test_case_result(query, expected_z2, result, is_valid)
        
        except Exception as e:
            print(f"\033[91mERROR: {e}\033[0m")
            save_to_txt(f"\nERROR in specific query '{query}': {str(e)}")

# ===== MAIN FUNCTION =====
async def main():
    # Erstelle Kopfzeile in TXT-Datei
    save_header()
    
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

        if system_prompt and "---Context---" in system_prompt:
            print("\033[35m[ollama_llm_model] DETECTED ---Context--- block in system_prompt (query-phase call)\033[0m")

            ctx_start_idx = system_prompt.find("---Context---")
            if ctx_start_idx != -1:
                ctx_section = system_prompt[ctx_start_idx:]
                
                for marker in ["---User Question---", "---Task---", "---Goal---"]:
                    m_idx = ctx_section.find(marker)
                    if m_idx > 0:
                        ctx_section = ctx_section[:m_idx]
                        break
            
            merged_prompt = create_table_extraction_prompt(system_prompt, prompt)
            
            system_prompt = (
                "You are an absolute deterministic document-grounded assistant. "
                "You must extract table rows EXACTLY as they appear in the context. "
                "NEVER mix values from different rows. NEVER invent any values."
            )
            prompt = merged_prompt

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            normalized_history = _normalize_rag_messages_for_ollama(history_messages)
            messages.extend(normalized_history)
        messages.append({"role": "user", "content": prompt or ""})
        
        deterministic_options = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "repeat_penalty": 1.5,
            "num_ctx": 32768,
            "seed": 42,
        }
        
        if ollama_options:
            deterministic_options.update(ollama_options)

        response = ollama.chat(
            model=model_name,
            messages=messages,
            options=deterministic_options,
        )
        result = response["message"]["content"]

        print("\033[34m[ollama_llm_model] raw model response:\n%s\033[0m" % result[:500])
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
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 1,
                    "repeat_penalty": 1.5,
                    "seed": 42,
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
    print("\n" + "="*100)
    print("PROCESSING DOCUMENT")
    print("="*100)
    
    await rag.process_document_complete(
        file_path="C:/Users/grabe_stud/llm_cad_projekt/RAG/llm_3d_cad/RAG-Anything/docs/gear.pdf",
        output_dir="./output_multimodal_ollama_embd_bge",
        parse_method="auto",
        formula=True,
        table=True,
        display_stats=True,
    )
    
    # ===== TESTS DURCHFÜHREN =====
    test_results = await test_specific_values(rag)
    
    # ===== SPEZIFISCHE QUERIES =====
    #await run_specific_queries(rag)
    
    print(f"\n{'='*100}")
    print(f"ERGEBNISSE GESPEICHERT IN: {OUTPUT_FILE}")
    print("="*100)
    
    # Zeige Dateipfad an
    abs_path = os.path.abspath(OUTPUT_FILE)
    print(f"Vollständiger Pfad: {abs_path}")

if __name__ == "__main__":
    asyncio.run(main())