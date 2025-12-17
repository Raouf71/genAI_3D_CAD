TO-DO List
---

### Current-setup:
* LLM/VLM: "llama3.2-vision:11b-instruct-q4_K_M" (context length 32k)
* Embedding model: "nomic-embed-text" (embedding_dim=768, context_length 8k)
* temperature: 0.05
* Query_modes=naive/hybrid; naive is better for table-lookups
* parser: mineru

### Task-timeline:

- [x] 🛠️ Fixed ollama + RAG-I/O incompatibility:
    * Method `ollama_model_complete()`, which was recommended by RAG-Anything/LightRAG community to pass as `llm_model_func` argument in `RAGAnything()` class instantiation, fails to call ollama_client successfully (a lot of users reporting the issue on github)

    &rarr; **Solutions:**
    - [x] Wrote our custom ollama LLM and VLM methods
    - [x] After investigation: found out that the list `history_messages` (which contains the sub-processes sent my RAG-Anything) is not normalized -> we wrote our custom method (check `_normalize_rag_messages_for_ollama()`)
    - [x] Based on robust prompt engineering (check this out https://www.promptingguide.ai/research/rag), we defined the exact role for the **system** and **user** role in our LLM `ollama_llm_model()` wrapper (90% of the work is happening in there).
    - [x] For `ollama_llm_model()`, we merge: 
        1) raganything_system_prompt (containing the query specific context) + 
        3) user_prompt (actual query)
        4) history_messages (unfinished subprocesses)
    - [x] For `vision_model_func()`, we do the same but with **three modes**:
        1) direct messages path: RAGAnything already built multimodal messages
        2) multimodal path: direct text + image_data
        3) fallback: pure text – delegate to text LLM wrapper `ollama_llm_model()`
    
        and feed then to &rarr; `ollama.chat()` inside method.

### Difficulty level: &starf; &starf; &starf; &starf; &star;
### Time spent: 2,5 Weeks

---
## At this stage the RAG-Anything pipeline was working ✅ (but with noisy output ... 🚧)

- [x] Testing table-lookups(within same page):
    - ❌ Hallucination; LLM is either:
        1) making up "smooth" values, which are not present in the table
        2) copying values from row i.e. ZZ=12 when asked about i.e. ZZ=28

    #### &rarr; 🛠️ FIX 1: Made our system_prompt from `ollama_llm_model()` wrapper stricter:

    ```python
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
    ```

    #### &rarr; 🛠️ FIX 2: RAG-Anything/LightRAG config/hyperparameters tuning
    - Augment LLM/VLM context window from 8k to 32k (set `num_ctx: 32768`)
    - Set `max_context_tokens=32000` (Maximum number of tokens in extracted context(LLM/VLM/EMBEDDING))
    - Set `context_mode="chunk"` &rarr; makes every element in the page a separate text chunk
    - Set `context_filter_content_types = ["text"] ` &rarr; filter image and table chunks (since all chunks are text chunks now)
    - Set `temperature: 0.05` (almost deterministic)
    - Set `vlm_enhanced=False` (Doesn't work with ollama self-hosted models)

    #### &rarr; 🛠️ FIX 3: Converted table output format from HTML to JSON
    - LLMs work better on structured data (especially JSON), so they don't get lost
    - Wrote our custom `PROMPTS["table_chunk_json"].format()` code using **BeautifulSoup4** library, that takes in HTML and spits out JSON format. (under RAG-Anything/raganything/processor.py)    

    #### &rarr; 🛠️ FIX 4: Made our system_prompt ultra-strict (added SPECIAL CASE)
    ```python
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
        (e.g. "full row for ZZ/Z2=25 teeth", "give me the row where ZZ/Z2 = 25", etc.):

        1) Scan the CONTEXT for any block that starts with the literal prefix:
            "TABLE_AS_JSON:" followed by a JSON object.

        2) For each such block:
            - Ignore any prose around it.
            - Mentally parse the JSON that follows "TABLE_AS_JSON:".
            - Inside that JSON, look under:
            - "table" -> "columns": list of column/header names.
            - "table" -> "rows": list of row objects (each row is a JSON object).

        3) From the user query, identify:
            - The requested column name (the token before "=", e.g. "Z2" in "Z2=65").
            - The requested value (the token after "=", e.g. "65" in "Z2=65").

            Match the requested column name to one of the entries in "columns"
            (case-insensitive). If there is no exact match, choose the most obviously
            corresponding header (for example "Z2" vs "ZZ" vs similar),
            but do NOT change or approximate the numeric value.

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
        

#### &rarr; THEN TESTED TABLE-LOOKUPS AGAIN:
- [x] Tested (**positive test cases**) largest/densest table:
    - The ultra-strict prompt (from FIX4) is **not as good as expected** ❌;
        - for some queries, it outputs the exact row and only that row
        - for other(a lot) queries, it outputs the entire table
        &rarr; The ***all-or-nothing*** prompt from FIX4 made the LLM unconfident
    - ✅ The mode-strict (from FIX1) gives a correct output **ALWAYS**.

- [x] Tested (**positive test cases**) small + mid size tables with the moderate prompt from FIX1: 
    - ALL ROWS ARE CORRECT ✅

- [x] Tested (**negative test cases**) in small, mid and large size tables with the moderate prompt from FIX1: 
    - **ALL ROWS ARE FALSE** ❌:
        - The LLM doesn't output "There is no such row for ZZ=29"
        - It outputs the row-values of a neighboring row (embedding issue/limitation)

    #### &rarr; We suggest 3 solutions:
    - [ ] Tune system_prompt with specific case, where row is missing (maybe give an example query)
    - [ ] Use a better embedding model (with higher embedding dimension: fine-grained representation of neighboring values/row (max. dimension is 1024 on ollama, where the model "text-embedding-3-large" from openAI, which was suggested from RAG-Anything, has embedding_dim=3072))
    - [ ] Use LLM tool-calling for dense tables

### Difficulty level: &starf; &starf; &starf; &starf; &star;
### Frustration level: &starf; &starf; &starf; &starf; &starf; &starf; &starf; &starf; &starf; &starf;
### Time spent: 3 Weeks

---
## Today (17.12.2025)



- [ ] Testing table-lookups(two separate tables within same page):
- [ ] Testing table-lookups(two pages with same gear type but different module):
- [ ] Testing table-lookups(table with reference to another page(**test document-level retrieval**)):

- [ ] Test **llama3.1:8b-instruct** LLM for faster responses in `ollama_llm_model()` wrapper
- [ ] Test **bge-m3** model for better embeddings (especially with dense tables with rows having similar/near values)
- [ ] Add reranker model (purify context at query time + get rid of discarded elements)

 VLM, embedding models for response quality

- [ ] Evaluation metrics for each sub-task (parsing, chunking, embedding, query, etc ..)
- [ ] Implement chatbot for our RAG:
    - Upload docs (get parsed, chunked, embedded, stored in vector databases)
    - Define roles (describe, retrieve, explain old answers, remebering, etc ..)
