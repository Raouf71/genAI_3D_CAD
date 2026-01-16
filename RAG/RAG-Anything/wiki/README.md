What Happens process_document_complete():
---
When you call the method, RAG-Anything does this:
1. Parse → Content saved to output_dir (optional storage)
2. Extract → Text, images, tables extracted
3. Embed → Creates embeddings + knowledge graph
4. Store → Graph + embeddings saved to working_dir

For a detailed overview, check out the following section:
<details>
    1. Parsing (MinerU)

    - process_document_complete() calls MinerU with parser="mineru", method="auto"
    - MinerU outputs a JSON list of blocks: text, tables, images, etc. This is stored in output_multimodal/... (this is just parsed content, not what is used for search directly).

    1. Content separation + multimodal source setup

    RAG-Anything:
        - Counts block types, separates pure text vs multimodal items (tables, images, equations).
        - Stores this parsed content as the “content source” for context-aware multimodal processing.
        - This step writes things like gear_content_list.json into output_dir, not the RAG store.

    2. Text → chunks + KG

    - Text-only content is passed to LightRAG.
        - LightRAG splits the text into chunks (it has its own default chunk size).
        - For each chunk, it calls your ollama_llm_model() with a “extract entities & relations” prompt.
        - The model response is turned into a knowledge graph (KG):
            - nodes = entities
            - edges = relations

    3. Embeddings → vector DB

    - Every text chunk is converted to a vector via your EmbeddingFunc → ollama_embed(nomic-embed-text)
    - These vectors are stored in the vector DB files:
        - vdb_chunks.json (for chunks)
        - vdb_entities.json (for entities)
        - vdb_relationships.json (for relations)

    - All of this lives in ./rag_storage_multimodal (the working_dir).

    3. Entities & relations → KG + their embeddings

    - The extracted entities/relations from each chunk are:
        - Normalized / merged (same entity names get merged).
        - Stored as nodes/edges in graph_chunk_entity_relation.graphml.
    - Entity and relation text descriptions are also embedded:
        - Stored in vdb_entities.json and vdb_relationships.json.

    4. Multimodal (images, tables, equations)

    - For each image/table/etc, RAG-Anything:
        - Builds a context-aware prompt (page context, captions, etc., controlled by context_mode, context_window, max_context_tokens, context_filter_content_types).
        - Calls vision_model_func() (or text LLM for tables) to get a JSON description.

        - That description is then treated like a chunk of text:
            - Passed through the same KG extraction step → entities + relations.
            - New entities/relations are merged into the same KG.
            - Multimodal entities get belongs_to links to their source chunk.
    - These “multimodal chunks” also get embeddings, contributing to vdb_chunks.json.

    5. KV stores + doc status

    - LightRAG maintains KV stores (e.g., full_docs, text_chunks, parse_cache, doc_status) to track:
        - Which docs were processed.
        - Which chunks belong to which doc.
        - Cache of LLM calls used during extraction.

    5. Query time (mode="naive" in your code)

    - Your question is embedded with nomic-embed-text.
    - LightRAG compares this vector to all chunk vectors in vdb_chunks.json.
    - It picks the top-K chunks and builds a big ---Context--- text.
    - This context plus your query is sent to ollama_llm_model() one last time.
    - The model answers only from that context (ideally, if your prompt enforces that).


    So:

    - output_multimodal = parsed raw content (for inspection/debugging).
    - rag_storage_multimodal = what actually matters for RAG (KG + vector DB + chunk store).
    - Chunking happens inside LightRAG when it takes text and splits it before KG + embeddings.
    - mode="naive" still queries the vector DB, but only uses chunk similarity (no fancy graph reasoning).

</details>

Where does chunking take place?
---
LightRAG pipeline invokes something like lightrag.insert(doc_text, ...) / insert_docs(...) 
(exact function name depends on the LightRAG version, but it’s the insert-document step).

Inside LightRAG, there is an internal function (often named similar to _split_to_chunks / _chunk_text in their codebase) that:
- takes the raw text for a doc,
- cuts it into chunks using its own default chunk size & overlap,
- saves those as text_chunks entries and creates embeddings into vdb_chunks.json.

Who decides which top-K chunks are picked?
---
LightRAG does it.
- Step 1 – Scoring:
For a given query, LightRAG scores all chunks (and sometimes entities/relations) using:
    - keyword / BM25–style matching in naive mode, or
    - vector similarity (cosine on embeddings) in other modes.

- Step 2 – Top-K selection:
It then sorts by that score and takes the best chunk_top_k chunks (the chunk_top_k you pass into rag.aquery()).

Optional rerank:
If enable_rerank=True, a reranker model re-sorts those candidates again and picks the final top-K.

What is the difference between chunks and context?
---
* Chunks = the individual stored pieces of text in the vector store (fixed segmentation done at index time).
* Context = the subset of those chunks that were retrieved for this query (top-K, plus any neighbors/pages), concatenated and sent to the LLM.

How do i prompt my RAG?
---
* For table queries check this out:
    <details>
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
    (e.g. “full row for Z2=25 teeth”, “give me the row where Z2 = 25”, etc.):
    1) Locate the `table_body` string inside CONTEXT (the HTML fragment starting with <table> and ending with </table>).
    2) Inside this `table_body` HTML, search ONLY within the <tr>...</tr> rows.
    3) Find all <tr>...</tr> rows where the FIRST <td> cell exactly matches the requested key
        (e.g. 25 for “Z2=25”; do NOT approximate, do NOT use closest value).
    4) If you find one or more matching rows:
        - OUTPUT ONLY the exact characters of each matching "<tr> ... </tr>" substring from the input.
        - Preserve all characters exactly: tags, spaces, commas, dots, decimal separators, units, and line breaks.
        - Do NOT wrap the output in Markdown, tables, code fences, or any extra text.
        - Do NOT reorder columns. Do NOT explain the result. Just output the raw <tr>...</tr> line(s).
    5) If you do NOT find any matching row:
        - Output EXACTLY: NOT_IN_CONTEXT
        - Do NOT construct a plausible row, do NOT guess, and do NOT reuse another row as a template.

    GENERAL TASK (NON-STRICT CASES)
    - For all other questions:
    - Read the user query.
    - Search the CONTEXT (tables, drawings, text, equations) for information relevant to the query.
    - You may combine information from tables, drawings, equations, and text,
        but NEVER create new numerical values.
    - Prefer precise table entries and dimension annotations from drawings over vague text.

    OUTPUT FORMAT FOR GENERAL QUESTIONS
    - When listing dimensions, use bullets:
    - <symbol or name> = <value> <unit> — <short description>

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
    </details>