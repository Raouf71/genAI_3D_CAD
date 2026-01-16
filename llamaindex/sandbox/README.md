# Document processing using LlamaIndex framework

Think of it as:
LlamaIndex turns **raw documents** into clean text, then into **smart/meaningful chunks**, stores them in a **searchable memory**, and retrieves the **best ones** when needed.

|    Step    |         Purpose         |  LlamaIndex Objects  |
|:----------:|:-----------------------:|:--------------------:|
| Parsing    | Read files into text    | Document, LlamaParse |
| Splitting  | Cut text into chunks    | Node, TextSplitter   |
| Extracting | Pull structured facts   | Pydantic schemas     |
| Indexing   | Store chunks for search | VectorStoreIndex     |
| Retrieval  | Fetch relevant chunks   | Retriever            |

## 1. Parsing (https://developers.llamaindex.ai/python/cloud/llamaparse/features/parsing_options/)

You start with messy stuff: PDFs, Word files, webpages, scanned documents.
This step opens those files and turns them into readable text, sometimes cleaning layouts, tables, or weird formatting. At the end of this step, the system has clean text objects, not files anymore.

LlamaIndex technical view
* Goal: Raw data → Document objects
* Main tools:
    - SimpleDirectoryReader
    - File-specific loaders (PDF, HTML, etc.)
    - LlamaParse (advanced parsing for PDFs, tables, complex layouts)
* Output:
    - Document(text=..., metadata=...)

📌 **Key concept: Parsing = understanding the file format and extracting usable text.**

- Parse into Nodes:
convert Document objects into Node objects (nodes are the chunk-level units used by indexes/retrievers).

The NodeParser API literally says: get_nodes_from_documents(...) = “Parse documents into nodes.”

The “Documents / Nodes” guide also states you can “parse source Documents into Nodes through our NodeParser classes.”


## 2a. Metadata normalization
Metadata normalization is just cleaning + standardizing the “labels” attached to each document/chunk so they’re consistent and reliable later.

What actually happens:
* Read whatever metadata exists (doc.metadata) from the loader/parser (often inconsistent: page, page_number, missing source, etc.).
* Create a consistent set of keys you will rely on everywhere, e.g.:
    * source_path (full path or URL)
    * source_file (filename)
    * source_id (stable document identifier)
    * page_number (integer)
    * optional: doc_type, created_at, etc.

* Convert types + fill missing values

* ensure page_number is an int

* if the parser didn’t provide page info, assign a fallback (like loop index + 1)

* Write the cleaned dict back to doc.metadata.

-> Result: every later chunk/node carries metadata that’s uniform, so you can:
* cite sources (“lager.pdf p.17”),
* filter retrieval by page/source,
* debug and trace where an answer came from.

## 2b. Splitting 
(Breaking big text into small, digestible pieces)
Plain explanation

Large documents are too big for AI to reason about at once.
So the text is cut into smaller overlapping chunks, each small enough to fit into an LLM’s context window.

Each chunk represents one idea-sized piece of text.

* LlamaIndex technical view
* Goal: Document → multiple Node chunks
* Main tools:
    * NodeParser
    * TextSplitter
    * Common splitters:
        * TokenTextSplitter
            * Use it when Documents are long, free-flowing text (reports, articles, manuals)
            * You want fixed-size chunks that fit LLM context windows.
            * Page boundaries don’t matter as much as semantic continuity.
        * SentenceSplitter
            * You want structured Nodes, not just text chunks.
            * You plan to use advanced indexes (KG, hierarchical, multi-modal).
            * You want automatic handling of metadata → node relationships.

* Key parameters:
    * chunk_size (default ~1024 tokens)
    * chunk_overlap (default ~20 tokens)
* Method:
    * get_nodes_from_documents(documents)

📌 Key concept: Splitting = chunking text into Nodes.

## 3. Extracting (Optional but powerful)
(Pulling structured facts out of text)
Plain explanation

Sometimes you don’t just want text — you want specific facts:
* names
* dates
* numbers
* entities
* table-like data
This step asks an LLM to extract structured information and ignore everything else.

LlamaIndex technical view

* Goal: Unstructured text → structured schema
* Main tools:
    * Structured Extraction APIs
    * Pydantic models to define schemas
* Typical outputs:
    * JSON
    * Python objects
* Used when:
    * Building knowledge graphs
    * Analytics
    * Entity-aware RAG

📌 Key concept: Extraction = turning text into structured data.

## 4. Indexing

(Storing knowledge so it can be found later)
Plain explanation

Now the system stores all chunks in a smart data structure so it can quickly find the right ones when a question is asked.

This is like building a search engine memory over your documents.

LlamaIndex technical view

* Goal: Store Node objects for fast retrieval
* Core abstraction: Index
* Most common index:
    * VectorStoreIndex
* What happens internally:
    * Each node → embedding vector
    * Stored in:
        * in-memory store
        * or external vector DB (FAISS, Pinecone, Weaviate, etc.) (https://developers.llamaindex.ai/python/framework/community/integrations/vector_stores/#using-a-vector-store-as-an-index)
        * or Graph Database (Neo4j, etc.)
* Exposes:
    * Retriever interface

LlamaIndex does not automatically use FAISS / Pinecone / Weaviate unless you explicitly configure one.

In-memory vector store (SimpleVectorStore) means:
* embeddings live in RAM
* lost on process restart
* fine for small data only

### hybrid retrieval (BM25 + vectors)
* BM25 (keyword search) is great for:
    * part numbers
    * standards (DIN 5412)
    * exact symbols (d = 55, d_w)
* Vector search (embeddings similarity) is great for:
    * paraphrasing
    * vague questions
    * semantic similarity
-> Hybrid = best of both.

* Knowledge Graph search
    * stored as nodes + edges (entities/relations/paths), not just text chunks.
    * Parameters:
        * extractors
        * similarity
        * depth

📌 Key concept: Indexing = making chunks searchable.

## 5. Retrieval (Query time, not ingestion — but crucial)
(Finding the right pieces when a question comes in)
Plain explanation

When a user asks a question, the system searches the index to find the most relevant chunks and feeds them to the LLM.

LlamaIndex technical view

* Retriever types:
    * Vector similarity search
    * Hybrid retrievers
* Outputs:
    * Ranked list of relevant Nodes
* Used by:
    * Query Engines
    * Chat Engines
    * RAG pipelines

📌 Key concept: Retrieval = selecting the best chunks for answering.


<br>
<br>
<br>
<br>

---
(WORK IN PROGRESS)
* Keyword filtering (https://developers.llamaindex.ai/python/framework/module_guides/indexing/index_guide/)

* DO WE RETURN NODES OR DOCS??????????????
* Persist/reuse the index: don’t re-extract KG every run; cache parse + extracted paths; incremental updates only.
* Use a stronger extractor LLM than gpt-3.5-turbo for relation extraction (quality matters more than cost here), and log/inspect extracted triples.
* Add schema / constraints for extraction (e.g., “DIN standard”, “bearing type”, “dimensions”, “load ratings”) to reduce noisy edges.
* Tune retrieval:
    * keep path_depth=1 (good default),
    * increase similarity_top_k slightly (often 10–20) then re-rank,
    * apply metadata filters (source_file / standard) when query implies it.

* Add evaluation & guardrails:
    * unit tests on “row lookup” queries,
    * validate extracted values (numbers, units) before answering.
* Don’t rebuild embeddings twice: he instantiates OpenAIEmbedding(...) multiple times; centralize it.
---