# Llamaindex RAG

## Setup

1. Create separate virtual env for llamaindex then activate it
```bash
python -m venv ./llamaindex-env
source ./llamaindex-env/bin/activate
```

2. Install requirements
```bash
pip install llama-index llama-index-core==0.10.42 llama-index-embeddings-openai llama-index-postprocessor-flag-embedding-reranker git+https://github.com/FlagOpen/FlagEmbedding.git llama-index-graph-stores-neo4j llama-cloud-services 
```

Or do:
```bash
pip install -r requirements.txt
```


## Usage

For a demo using **API-Keys**, please refer to:
- `gear_diff_m.ipynb` for gear_diff_m.pdf
- `lager.ipynb` for lager.pdf
- `drehmoment_s30.ipynb` for drehmoment.pdf

All three notebooks test document descriptions, table-lookups and memory ability. It uses a naive retriever (using vector search only), then a custom retriever (combining vector and Knowledge Graph search). Then we leverage the pipeline capability by introducing **agentic RAG**. <br>

### Agentic RAG

Instead of calling a basic LLM to answer the question based on the context, agentic RAG now relies on an agent that operates with a thinking mode being on using models trained to **"think before acting"**. It works as follows:

1. **Planning**: The agent analyzes the user's complex query and breaks it down into sub-tasks.
2. **Tool Use**: The agent selects and uses different tools, such as:
    * Vector Search for unstructured data.
    * Knowledge Graph search.
    * Web Search for real-time info.
    * Table-lookup tool etc.
3. **Retrieval & Reasoning**: It retrieves data from chosen sources, potentially refining searches or validating information.
4. **Generation & Self-Critique**: An LLM synthesizes the gathered information, while the agent might review the output and decide if another retrieval step is needed (like a closed loop)

## Useful links
Check out [this](https://www.llamaindex.ai/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5) blog from llamaindex, where they investigated the ideal chunk size for a RAG system. See the following table for the results:

| Chunk size | Average Response Time (s) | Average Faithfulness | Average Relevancy |
|------------|---------------------------|----------------------|-------------------|
| 128        | 1.55                      | 0.85                 | 0.78              |
| 256        | 1.57                      | 0.90                 | 0.78              |
| 512        | 1.68                      | 0.85                 | 0.85              |
| 1024       | 1.68                      | **0.93**                 | **0.90**              |
| 2048       | **1.72**                      | 0.90                 | 0.89              |

&rarr; This suggests that a chunk size of **1024** might strike an optimal balance between response time and the quality of the responses, measured in terms of faithfulness and relevancy.