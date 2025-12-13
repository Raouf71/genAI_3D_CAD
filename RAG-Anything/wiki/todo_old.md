* kg_weight / use_kg=False

* mineru params

* options = {
    "temperature": 0.2,
    "num_ctx": 8192,
}

* chunk_top_k=20,        # make sure enough raw chunks come in

* reranker


Advanced Querying Techniques

* Multi-mode querying for better results
async def comprehensive_query(question):
    results = {}
    
    # Try different modes
    modes = ["naive", "local", "global", "hybrid"]
    
    for mode in modes:
        try:
            result = await rag.aquery(question, mode=mode)
            results[mode] = result
        except Exception as e:
            results[mode] = f"Error: {e}"
    
    # Return best result (hybrid usually wins)
    return results.get("hybrid", results.get("global", "No results"))
answer = await comprehensive_query("Explain the main research findings")