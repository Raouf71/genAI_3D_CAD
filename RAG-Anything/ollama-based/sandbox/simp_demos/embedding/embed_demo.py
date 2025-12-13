import ollama

resp = ollama.embeddings(
    model='nomic-embed-text',
    prompt='hello world',
)
print(len(resp['embedding']), resp['embedding'][:5])
