TO-DO List
---

### Closed:
- [x] Fixed ollama+RAG I/O incompatibility via custom method (**_normalize_rag_messages_for_ollama()**)
- [x] 
- [x] Fixed a LOT of system+user prompt phrasing (prompt engineering: check this out https://www.promptingguide.ai/research/rag) 
- [x] Defined robust system+user roles for VLM and LLM
- [x] Fixed unmerged prompts (floating context) for LLM

### Open:
- [ ] Test multiple LLM, VLM, embedding models for response quality

- [ ] evalutate/fix chunking technique and params (context_mode, context_window, max_context_tokens, context_filter_content_types, static or dynamic chunking)

- [ ] Add reranker model

- [ ] test aquery_with_multimodal()

- [ ] Evaluation metrics for each sub-task (parsing, chunking, embedding, etc ..)

