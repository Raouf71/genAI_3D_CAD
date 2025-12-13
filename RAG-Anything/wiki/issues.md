SOLUTIONS:
---

1. Keep the whole table in one chunk

Ensure the entire table is in a single chunk (like now), not split across chunks.

2. Make the task explicitly “copy raw text”

- Ask: “Find the <tr> where Z2=25 and output that exact <tr>...</tr> HTML line, no changes, no formatting, no Markdown.”
- If the user asks for a specific table row (e.g. “Z2 = 25”):
    - Search the CONTEXT for the HTML <tr>...</tr> line where the first <td> is 25.
    - Copy that <tr>...</tr> substring exactly as it appears (no changes, no reformatting).
- Add explicit anti-“smartness” instructions for numbers
    - “Do not recompute or recombine any numbers.”
    - “You are not allowed to use numbers from other rows.”
    - If you cannot find a <tr> whose first <td> is exactly 25, answer NOT_IN_CONTEXT.

3. Ban transformations in the prompt

Add: “You MUST NOT change any numbers, commas, dots, or spacing. If you cannot find the exact row, output EXACTLY: NOT_IN_CONTEXT.”

4. Use a smaller extraction-style / JSON model

Call a more “tool-like” model (e.g. instruction: “Return JSON with the exact row text from the table_body string”), then render on your side.

5. (Best but breaks your ‘LLM-only’ rule) Deterministic select

-Parse table_body as HTML in Python and:
    - Find <tr> where first <td> == "25".
    - Return that row directly → 100% correct, no hallucination.

5. (Optional) Two-step verification with the LLM

- Step 1: LLM outputs the candidate row.
- Step 2: Ask the LLM: “Is this exact row (character by character) present in the provided table_body? Answer YES or NO only.”

If “NO”, treat as failure on your side.