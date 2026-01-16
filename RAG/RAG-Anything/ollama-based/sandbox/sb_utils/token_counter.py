from __future__ import annotations
from typing import Any, Dict, List, Optional
import base64
import requests

def count_tokens_ollama(
    *,
    model: str,
    prompt: str,
    base_url: str = "http://localhost:11434",
    system: Optional[str] = None,
    images: Optional[List[bytes]] = None,
    raw: bool = False,
    timeout_s: float = 60.0,
) -> int:
    """
    Returns the exact input token count as evaluated by Ollama for the given prompt.

    Uses /api/generate and reads `prompt_eval_count` from the response. :contentReference[oaicite:3]{index=3}
    Set options.num_predict=0 to avoid generating output (fast + cheap).

    images: optional list of image bytes (e.g., open('x.jpg','rb').read()) for vision models.
            For many pipelines, you should NOT manually insert <|image|> into the prompt when
            providing images; the system/tokenizer may handle it. :contentReference[oaicite:4]{index=4}
    """
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "raw": raw,  # if True, disables Ollama's prompt templating
        "options": {"num_predict": 0},
    }
    if system is not None:
        payload["system"] = system
    if images:
        payload["images"] = [base64.b64encode(b).decode("utf-8") for b in images]

    r = requests.post(f"{base_url.rstrip('/')}/api/generate", json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()

    if "prompt_eval_count" not in data:
        raise RuntimeError(f"Ollama response missing prompt_eval_count. Keys: {list(data.keys())}")

    return int(data["prompt_eval_count"])


if __name__ == "__main__":

    text = """
[{'role': 'system', 'content': 'You are a deterministic, document-grounded assistant. You must treat the CONTEXT as the only source of truth and you must never invent, modify, or reformat numeric values. When the STRICT TABLE ROW COPY rules apply, you MUST 
follow them exactly.'}, {'role': 'user', 'content': '\n            You are a document-grounded assistant working on technical PDFs for mechanical components\n            (e.g. spur gears, shafts, bearings).\n\n            You are given CONTEXT extracted from 
the PDF. CONTEXT may include:\n            - OCR\'d tables with dimensions and properties\n            - Text paragraphs (materials, fits, tolerances, notes)\n            - Images of 2D technical drawings portraying geometry\n            - Descriptions of 
formulas and equations\n\n            GLOBAL RULES (MUST OBEY)\n            - Use ONLY information that appears in the CONTEXT.\n            - You are NOT allowed to invent, modify, or "correct" any numeric value.\n            - Do NOT use external standards, 
formulas, or domain knowledge.\n            - Do NOT interpolate, smooth, or approximate values.\n            - If you output a numeric value, it MUST literally appear somewhere in the CONTEXT\n            (same digits, commas/dots, units, spacing).\n          
  - If the information needed to answer the question is missing or incomplete, answer EXACTLY:\n            CANNOT_FIND_ANSWER_IN_DOCUMENT\n            and do not guess.\n\n            OUTPUT FORMAT FOR GENERAL QUESTIONS\n            - When listing dimensions, 
use bullets:\n            - <symbol or header name> = <value> <unit> ù <short description>\n\n            CONTEXT START\n            ---Role---\n\nYou are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your 
primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.\n\n---Goal---\n\nGenerate a comprehensive, well-structured answer to the user query.\nThe answer must integrate relevant facts from the 
Document Chunks found in the **Context**.\nConsider the conversation history if provided to maintain conversational flow and avoid repeating information.\n\n---Instructions---\n\n1. Step-by-Step Instruction:\n  - Carefully determine the user\'s query intent in 
the context of the conversation history to fully understand the user\'s information need.\n  - Scrutinize `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.\n  - Weave 
the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.\n  - Track the reference_id of the document chunk which directly support 
the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.\n  - Generate a **References** section at the end of the response. Each reference document must directly 
support the facts presented in the response.\n  - Do not generate anything after the reference section.\n\n2. Content & Grounding:\n  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly 
stated.\n  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.\n\n3. Formatting & Language:\n  - The response MUST be in the same language as the user query.\n  - The response 
MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).\n  - The response should be presented in Multiple Paragraphs.\n\n4. References Section Format:\n  - The References section should be under heading: 
`### References`\n  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).\n  - The Document Title in the citation must retain its original language.\n  - Output each 
citation on an individual line\n  - Provide maximum of 5 most relevant citations.\n  - Do not generate footnotes section or any comment, summary, or explanation after the references.\n\n5. Reference Section Example:\n```\n### References\n\n- [1] Document Title 
One\n- [2] Document Title Two\n- [3] Document Title Three\n```\n\n6. Additional Instructions: n/a\n\n\n---Context---\n\n\nDocument Chunks (Each entry has a reference_id refer to the `Reference Document List`):\n\n```json\n{"reference_id": "1", "content": 
"\\nImage Content Analysis:\\nImage Path: C:\\\\Users\\\\grabe_stud\\\\llm_cad_projekt\\\\RAG\\\\llm_3d_cad\\\\RAG-Anything\\\\ollama-based\\\\multimodal-rag\\\\bevel_gear\\\\debug\\\\output_ollama_debugger\\\\bevel_gear\\\\auto\\\\images\\\\f5b13f8b5e14698d40b
a8b09088a0d831065b93f3ed8b42f3c58d745e4153cc3.jpg\\nCaptions: None\\nFootnotes: None\\n\\nVisual Analysis: The image presents a technical drawing of a bevel gear, showcasing its intricate design and components. The drawing is rendered in black and white, with 
various lines, shapes, and symbols used to convey specific information about the gear\'s dimensions, angles, and features. The gear\'s teeth are depicted in a detailed manner, with each tooth labeled and dimensioned to provide precise measurements. The drawing 
also includes annotations and labels to highlight important features, such as the gear\'s pitch diameter, face width, and tooth thickness. The overall layout is clear and organized, making it easy to follow and understand the gear\'s design. The use of 
different line styles and symbols adds to the visual clarity, allowing the viewer to quickly identify key components and features. The image provides a comprehensive and detailed visual representation of the bevel gear, making it an invaluable resource for 
engineers, designers, and anyone interested in understanding the gear\'s design and functionality."}\n{"reference_id": "1", "content": "\\nImage Content Analysis:\\nImage Path: C:\\\\Users\\\\grabe_stud\\\\llm_cad_projekt\\\\RAG\\\\llm_3d_cad\\\\RAG-Anything\\\
\ollama-based\\\\multimodal-rag\\\\bevel_gear\\\\debug\\\\output_ollama_debugger\\\\bevel_gear\\\\auto\\\\images\\\\8350a69bafa49ba9f73ed1eb25342b446c3a3c19576ebc8ed7bd6f38b7cbce66.jpg\\nCaptions: None\\nFootnotes: None\\n\\nVisual Analysis: The image depicts 
a gear with a beveled edge, likely used in a mechanical system. The gear is made of metal and has a circular shape with a series of teeth along its edge. The teeth are evenly spaced and have a slight curve to them, indicating that they are beveled. The gear is 
positioned on a flat surface, with the beveled edge facing upwards. The background of the image is a plain gray color, which helps to highlight the details of the gear. The overall composition of the image is simple and straightforward, with the gear being the 
main focus. The layout is also simple, with the gear being centered in the image and the background being a solid color. The visual elements of the image include the gear, the beveled edge, and the background. The relationships between these elements are 
straightforward, with the gear being the main object and the beveled edge being a key feature. The colors used in the image are primarily gray, with the gear having a slightly darker tone than the background. The lighting in the image is even and consistent, 
with no shadows or highlights. The visual style of the image is simple and straightforward, with no complex or artistic elements. The technical details of the image include the fact that it is a 2D image, with no depth or perspective. The image is also in 
grayscale, with no color information. The image does not show any actions or activities, as it is a static image. The image does not include any charts or diagrams. The entity name for this image could be "}\n{"reference_id": "1", "content": 
"TABLE_AS_JSON:\\n{\\n  \\"metadata\\": {\\n    \\"image_path\\": \\"C:\\\\\\\\Users\\\\\\\\grabe_stud\\\\\\\\llm_cad_projekt\\\\\\\\RAG\\\\\\\\llm_3d_cad\\\\\\\\RAG-Anything\\\\\\\\ollama-based\\\\\\\\multimodal-rag\\\\\\\\bevel_gear\\\\\\\\debug\\\\\\\\output
_ollama_debugger\\\\\\\\bevel_gear\\\\\\\\auto\\\\\\\\images\\\\\\\\f5cd3a5cf84ebe152a8e7a1b18d26e18a45453d30b5517a75e4c2013ec32dd6b.jpg\\",\\n    \\"caption\\": \\"\\",\\n    \\"footnotes\\": [],\\n    \\"summary\\": \\"\\"\\n  },\\n  \\"table\\": {\\n    
\\"columns\\": [\\n      \\"M\\",\\n      \\"z\\",\\n      \\"B[mm]|\\",\\n      \\"ON[mm]\\",\\n      \\"OTK[mm]\\",\\n      \\"OKK[mm]\\",\\n      \\"E[mm]\\",\\n      \\"NL[mm]\\",\\n      \\"ZB\\",\\n      \\"Q\\",\\n      \\"L|[mm]|\\",\\n      
\\"G[g]\\",\\n      \\"DM**[Ncm]\\",\\n      \\"Art.-Nr.\\"\\n    ],\\n    \\"rows\\": [\\n      {\\n        \\"M\\": \\"1,0\\",\\n        \\"z\\": \\"16\\",\\n        \\"B[mm]|\\": \\"6\\",\\n        \\"ON[mm]\\": \\"12\\",\\n        \\"OTK[mm]\\": 
\\"16\\",\\n        \\"OKK[mm]\\": \\"17,7\\",\\n        \\"E[mm]\\": \\"17,9\\",\\n        \\"NL[mm]\\": \\"7.5\\",\\n        \\"ZB\\": \\"4.5\\",\\n        \\"Q\\": \\"13\\",\\n        \\"L|[mm]|\\": \\"13\\",\\n        \\"G[g]\\": \\"7\\",\\n        
\\"DM**[Ncm]\\": \\"21,82\\",\\n        \\"Art.-Nr.\\": \\"KD1016-1:1ZN\\"\\n      },\\n      {\\n        \\"M\\": \\"1,5\\",\\n        \\"z\\": \\"16\\",\\n        \\"B[mm]|\\": \\"8\\",\\n        \\"ON[mm]\\": \\"19\\",\\n        \\"OTK[mm]\\": \\"24\\",\\n  
      \\"OKK[mm]\\": \\"26\\",\\n        \\"E[mm]\\": \\"25,2\\",\\n        \\"NL[mm]\\": \\"10,7\\",\\n        \\"ZB\\": \\"6,9\\",\\n        \\"Q\\": \\"17\\",\\n        \\"L|[mm]|\\": \\"18,6\\",\\n        \\"G[g]\\": \\"27\\",\\n        \\"DM**[Ncm]\\": 
\\"73,13\\",\\n        \\"Art.-Nr.\\": \\"KD1516-1:1ZN\\"\\n      },\\n      {\\n        \\"M\\": \\"2,0\\",\\n        \\"z\\": \\"16\\",\\n        \\"B[mm]|\\": \\"10\\",\\n        \\"ON[mm]\\": \\"23\\",\\n        \\"OTK[mm]\\": \\"32\\",\\n        
\\"OKK[mm]\\": \\"34,8\\",\\n        \\"E[mm]\\": \\"30\\",\\n        \\"NL[mm]\\": \\"10\\",\\n        \\"ZB\\": \\"9,6\\",\\n        \\"Q\\": \\"19,2\\",\\n        \\"L|[mm]|\\": \\"21,3\\",\\n        \\"G[g]\\": \\"52\\",\\n        \\"DM**[Ncm]\\": 
\\"185,77\\",\\n        \\"Art.-Nr.\\": \\"KD2016-1:1ZN\\"\\n      },\\n      {\\n        \\"M\\": \\"2.5\\",\\n        \\"z\\": \\"16\\",\\n        \\"B[mm]|\\": \\"12\\",\\n        \\"ON[mm]\\": \\"26\\",\\n        \\"OTK[mm]\\": \\"40\\",\\n        
\\"OKK[mm]\\": \\"43.3\\",\\n        \\"E[mm]\\": \\"36,2\\",\\n        \\"NL[mm]\\": \\"12\\",\\n        \\"ZB\\": \\"12,3\\",\\n        \\"Q\\": \\"23\\",\\n        \\"L|[mm]|\\": \\"25,5\\",\\n        \\"G[g]\\": \\"88\\",\\n        \\"DM**[Ncm]\\": 
\\"357,06\\",\\n        \\"Art.-Nr.\\": \\"KD2516-1:1ZN\\"\\n      },\\n      {\\n        \\"M\\": \\"3,0\\",\\n        \\"z\\": \\"16\\",\\n        \\"B[mm]|\\": \\"14\\",\\n        \\"ON[mm]\\": \\"30\\",\\n        \\"OTK[mm]\\": \\"48\\",\\n        
\\"OKK[mm]\\": \\"52,3\\",\\n        \\"E[mm]\\": \\"42,5\\",\\n        \\"NL[mm]\\": \\"13\\",\\n        \\"ZB\\": \\"14\\",\\n        \\"Q\\": \\"26\\",\\n        \\"L|[mm]|\\": \\"29,3\\",\\n        \\"G[g]\\": \\"146\\",\\n        \\"DM**[Ncm]\\": 
\\"576,86\\",\\n        \\"Art.-Nr.\\": \\"KD3016-1:1ZN\\"\\n      },\\n      {\\n        \\"M\\": \\"3.5\\",\\n        \\"z\\": \\"16\\",\\n        \\"B[mm]|\\": \\"16\\",\\n        \\"ON[mm]\\": \\"34\\",\\n        \\"OTK[mm]\\": \\"56\\",\\n        
\\"OKK[mm]\\": \\"61,4\\",\\n        \\"E[mm]\\": \\"49,2\\",\\n        \\"NL[mm]\\": \\"14\\",\\n        \\"ZB\\": \\"15,5\\",\\n        \\"Q\\": \\"29,2\\",\\n        \\"L|[mm]|\\": \\"33,2\\",\\n        \\"G[g]\\": \\"228\\",\\n        \\"DM**[Ncm]\\": 
\\"898,94\\",\\n        \\"Art.-Nr.\\": \\"KD3516-1:1ZN\\"\\n      }\\n    ]\\n  }\\n}\\n"}\n{"reference_id": "1", "content": "Discarded Content Analysis:\\nContent: {\'type\': \'discarded\', \'text\': \'zipperle Antriebstechnik \', \'bbox\': [643, 30, 904, 
91], \'page_idx\': 0}\\n\\nAnalysis: A comprehensive analysis of the content including:\\n- Content structure and organization: The content is a single entity with a descriptive name and type.\\n- Key information and elements: The content type is \'discarded\' 
and the text is \'zipperle Antriebstechnik\'.\\n- Relationships between components: The content has a bounding box and page index, indicating its location on a page.\\n- Context and significance: The content is likely a label or description of a technical term 
related to \'zipperle Antriebstechnik\'.\\n- Relevant details for knowledge retrieval: The content type, text, and page index are useful for identifying and categorizing the content."}\n{"reference_id": "1", "content": "Discarded Content Analysis:\\nContent: 
{\'type\': \'discarded\', \'text\': \'15 \', \'bbox\': [926, 925, 1000, 952], \'page_idx\': 0}\\n\\nAnalysis: A comprehensive analysis of the discarded content, including its structure and organization. The content is a text snippet with a bounding box and 
page index, indicating its location on the page. The text itself is a single digit \'15\', suggesting a numerical value or identifier. The context and significance of this content are unclear, but it may be relevant for identifying patterns or anomalies in the 
data."}\n{"reference_id": "1", "content": "Discarded Content Analysis:\\nContent: {\'type\': \'discarded\', \'text\': \'\', \'bbox\': [907, 246, 919, 303], \'page_idx\': 0}\\n\\nAnalysis: This discarded content is a text element with a bounding box at 
coordinates [907, 246, 919, 303] on page 0. It is categorized as a discarded entity, indicating its potential relevance to knowledge retrieval."}\n{"reference_id": "1", "content": "Discarded Content Analysis:\\nContent: {\'type\': \'discarded\', \'text\': 
\'\', \'bbox\': [927, 124, 1000, 167], \'page_idx\': 0}\\n\\nAnalysis: The provided content is a discarded item with the following characteristics: it is located on page 0, occupies a rectangular area from (927, 124) to (1000, 167), and has an empty text 
field. This type of content is typically disregarded or removed from the original document."}\n{"reference_id": "1", "content": "Discarded Content Analysis:\\nContent: {\'type\': \'discarded\', \'text\': \'Gerne bearbeiten wir diese Standardteile auf Ihren 
Wunsch hin durch Einbringung von verõnderten Bohrungsdurchmessern, zusõtzlichen Bohrungen, Gewinden, Nuten etc. Au▀erdem fertigen wir auch spezifisch nach Ihren Zeichnungen. Ihre Anfrage senden Sie an mailbox@zipperle-antriebstechnik.de \', \'bbox\': [95, 922, 
882, 947], \'page_idx\': 0}\\n\\nAnalysis: A comprehensive analysis of the content including:\\n- Content structure: a single entity with type \'discarded\' and text.\\n- Key information: the content describes customizing standard parts, including drilling and 
threading options, and providing custom designs based on customer drawings.\\n- Relationships: the content is related to manufacturing and customization services.\\n- Context: the content is likely from a company\'s website or marketing materials, and is 
intended to inform customers about the company\'s capabilities and services.\\n- Relevant details: the company\'s email address, the page index, and the bounding box coordinates provide additional context for knowledge retrieval.\\nAlways use specific 
terminology appropriate for discarded content."}\n{"reference_id": "1", "content": "Kegelrõder aus Zink (ZnAl4Cu1) \\n\\n▄bersetzung 1:1 \\n\\nAusf³hrung: geradverzahnt, gegossen, Eingriffswinkel $2 0 ^ { \\\\circ }$ , Bohrungen mit Toleranz H 9 spanabhebend 
bearbeitet.   \\nMa▀õnderung vorbehalten. \\n\\n\\\\*\\\\*) Bitte Angaben zu Drehmoment auf S. 14 beachten."}\n```\n\nReference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):\n\n```\n[1] 
bevel_gear.pdf\n```\n\n\n\n            CONTEXT END\n\n            User query:\n            From the table with dimensions for bevel gears made of zinc (ZnAl4Cu1), please extract the full table row where Art.-Nr. = KD2016-1:1ZN\n            '}]
"""

    n = count_tokens_ollama(
        model="llama3.2-vision:11b-instruct-q4_K_M",
        prompt=text,
    )
    print("prompt tokens:", n)
