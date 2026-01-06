import os
# llama-parse is async-first, running the async code in a notebook requires the use of nest_asyncio
import nest_asyncio
from llama_cloud_services import LlamaParse
from llama_index.core import Document


nest_asyncio.apply()
     
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-6Hj1Kk9OKLOMika2AP1JRrFsGYiH8lnikdflwNwYfGgSH4bx"
parser = LlamaParse(parse_mode="parse_page_with_layout_agent")
file_name = "C:/Users/grabe_stud/llm_cad_projekt/RAG/llm_3d_cad/RAG-Anything/ollama-based/sandbox/testing-parsers-4healthy-OCR/docs/gear_m2.pdf"
parsed_pdf = parser.get_json_result(file_name)
     
# convert this into a list of Documents. We also save the bbox along with the text and page index.
pages = parsed_pdf[0]["pages"]
documents = []

for i, page in enumerate(pages):
    # loop through items of the page
    for item in page["items"]:
        document = Document(
            text=item["md"], extra_info={"bbox": item["bBox"], "page": i}
        )
        documents.append(document)
     
print(documents)