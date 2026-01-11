import os
from llama_cloud_services import LlamaParse

api_key="llx-WROpF69GBXDRmw9jP8oiN1lwU6iDTbJs0kRY0nj3ReVfXtuY"
os.environ["LLAMA_CLOUD_API_KEY"] = api_key

base_path = "C:/Users/grabe_stud/llm_cad_projekt/RAG/llm_3d_cad/RAG-Anything/ollama-based/sandbox/testing-parsers-4-healthiness/docs/"
file_path = base_path + "gear_m2.pdf"
# file_path = base_path + "gear___scanned.pdf"

documents = LlamaParse(
    result_type="markdown",
    auto_mode=True,
    auto_mode_trigger_on_image_in_page=True,
    auto_mode_trigger_on_table_in_page=True,
    # auto_mode_trigger_on_text_in_page="<text_on_page>"
    # auto_mode_trigger_on_regexp_in_page="<regexp_on_page>"
).load_data(file_path)


from copy import deepcopy
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex


def get_page_nodes(docs, separator="\n---\n"):
    """Split each document into page node, by separator."""
    nodes = []
    for doc in docs:
        doc_chunks = doc.text.split(separator)
        for doc_chunk in doc_chunks:
            node = TextNode(
                text=doc_chunk,
                metadata=deepcopy(doc.metadata),
            )
            nodes.append(node)

    return nodes

page_nodes = get_page_nodes(documents)
print(page_nodes)
print(page_nodes[0].get_content())