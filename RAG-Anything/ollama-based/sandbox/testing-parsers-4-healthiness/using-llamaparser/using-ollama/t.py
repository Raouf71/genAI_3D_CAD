# bring in our LLAMA_CLOUD_API_KEY
import os
from dotenv import load_dotenv
from llama_parse import LlamaParse  # noqa: E402
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader  # noqa: E402
import nest_asyncio  # noqa: E402

load_dotenv()
nest_asyncio.apply()

api_key="llx-WROpF69GBXDRmw9jP8oiN1lwU6iDTbJs0kRY0nj3ReVfXtuY"
os.environ["LLAMA_CLOUD_API_KEY"] = api_key

base_path = "C:/Users/grabe_stud/llm_cad_projekt/RAG/llm_3d_cad/RAG-Anything/ollama-based/sandbox/testing-parsers-4healthy-OCR/docs/"
# file_name = base_path + "gear_m2.pdf"
file_name = base_path + "gear___scanned.pdf"

# set up parser
llamaparse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
parser = LlamaParse(
    api_key=llamaparse_api_key,
    result_type="markdown"  # "markdown" and "text" are available
)

# use SimpleDirectoryReader to parse our file
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(input_files=[file_name], file_extractor=file_extractor).load_data()

print(documents)

#len(documents)

#documents[0].text
print(documents[0].text[:200])

########### Ollama Models ###############

# # by default llamaindex uses OpenAI models
# from llama_index.embeddings.ollama import OllamaEmbedding  # noqa: E402

# embed_model = OllamaEmbedding(
#     #model_name="nomic-embed-text",
#     model_name="llama2",
#     base_url="http://localhost:11434",
#     ollama_additional_kwargs={"mirostat": 0},
# )

# from llama_index.llms.ollama import Ollama  # noqa: E402
# llm = Ollama(model="llama2", request_timeout=30.0)

# from llama_index.core import Settings  # noqa: E402

# Settings.llm = llm
# Settings.embed_model = embed_model

# # get the answer out of it
# # create an index from the parsed markdown
# index = VectorStoreIndex.from_documents(documents)

# # create a query engine for the index
# query_engine = index.as_query_engine()

# # query the engine
# from IPython.display import Markdown, display  # noqa: E402

# # query the engine
# query = "what is the BoolQ value of GPT4All-J 6B v1.0* model ?"
# response = query_engine.query(query)
# display(Markdown(f"<b>{response}</b>"))