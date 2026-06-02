import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load your API key from .env
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load documents from a folder called 'data'
documents = SimpleDirectoryReader("data").load_data()

# Build the index (this is the RAG "retrieval" part)
index = VectorStoreIndex.from_documents(documents)

# Create a query engine
query_engine = index.as_query_engine()

# Ask it a question
response = query_engine.query("What is this document about?")
print(response)
response = query_engine.query("What companies has this person worked for and what were their roles?")
print(response)
response = query_engine.query("What are this person's AI and data engineering skills?")
print(response)
response = query_engine.query("What quantifiable results has this person achieved?")
print(response)
response = query_engine.query("How would this person approach AI governance in a regulated environment?")
print(response)