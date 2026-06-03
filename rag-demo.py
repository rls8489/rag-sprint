import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load your API key from .env
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load documents from a folder called 'data'
# documents = SimpleDirectoryReader("data").load_data()

# Load all documents from data folder
print("Loading documents...")
documents = SimpleDirectoryReader("data").load_data()
print(f"Loaded {len(documents)} document(s)")

# Build the index (this is the RAG "retrieval" part)
# index = VectorStoreIndex.from_documents(documents)

# Build index
print("Building index...")
index = VectorStoreIndex.from_documents(documents)
# Create a query engine
query_engine = index.as_query_engine()

# Ask it a question
# response = query_engine.query("What is this document about?")
# print(response)
# response = query_engine.query("What companies has this person worked for and what were their roles?")
# print(response)
# response = query_engine.query("What are this person's AI and data engineering skills?")
# print(response)
# response = query_engine.query("What quantifiable results has this person achieved?")
# print(response)
# response = query_engine.query("How would this person approach AI governance in a regulated environment?")
# print(response)


# Create query engine with source citations
query_engine = index.as_query_engine(
    similarity_top_k=3,
    response_mode="compact"
)

print("\n✅ Ready! Ask questions about your documents.")
print("Type 'quit' to exit.\n")

# Interactive chat loop
while True:
    question = input("You: ").strip()
    
    if question.lower() in ["quit", "exit", "q"]:
        print("Exiting. Good work today!")
        break
    
    if not question:
        continue
    
 # Interactive chat bot - initial pass
#    response = query_engine.query(question)
#    print(f"\nAssistant: {response}\n")

    response = query_engine.query(question)
    print(f"\nAssistant: {response}\n")
    
    # Show source citations
    if response.source_nodes:
        print("📎 Sources:")
        for node in response.source_nodes:
            filename = node.metadata.get("file_name", "unknown")
            score = round(node.score, 3) if node.score else "N/A"
            print(f"  - {filename} (relevance score: {score})")
        print()