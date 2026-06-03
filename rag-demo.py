import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

# Load API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load all documents with metadata
print("Loading documents...")
documents = SimpleDirectoryReader("data", filename_as_id=True).load_data()

# Tag each document with its filename as metadata
for doc in documents:
    doc.metadata["file_name"] = os.path.basename(doc.metadata["file_path"])

print(f"Loaded {len(documents)} document(s):")
for doc in documents:
    print(f"  - {doc.metadata['file_name']}")

# Build index
print("\nBuilding index...")
index = VectorStoreIndex.from_documents(documents)

# Document menu
doc_map = {
    "1": "resume.txt",
    "2": "job_description.txt",
    "3": "job_description_2.txt",
    "all": None
}

print("\n✅ Ready!")
print("\nAvailable document filters:")
print("  1 - Resume only")
print("  2 - Job Description 1 only")
print("  3 - Job Description 2 only")
print("  all - All documents")
print("\nType 'quit' to exit.\n")

while True:
    # Ask which document to filter on
    doc_choice = input("Filter by document (1/2/3/all): ").strip().lower()
    
    if doc_choice in ["quit", "exit", "q"]:
        print("Exiting. Good work today!")
        break
        
    if doc_choice not in doc_map:
        print("Invalid choice. Please enter 1, 2, 3, or all.")
        continue
    
    # Build query engine with or without filter
    selected_file = doc_map[doc_choice]
    
    if selected_file:
        filters = MetadataFilters(filters=[
            MetadataFilter(key="file_name", value=selected_file)
        ])
        query_engine = index.as_query_engine(
            similarity_top_k=3,
            filters=filters
        )
        print(f"\n🔍 Filtering to: {selected_file}")
    else:
        query_engine = index.as_query_engine(similarity_top_k=3)
        print(f"\n🔍 Searching all documents")
    
    # Ask question
    question = input("You: ").strip()
    
    if not question:
        continue
    
    response = query_engine.query(question)
    print(f"\nAssistant: {response}\n")
    
    # Show citations
    if response.source_nodes:
        print("📎 Sources:")
        for node in response.source_nodes:
            filename = node.metadata.get("file_name", "unknown")
            score = round(node.score, 3) if node.score else "N/A"
            print(f"  - {filename} (relevance score: {score})")
        print()