import os
import json
import time
import sys
from datetime import datetime
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from chromadb import Client
from chromadb.config import Settings

PERSISTENT_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "your_collection_name"

console = Console()

# -------------------------------
# 1) Set Up ChromaDB
# -------------------------------

# Initialize ChromaDB client with persistence
# Configure settings with the directory to persist the database
settings = Settings(persist_directory=PERSISTENT_DIRECTORY, is_persistent=True)

# Initialize the Chroma client with the updated settings
client = Client(settings)

# Create or get a collection
collection = client.get_or_create_collection(COLLECTION_NAME)

# -------------------------------
# 2) Set Up Sentence Transformers
# -------------------------------

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str) -> list:
    """
    Generate an embedding for the given text using Sentence Transformers.
    """
    try:
        embedding = model.encode(text).tolist()
        return embedding
    except Exception as e:
        console.print(f"[red]Error obtaining embedding: {e}[/red]")
        return None
    
# -------------------------------
# 3) Utility: Track Processed Files
# -------------------------------
PROCESSED_FILES_PATH = "processed_files.json"

def load_processed_files():
    """
    Returns a dict with { file_id: { modified: str, vectors: [vector_ids], name: str }, ...}
    """
    if os.path.exists(PROCESSED_FILES_PATH):
        with open(PROCESSED_FILES_PATH, "r") as f:
            return json.load(f)
    return {}

def save_processed_files(processed):
    with open(PROCESSED_FILES_PATH, "w") as f:
        json.dump(processed, f, indent=2)

# -------------------------------
# 4) Get Local Files
# -------------------------------

def read_local_file(file_path: str) -> str:
    """
    Read a file from the local 'documents' directory and return its content as a string.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        console.print(f"[red]Error reading file {file_path}: {e}[/red]")
        return ""
    
# -------------------------------
# 5) Split Text into Chunks
# -------------------------------
def split_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """
    Split large text into smaller chunks for embedding.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks

# -------------------------------
# 6) Process a Single File
# -------------------------------

def process_file(file_path: str):
    file_name = os.path.basename(file_path)
    console.rule(f"[bold blue]Processing File: {file_name}")

    # 1. Read the file
    content = read_local_file(file_path)
    if not content:
        console.print(f"[red]No content for file {file_name}[/red]")
        return

    # 2. Split into chunks
    chunks = split_text(content)
    console.print(f"[green]Split text into {len(chunks)} chunks.[/green]")

    # 3. Embed and Upsert
    vector_ids = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        if embedding is None:
            continue

        vector_id = f"{file_name}_{i}"
        vector_ids.append(vector_id)

        metadata = {
            "file_name": file_name,
            "chunk_index": i,
            "text": chunk[:200]  # Store a preview
        }

        try:
            # print(embedding)
            collection.add(
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[chunk],
                ids=[vector_id]
            )
            console.print(f"[green]Upserted chunk {i} successfully.[/green]")
        except Exception as e:
            console.print(f"[red]Error upserting vector: {e}[/red]")

    # 4. Save processed state
    processed = load_processed_files()
    processed[file_name] = {
        "modified": os.path.getmtime(file_path),
        "vectors": vector_ids,
        "name": file_name
    }
    save_processed_files(processed)

    console.print("[bold green]File processed & upserted to ChromaDB.[/bold green]\n")

# -------------------------------
# 7) Delete Vectors for a File
# -------------------------------

def delete_vectors(file_name: str):
    """
    Remove existing vectors for a file using metadata filter in ChromaDB.
    """
    processed = load_processed_files()
    file_data = processed.get(file_name, {})

    # Fallback to metadata-based deletion (Use "file_name" instead of "file_id")
    try:
        collection.delete(where={"file_name": file_name})
        console.print(f"Used metadata filter to delete vectors for {file_name}")
        return True
    except Exception as e:
        console.print(f"[red]Metadata filter deletion failed: {e}[/red]")
        return False

# -------------------------------
# 8) Poll & Update: 
#     Checks for new/changed/deleted files in your Drive folder.
# -------------------------------
def list_local_files():
    """
    List all text files in the 'documents' folder.
    """
    folder_path = "documents/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    files = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):  # Only process .txt files
            files.append({"path": file_path, "name": file_name, "modified": os.path.getmtime(file_path)})
    return files

def update_files():
    console.print(f"\n=== Update started {datetime.now().isoformat()} ===\n")
    processed = load_processed_files()

    try:
        current_files = list_local_files()

        # 1) Handle deletions
        for file_name in list(processed.keys()):
            file_path = os.path.join("documents/", file_name)
            if not os.path.exists(file_path):
                console.print(f"Removing vectors for deleted file: {file_name}")
                if delete_vectors(file_name):  # <=== This now properly deletes old entries
                    del processed[file_name]
                    save_processed_files(processed)

        # 2) Handle new or modified files
        for file in current_files:
            existing = processed.get(file["name"])
            if (not existing) or (file["modified"] > existing["modified"]):
                console.print(f"Deleting old vectors for: {file['name']}")  # Debugging
                delete_vectors(file["name"])  # <=== ADD THIS before reprocessing!
                process_file(file["path"])  # Reprocess file after deleting old entries

    except Exception as e:
        console.print(f"Update failed: {str(e)}")

# -------------------------------
# Optional: A simple main loop
# -------------------------------
def wait_or_pull(interval=3600):
    """
    Wait for a specified interval, but allow typing 'pull' to do an immediate update
    or 'q' to quit.
    """
    start_time = time.time()
    while time.time() - start_time < interval:
        user_input = input("Type 'pull' to run update immediately or 'q' to quit: ").strip().lower()
        if user_input == "pull":
            return
        elif user_input == "q":
            print("Exiting...")
            sys.exit(0)
        time.sleep(1)

if __name__ == "__main__":
    while True:
        update_files()
        wait_or_pull()