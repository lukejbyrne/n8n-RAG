import sys
from ollama import chat
from rich.console import Console

try:
    from app import get_embedding, console, collection
except ImportError:
    print("Make sure 'app.py' is in the same folder and named appropriately.")
    sys.exit(1)

def main():
    """
    Run the company documents chat interface.
    """
    console.rule("[bold magenta]Company Documents Chat[/]")
    console.print("[bold green]Type 'exit' to quit.[/]\n")
    
    while True:
        query = console.input("[bold cyan]Your Question> [/]").strip()
        if query.lower() in ("exit", "quit"):
            break
        
        answer = chat_agent(query)
        console.print(f"\n[bold yellow]Answer:[/] {answer}\n")

def chat_agent(query: str) -> str:
    """
    1) Embed the query
    2) Retrieve top matches from ChromaDB
    3) Pass them into the Ollama Chat model for a final answer
    """
    system_message = (
        "You are a helpful HR assistant designed to answer employee questions based on company policies. "
        "Retrieve relevant information from the provided internal documents and provide a concise, accurate answer. "
        "If the answer cannot be found in the provided documents, say 'I cannot find the answer in the available resources.'"
    )

    # 1) Embed the user query
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return "Error obtaining query embedding."

    # 2) Query ChromaDB
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
    except Exception as e:
        return f"Error querying ChromaDB: {str(e)}"

    if not results or "documents" not in results or not results["documents"]:
        return "I cannot find the answer in the available resources."

    # Combine top results into a single string for context
    documents = results["documents"][0]
    context = " ".join(documents)
    if not context.strip():
        return "I cannot find the answer in the available resources."

    # 3) Use Ollama chat completion
    return ollama_chat(system_message, query, context)

def ollama_chat(system_message: str, query: str, context: str) -> str:
    """
    Calls Ollama's chat completion with a system message, user's query, and retrieved context.
    """
    try:
        response = chat(
            model='llama2:7b',
            messages=[
                {
                    'role': 'system',
                    'content': f"{system_message}\n\nContext: {context}"
                },
                {
                    'role': 'user',
                    'content': query
                }
            ],
            stream=False  # Set to True if you want to handle streaming responses
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    main()