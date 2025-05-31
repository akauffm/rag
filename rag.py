#pip install llama-index-llms-ollama
#pip install llama-index
#pip install faiss-cpu
#pip install llama-index-vector-stores-faiss
#pip install llama-index-embeddings-huggingface

# load index from disk
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
import time

from llama_index.core.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

Settings.llm = Ollama(model="gemma3:4b", request_timeout=30.0, stream=True)

vector_store = FaissVectorStore.from_persist_dir("./storage")
storage_context = StorageContext.from_defaults(
    vector_store=vector_store, persist_dir="./storage"
)
index = load_index_from_storage(storage_context=storage_context)
query_engine = index.as_query_engine(similarity_top_k=1)

chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    context_prompt=(
        "You are an efficient chatbot, able to have normal interactions, as well as talk"
        " succinctly about the information in your wikipedia knowledge base."
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history, or the context above to interact "
        "and help the user. Just answer the question, don't use fancy formatting, "
        "don't suggest other questions, don't offer to help the user. If you can't find "
        "requested information in your knoweldge base, say 'the provided text doesn't include this "
        "information.' Do not invent an answer. If the user is just chatting, feel free to respond."
    ),
    verbose=False,
)

def process_input(prompt):
    """
    This is where you define what to do with the user's input.
    """
    query_str = prompt

    start_time = time.time()
    #response = query_engine.query(query_str)
    response = chat_engine.chat(query_str)
    elapsed_time = time.time() - start_time
    print(response)
    print(f"Process took: {elapsed_time:.2f} seconds")

# --- Main loop ---
print("Script started. Type 'quit' or 'exit' to stop.")

while True:
    try:
        # 1. Await user input
        user_command = input("> ") # The "> " is a prompt

        # 2. Act on it (check for exit condition first)
        if user_command.lower() in ["quit", "exit"]:
            print("Exiting script...")
            break # Exit the loop

        process_input(user_command)

    except KeyboardInterrupt: # Handle Ctrl+C gracefully
        print("\nExiting due to KeyboardInterrupt...")
        break
    except Exception as e: # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        # Optionally, you might want to break here or log the error and continue
        # break

print("Script finished.")