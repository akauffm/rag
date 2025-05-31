#pip install llama-index
#pip install faiss-cpu
#pip install llama-index-vector-stores-faiss
#pip install llama-index-embeddings-huggingface

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.readers.file import FlatReader
import faiss

# dimensions of all-MiniLM-L6-v2
d = 384
faiss_index = faiss.IndexIVFFlat(d)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# load documents
documents = SimpleDirectoryReader("./data/").load_data()

vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context,
)

# save the vector database to disk
index.storage_context.persist()