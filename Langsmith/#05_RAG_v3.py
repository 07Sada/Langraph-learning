import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

# LangSmith decorator to automatically log execution traces, inputs, and outputs
from langsmith import traceable

# LangChain utilities for parsing PDFs, splitting text, and integrating with OpenAI/FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (e.g., OPENAI_API_KEY, LANGCHAIN_API_KEY) from a .env file
load_dotenv()

# Global configurations
PDF_FILE = 'islr.pdf'
INDEX_ROOT = Path(".indices")  # Directory where FAISS vector indices will be cached
INDEX_ROOT.mkdir(exist_ok=True)


# ---------- Helper Functions ------

@traceable(name='load_pdf')  # Logs this specific parsing step in LangSmith
def load_pdf(pdf_file_path:str):
    loader = PyPDFLoader(pdf_file_path)
    return loader.load()

@traceable(name='split_documents')  # Logs text splitting parameters and chunks
def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

@traceable(name="build_vector_store")  # Logs the generation and storage of vector embeddings
def build_vector_store(splits, embed_model_name:str):
    emb = OpenAIEmbeddings(model=embed_model_name)
    return FAISS.from_documents(splits, emb)


# ---------- Cache key / fingerprint ------

def _file_fingerprint(path: str) -> dict:
    """Generates a unique hash based on file contents, size, and modification time to detect source changes."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        # Read file in 1MB chunks to efficiently hash large PDFs without exhausting memory
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"sha256": h.hexdigest(), "size": p.stat().st_size, "mtime": int(p.stat().st_mtime)}

def _index_key(pdf_path: str, chunk_size: int, chunk_overlap: int, embed_model_name: str) -> str:
    """Creates a unique hash combining file state and chunking configs to use as a cache folder name."""
    meta = {
        "pdf_fingerprint": _file_fingerprint(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
        "format": "v1",
    }
    # sort_keys guarantees identical dictionaries always generate the exact same hash string
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()


# ----------------- explicitly traced load/build runs -----------------

@traceable(name="load_index", tags=["index"])  # Categorizes this trace under the 'index' tag
def load_index_run(index_dir: Path, embed_model_name: str):
    emb = OpenAIEmbeddings(model=embed_model_name)
    # allow_dangerous_deserialization is required to load local pickle-based FAISS files safely
    return FAISS.load_local(
        str(index_dir),
        emb,
        allow_dangerous_deserialization=True
    )

@traceable(name="build_index", tags=["index"])
def build_index_run(pdf_path: str, index_dir: Path, chunk_size: int, chunk_overlap: int, embed_model_name: str):
    # Executes the data pipeline sequentially and saves results locally
    docs = load_pdf(pdf_path)  # child trace
    splits = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)  # child trace
    vs = build_vector_store(splits, embed_model_name)  # child trace
    
    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))  # Saves the FAISS index binaries
    
    # Write a human-readable metadata file inside the cache folder for debugging
    (index_dir / "meta.json").write_text(json.dumps({
        "pdf_path": os.path.abspath(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
    }, indent=2))
    return vs


# ----------------- dispatcher (not traced) -----------------

def load_or_build_index(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embed_model_name: str = "text-embedding-3-small",
    force_rebuild: bool = False,
):
    """Determines whether to fetch the vector store from local cache or build it from scratch."""
    key = _index_key(pdf_path, chunk_size, chunk_overlap, embed_model_name)
    index_dir = INDEX_ROOT / key
    
    # Check if index already exists and we are not forcing a fresh build
    cache_hit = index_dir.exists() and not force_rebuild
    if cache_hit:
        return load_index_run(index_dir, embed_model_name)
    else:
        return build_index_run(pdf_path, index_dir, chunk_size, chunk_overlap, embed_model_name)


# ----------------- model, prompt, and pipeline -----------------

# Initialize the LLM with 0 temperature to enforce strict, deterministic answers
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# System prompt sets strict guardrails to prevent hallucinations
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    """Concatenates the page text contents from multiple retrieved document objects into one string."""
    return "\n\n".join(d.page_content for d in docs)

@traceable(name="setup_pipeline", tags=["setup"])
def setup_pipeline(pdf_path: str, chunk_size=1000, chunk_overlap=150, embed_model_name="text-embedding-3-small", force_rebuild=False):
    return load_or_build_index(
        pdf_path=pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model_name=embed_model_name,
        force_rebuild=force_rebuild,
    )

@traceable(name="pdf_rag_full_run")  # Acts as the parent span for the entire execution lifespan
def setup_pipeline_and_query(
    pdf_path: str,
    question: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embed_model_name: str = "text-embedding-3-small",
    force_rebuild: bool = False,
):
    # Step 1: Initialize or fetch the vector store
    vectorstore = setup_pipeline(pdf_path, chunk_size, chunk_overlap, embed_model_name, force_rebuild)
    
    # Step 2: Convert the vector store into a retriever that fetches top 4 closest document splits
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Step 3: Define the parallel input preparation stage
    parallel = RunnableParallel({
        # Maps user question -> searches vector db -> flattens results into text block
        "context": retriever | RunnableLambda(format_docs),
        # Passes the user's question through unchanged to the next stage
        "question": RunnablePassthrough(),
    })
    
    # Step 4: Assemble the end-to-end LCEL (LangChain Expression Language) pipeline
    chain = parallel | prompt | llm | StrOutputParser()

    # Run the chain with specific runtime tracking parameters for LangSmith dashboarding
    return chain.invoke(question, config={"run_name": "pdf_rag_query", "tags": ["qa"], "metadata": {"k": 4}})


# ----------------- CLI -----------------
if __name__ == "__main__":
    print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
    q = input("\nQ: ").strip()
    ans = setup_pipeline_and_query(PDF_FILE, q)
    print("\nA:", ans)