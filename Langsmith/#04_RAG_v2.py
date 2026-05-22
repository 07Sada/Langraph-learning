import os 
from dotenv import load_dotenv
from langsmith import traceable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from typing import List

# importing the api keys
load_dotenv()

# importing the pdf
PDF_FILE = 'islr.pdf'

@traceable(name='load_pdf')
def pdf_loader(pdf_file_path: str):
    # load the pdf file
    loader = PyPDFLoader(pdf_file_path) 
    return loader.load() # list[documents]

@traceable(name='split_documents')
def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    # Splitting the docs
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)
    return splits

@traceable(name='build_vector_store')
def build_vector_store(splits):
    # converting the chunks in embedding & storing it in vector store
    emb = OpenAIEmbeddings(model='text-embedding-3-small')
    vector_store = FAISS.from_documents(splits, emb)
    return vector_store

@traceable(name='setup_pipeline')
def setup_pipeline(pdf_file_path:str):
    docs = pdf_loader(pdf_file_path)
    splits = split_documents(docs)
    vector_store = build_vector_store(splits)
    return vector_store

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# LLM Model
llm_model = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# prompt 
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know"),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# building the vector store
vector_store = setup_pipeline(PDF_FILE)
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})

# parallel chain 
parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

# chain 
chain = parallel_chain | prompt | llm_model | StrOutputParser()

print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ").strip()

# Give the visible run name + tags/metadata so it’s easy to find:
config = {"run_name": "pdf_rag_query"}

ans = chain.invoke(q, config=config)
print("\nA:", ans)