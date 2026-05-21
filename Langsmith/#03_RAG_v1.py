import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Load the API Keys
load_dotenv()

PDF_FILE_PATH = 'islr.pdf'

# Load the pdf
loader = PyPDFLoader(PDF_FILE_PATH)
docs = loader.load() # One document per page

# Creating chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

# Embed + Index
emb = OpenAIEmbeddings(model='text-embedding-3-small')
vs = FAISS.from_documents(splits, emb)
retriever = vs.as_retriever(search_type='similarity', search_kwargs={'k': 4})

# prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know"),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# chain
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

# 6) Ask questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ")
ans = chain.invoke(q.strip())
print("\nA:", ans)