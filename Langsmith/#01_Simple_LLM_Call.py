from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# LLM Model
MODEL_NAME = 'gpt-4o-mini'
llm_model = ChatOpenAI(model_name=MODEL_NAME)

# Simple one line prompt
prompt = PromptTemplate.from_template("{Question}")
parser = StrOutputParser()

# chain 
chain = prompt | llm_model | parser

result = chain.invoke({"Question": "What is the capital of India?"})
print(result)