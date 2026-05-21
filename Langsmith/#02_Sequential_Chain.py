from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load the API keys
load_dotenv()

# LLM Model
MODEL_NAME = 'gpt-4o-mini'
llm_model = ChatOpenAI(model_name=MODEL_NAME)

# Templates
prompt_one = PromptTemplate(
    template='Write a detailed report on the {topic}',
    input_variables=['topic']
)

prompt_two = PromptTemplate(
    template="Write a 5 pointer summary on below report\n{report_text}",
    input_variables=['topic_text']
)

# Parser
parser = StrOutputParser()

# Sequential chain
chain = prompt_one | llm_model | parser | llm_model | parser

# Invoking the chain
result = chain.invoke({"topic": "Recruitement in IT in India"})
print(result)