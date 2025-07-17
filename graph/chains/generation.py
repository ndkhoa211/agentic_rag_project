# implement 'generation' chain
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
prompt = hub.pull("rlm/rag-prompt")

# create the chain
generation_chain = prompt | llm | StrOutputParser()
