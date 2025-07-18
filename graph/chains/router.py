from dotenv import load_dotenv

load_dotenv()


from typing import Literal

# Literal provides a way to specify that a variable can only take
# one of predefined set of values.
# Useful for validation and type checking


from pydantic import BaseModel, Field


from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,  # means this field is required once we instantiate an object of this class
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)

structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search."""
route_prompt = ChatPromptTemplate(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# create the router chain
question_router = route_prompt | structured_llm_router
