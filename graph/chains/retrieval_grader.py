from dotenv import load_dotenv

load_dotenv()


# this chain receives as an input the original question
# then retrieves documents and determines whether they
# are relevant to the question or not
# we'll run this chain for each doc we retrieve
# we'll leverage structured output for this
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)


# inherited from Pydantic's BaseModel class
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )  # LLM will leverage the description to decide whether this doc is relevant or not


# under the hood, LangChain uses function calling, and for every LLM call we make
# we'll return a Pydantic object, and LLM will return in the schema that we want
# and if we want to use LLM with structured output, make sure that our LLM supports function calling
structured_llm_grader = llm.with_structured_output(GradeDocuments)


system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# create the chain
retrieval_grader = grade_prompt | structured_llm_grader

##### WRITING TEST #####
# Testing for GenAI and LLM-based application is tricky:
# 1. no idempotency (LLMs are statistical)
# 2. relying on a 3rd party (no control about availability,
#    durability (e.g. rate limit, service not available, internal service error))
# 3. cost

# I want to put those testing problems aside and let's implement some
# tests for our application, which is still better than nothing
# And even though we're not going to run it in a CI/CD pipeline, we can run it
# manually and it'll give us a sanity check to see that our application is working
# and doing what it's supposed to do

# lastly, admittedly, recently all new (top tier) models, have become so much better,
# judging by the quality of the answer, latency and cost,
# And I've seen companies integrating that into their CI/CD system, despite all the
# disadvantages I've noted before, and I have to say that there are ways to address
# and to mitigate those disadvantages, but we're not going to discuss them in the scope
# of this course
