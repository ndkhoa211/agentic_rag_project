from dotenv import load_dotenv

from graph.chains.router import RouteQuery

load_dotenv()


from rich import print
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import GradeHallucinations, hallucination_grader
from graph.chains.router import RouteQuery, question_router
from ingestion import retriever


# -----------------------retrieval_grader chain-------------------------------
# the first test case where this is a happy flow, i.e. we get the relevant docs
def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"

    # get relevant docs
    docs = retriever.invoke(question)

    # get the first (aka the highest score) doc we found and retrieve its content
    doc_txt = docs[0].page_content

    # invoke the chain, the result is an object of GradeDocuments
    # remind that we did use structured output
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"


def test_retrieval_grader_answer_no() -> None:
    question = "how to make cheese cake"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "no"


# -----------------------generation chain-------------------------------
def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    print(generation)


# -----------------------hallucination_grader chain-------------------------------
def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score


def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    # generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make cheese cake we need to first start with the cheese.",
        }
    )
    assert not res.binary_score


# -----------------------question_router chain-------------------------------
def test_router_to_vectorstore() -> None:
    question = "agent memory"
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"


def test_router_to_web_search() -> None:
    question = "how to make a cheesecake?"
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "web_search"
