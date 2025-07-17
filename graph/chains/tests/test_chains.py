from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever


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
