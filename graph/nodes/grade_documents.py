# here we implement the grade_documents node
from typing import Any, Dict


from graph.chains.retrieval_grader import retrieval_grader, GradeDocuments
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determine whether the retrieved documents are relevant to the question.
    If any document is not relevant, we will set a flag to run web search.

    Args:
        state (Dict): The current state of the graph.

    Returns:
        state (Dict): Filtered out irrelevant documents and updated web_search state.
    """

    print("-----CHECK DOCUMENT RELEVANCE TO QUESTION-----")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []  # contains relevant docs
    web_search = False
    for doc in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("-----GRADE: DOCUMENT RELEVANT TO QUESTION-----")
            filtered_docs.append(doc)
        else:
            print("-----GRADE: DOCUMENT NOT RELEVANT TO QUESTION-----")
            web_search = True
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}
