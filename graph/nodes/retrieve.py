from typing import Any, Dict


from graph.state import GraphState
from ingestion import retriever

# by now the retriever is supposed to reference our
# Pinecone vectorstore with all embeddings stored already


# define the retrieve node
def retrieve(state: GraphState) -> Dict[str, Any]:
    print("-----RETRIEVE-----")

    # extract the question from the current state
    question = state["question"]

    # invoke: do semantic search to get relevant docs
    documents = retriever.invoke(question)

    # update "documents" field in our current state with the retrieved documents
    # also add the original question (don't have to do this, but just being cautious)
    return {"documents": documents, "question": question}
