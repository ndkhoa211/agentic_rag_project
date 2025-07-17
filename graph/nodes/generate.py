# this node simply takes the question and documents from
# our state, the run the 'generation_chain'
from typing import Any, Dict
from rich import print


from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("-----GENERATE-----")
    question = state["question"]
    documents = state["document"]

    # update 'generation' key in our graph state with LLM response
    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
