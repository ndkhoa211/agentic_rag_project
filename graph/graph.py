from dotenv import load_dotenv

load_dotenv()


from langgraph.graph import END, StateGraph


from graph.nodes import retrieve, grade_documents, web_search, generate
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, WEB_SEARCH, GENERATE
from graph.state import GraphState


# chains from Self-RAG
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.answer_grader import answer_grader


# at 'grade_documents' node, decide which node to go next: generate vs. web_search
def decide_to_generate(state: GraphState):
    print("-----ACCESS GRADE DOCUMENTS-----")

    if state["web_search"]:
        print(
            "-----DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH-----"
        )
        return WEB_SEARCH
    else:
        print("-----DECISION: GENERATE-----")
        return GENERATE


# Self-RAG: define conditional edge function for 'generate' node: if the answer
# 1. not hallucinate/grounded ->
#   (a) did answer question -> "useful"
#   (b) did not answer question -> "not useful"
# 2. hallucinate/not grounded -> "not supported"
def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("-----CHECK HALLUCINATIONS-----")
    # extract all information we got so far when we get to this conditional edge
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # invoke 'hallucination_grader' chain
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("-----DECISION: GENERATION IS GROUNDED IN DOCUMENTS-----")

        print("-----GRADE GENERATION vs QUESTION-----")
        # invoke 'answer_grader' chain
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("-----DECISION: GENERATION ADDRESSES QUESTION-----")
            return "useful"
        else:
            print("-----DECISION: GENERATION DOES NOT ADDRESSES QUESTION-----")
            return "not useful"
    else:
        print("-----DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY-----")
        return "not supported"


# let's build the graph
builder = StateGraph(GraphState)

builder.add_node(RETRIEVE, retrieve)
builder.add_node(GRADE_DOCUMENTS, grade_documents)
builder.add_node(WEB_SEARCH, web_search)
builder.add_node(GENERATE, generate)

builder.set_entry_point(RETRIEVE)

builder.add_edge(RETRIEVE, GRADE_DOCUMENTS)

builder.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEB_SEARCH: WEB_SEARCH,
        GENERATE: GENERATE,
    },
)


# Self-RAG: add conditional edges at 'generate' node
builder.add_conditional_edges(
    GENERATE,  # the source node for Self-RAG
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEB_SEARCH,
    },  # since "useful", "not useful", "not supported" don't represent a real node name,
    # we're going to map them into node names, and they'll be displayed on the edges
)


builder.add_edge(WEB_SEARCH, GENERATE)
builder.add_edge(GENERATE, END)

# let's compile the graph
graph = builder.compile()

# generate mermaid .png
# graph.get_graph().draw_mermaid_png(output_file_path="agentic_RAG_graph.png")
# graph.get_graph().draw_mermaid_png(output_file_path="Corrective_RAG_graph.png")
graph.get_graph().draw_mermaid_png(output_file_path="Self_RAG_graph.png")
