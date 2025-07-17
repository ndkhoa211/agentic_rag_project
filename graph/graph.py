from dotenv import load_dotenv

load_dotenv()


from langgraph.graph import END, StateGraph
from graph.nodes import retrieve, grade_documents, web_search, generate
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, WEB_SEARCH, GENERATE
from graph.state import GraphState


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

builder.add_edge(WEB_SEARCH, GENERATE)
builder.add_edge(GENERATE, END)

# let's compile the graph
graph = builder.compile()

# generate mermaid .png
graph.get_graph().draw_mermaid_png(output_file_path="agentic_RAG_graph.png")
