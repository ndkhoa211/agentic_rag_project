from dotenv import load_dotenv

load_dotenv()


from typing import Any, Dict


from langchain.schema import Document
from langchain_tavily import TavilySearch


from graph.state import GraphState


web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("-----WEB SEARCH-----")

    # extract 'question' and 'documents' from the graph state
    # since we execute 'web_search' mode after 'grade_documents' node,
    # all documents in the list now must be relevant to our query (or empty)
    question = state["question"]
    # documents = state["documents"]
    # When the router goes to web_search, there are no documents in the state and the state has
    # only the key "question". No other Key. You probably should use
    documents = state.get("documents", None)

    # perform search
    tavily_results = web_search_tool.invoke({"query": question})["results"]

    # join the content snippets: i.e.
    # take the 'content' keys of all elements (i.e. dictionaries) in 'tavily_results' list
    # and combine them into one 'Document' of LangChain, called 'web_results'
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_result)

    # append or initialize documents list
    if documents is not None:  # if we currently have docs in our state (relevant ones)
        documents.append(web_results)  # append web search results to those docs
    else:  # no relevant docs
        documents = [web_results]

    return {"documents": documents, "question": question}  # return update of our state


if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None})
