from dotenv import load_dotenv

load_dotenv()


from graph.graph import graph


if __name__ == "__main__":
    print("::::::::::Hello Advanced RAG::::::::::")

    # EXAMPLE 1
    # print(graph.invoke(
    #     input={"question": "what is agent memory?"} # all relevant
    # ))

    # EXAMPLE 2
    # print(graph.invoke(
    #     input={"question": "what is AGI?"}  # all irrelevant
    # ))

    # EXAMPLE 3
    print(
        graph.invoke(input={"question": "what us agent autonomous?"})  # partly relevant
    )
