from dotenv import load_dotenv

load_dotenv()


from graph.graph import graph


if __name__ == "__main__":
    print("::::::::::Hello Agentic RAG::::::::::")

    # EXAMPLE 1
    # print(graph.invoke(
    #     input={"question": "what is agent memory?"} # all relevant
    # ))

    # EXAMPLE 2
    # print(graph.invoke(
    #     input={"question": "what is AGI?"}  # all irrelevant
    # ))

    # EXAMPLE 3
    # print(
    #     graph.invoke(input={"question": "what is agent autonomous?"})  # partly relevant
    # )

    # EXAMPLE 4: Self-RAG
    # print(
    #     graph.invoke(input={"question": "dsadafs?"}, # not address question
    #                  config={"recursion_limit": 5}, # default: 25
    #                  )
    # )

    # EXAMPLE 5: Self-RAG
    hallucination_prompt = (
        "In Lilian Weng’s fictional post ‘Reversible Memory Transformers’, she claims "
        "Bidirectional Episodic Replay (BER) boosts Procgen Jumper‑Hard by 45 %. "
        "Explain how BER plugs into a PPO loop (PyTorch pseudo‑code) and list her hyper‑parameters."
    )

    print(
        graph.invoke(
            input={"question": hallucination_prompt},  # not address question
            config={"recursion_limit": 10},  # default: 25,
        )
    )
