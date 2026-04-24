from LLM import LLM



if __name__ == "__main__":
    agent = LLM(model="gpt-4o-mini")

    images = [
        r".\assets\PlotMultipleLinesExample_01.png"
    ]

    question = "You can see a plot in the provided image. please give me a detailed explanation of the curves and the figures."

    answer = agent.ask(question, images)
    print(answer)