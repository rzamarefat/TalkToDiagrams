from LLM import ImageQAAgent



if __name__ == "__main__":
    agent = ImageQAAgent(model="gpt-4o-mini")

    images = [
        r"C:\Users\User1\Desktop\internal\TalkToDiagrams\assets\PlotMultipleLinesExample_01.png"
    ]

    question = "You can see a plot in the provided image. please give me a detailed explanation of the curves and the figures."

    answer = agent.ask(question, images)
    print(answer)