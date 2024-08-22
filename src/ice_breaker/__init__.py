import os
from langchain.prompts.prompt import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from ice_breaker.third_parties.linkedin import scrape_linkedin_profile


def main():
    information = """
    Carlos Alcaraz Garfia (Spanish pronunciation: [ˈkaɾlos alkaˈɾaθ ˈɣaɾfja];[4] born 5 May 2003) is a Spanish professional tennis player. He has been ranked as high as World No. 1 in singles by the Association of Tennis Professionals (ATP). Alcaraz has won 15 ATP Tour-level singles titles, including four Grand Slam titles and five Masters 1000 titles.

    Alcaraz began his professional career in 2018 aged 15, going on to win three titles on the ITF Men's World Tennis Tour and four on the ATP Challenger Tour. He broke into the top 100 in rankings in May 2021, and ended that year in the top 35 after reaching his first major quarterfinal at the US Open. In March 2022, Alcaraz won his first Masters 1000 title at the Miami Open at the age of 18, and then won his second at the Madrid Open where he defeated Rafael Nadal, Novak Djokovic, and Alexander Zverev in succession. In late 2022, Alcaraz won his first major singles title at the 2022 US Open, becoming the youngest man and the first male teenager in the Open Era to top the singles rankings, at 19 years, 4 months, and 6 days old.[5][6][7] Finishing the year as the youngest year-end No. 1 in ATP ranking history, he was later named the Laureus World Breakthrough of the Year for his performance in the season. In 2023, Alcaraz claimed two additional Masters 1000 titles at Indian Wells and Madrid and his second Grand Slam title at the Wimbledon Championships. In 2024, he won his third and fourth major titles at the French Open[8] and Wimbledon,[9] followed by an Olympic silver medal at the Paris Olympics.
    """
    summary_template = """
        given the information {information} about a person, I want you to create:
        1. a short summary
        2. two interesting facts about the person
    """

    # create a summary prompt template
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )
    # initiate a ChatOpenAI model
    # llm = ChatAnthropic(model="claude-3-opus-20240229")
    llm = ChatOllama(model="llama3")
    # create a chain
    chain = summary_prompt_template | llm | StrOutputParser()

    linkedin_data = scrape_linkedin_profile("test", True)

    res = chain.invoke(input={"information": linkedin_data})
    print(res)