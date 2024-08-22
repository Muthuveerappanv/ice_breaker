from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool


def lookup(name: str) -> str:
    llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0.0)
    template = """given the full name {name_of_person} I want you to find a link to their 
    Linkedin profile page. Provide the link only. Do not provide any other information.
    """
    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func="?",
            description="useful for when you need to get the linkedin page URL",
        )
    ]
