import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model = "gpt-4-1106-preview", temperature=0)

import os
from crewai import Agent, Task, Process, Crew
from langchain.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()

researcher = Agent(
    role = 'researcher',
    goal = 'Research mathematical methods to building AGI focusing on the mathematics of the methods',
    backstory = 'you are an AI research assistant',
    tools = [search_tool],
    verbose = True,
    llm=llm,
    allow_delegation = False
)

writer = Agent(
    role = 'research writer',
    goal = 'Write technical posts about how to build AGI',
    backstory = 'You are an AI writer writing in percise technical language',
    verbose = True,
    llm = llm,
    allow_delegation = False
)

task1 = Task(description = 'Investigate mathematics of methods of building AGI',agent=researcher)
task2 = Task(description = 'Compare methods of building AGI',agent=researcher)
task3 = Task(description = 'Write a metanalysis about building AGI mentioning all relevant mathematics',agent=writer)

crew = Crew(
    Agent = [researcher,writer],
    tasks = [task1,task2,task3],
    verbose = True,
    Process = Process.sequential
)

result = crew.kickoff()

print("####################")
print(result)

