import os
import pandas as pd

from llama_index.core.query_engine import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from pdf import canada_engine


# read csv
population_path = os.path.join("data", "population.csv")
population_df = pd.read_csv(population_path)

# query engine for populations
population_query_engine = PandasQueryEngine(
    df=population_df, 
    verbose=True, 
    instruction_str=instruction_str
)
population_query_engine.update_prompts({"pandas_prompt": new_prompt})

# Tools
tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine, 
        metadata=ToolMetadata(
            name="population_data",
            description="this gives information at the world population and demographics"
        )
    ),
    QueryEngineTool(
        query_engine=canada_engine, 
        metadata=ToolMetadata(
            name="canada_data",
            description="this gives detailed information about Canada the country"
        )
    )
]

# llm
llm = OpenAI(model="gpt-3.5-turbo-0613")

# Agent
agent = ReActAgent.from_tools(tools=tools, verbose=True, context=context)

# Prompting
while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)


# # Prompt Examples
# what is the population of india
# can you save a note for me saying "I love George"
# tell me about the languages in canada and save a note of that
