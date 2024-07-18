from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

llm = ChatOpenAI(model="gpt-3.5-turbo")

tools = load_tools(["ddg-search", "wikipedia"], llm=llm)

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agente_executor = AgentExecutor(agent=agent, llm=llm, tools=tools)

query = "Seja direto na resposta. Que cor é o céu?"

agente_executor.invoke({"input": query})
