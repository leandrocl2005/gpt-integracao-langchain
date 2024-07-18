from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
import bs4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path(".env")
load_dotenv(dotenv_path=dotenv_path)


llm = ChatOpenAI(model="gpt-3.5-turbo")
query = "Vou viajar para Londres em agosto de 2024. Faça um roteiro de viagem para mim."


def researchAgent(query, llm):
    tools = load_tools(["ddg-search", "wikipedia"], llm=llm)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    agente_executor = AgentExecutor(agent=agent, llm=llm, tools=tools)
    webContext = agente_executor.invoke({"input": query})
    return webContext


def loadData():
    loader = WebBaseLoader(
        web_paths=("https://www.dicasdeviagem.com/inglaterra",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=(
                    "postcontentwrap",
                    "pagetitleloading background-imaged loading-dark",
                )
            )
        ),
        verify_ssl=False,
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def getRelevantDocs(query):
    retriever = loadData()
    relevant_documents = retriever.invoke(query)
    return relevant_documents


def supervisorAgent(query, llm, webContext, relevant_documents):
    prompt_template = """
    Você é um gerente de uma agência de viagens.
    Sua resposta final deverá ser um roteiro de viagens resumido.
    Utilize o contexto de eventos e preços de passagens,
    o input do usuário e também documentos relevantes para criar o roteiro.
    Contexto: {webContext}
    Documento relevante: {relevant_documents}
    Usuário: {query}
    Assistente:
    """
    prompt = PromptTemplate(
        input_variables=["webContext", "relevant_documents", "query"],
        template=prompt_template,
    )
    sequence = RunnableSequence(prompt | llm)
    response = sequence.invoke(
        {
            "webContext": webContext,
            "relevant_documents": relevant_documents,
            "query": query,
        }
    )
    return response


def getResponse(query, llm):
    webContext = researchAgent(query, llm)
    relevant_documents = getRelevantDocs(query)
    response = supervisorAgent(query, llm, webContext, relevant_documents)
    return response


print(getResponse(query, llm))
