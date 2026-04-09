from .agent import Agent

def build_agent(rag_callable):
    return Agent(rag_callable=rag_callable)