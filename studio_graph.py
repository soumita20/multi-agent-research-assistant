import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from research_agent import ResearchAssistant

load_dotenv()

llm = ChatOllama(model = os.getenv("OLLAMA_MODEL", "qwen2.5:14b"), temperature=float(os.getenv("TEMPERATURE", "0")))
search_tool = TavilySearch(max_results = os.getenv("TAVILY_MAX_RESULTS", 3), thread_id = os.getenv("TAVILY_THREAD_ID", "1"))
assistant = ResearchAssistant(llm=llm, search_tool=search_tool, max_interview_turns=int(os.getenv("MAX_INTERVIEW_TURNS", 2)), use_checkpointer=False, save_to_disk=True)
graph = assistant.graph