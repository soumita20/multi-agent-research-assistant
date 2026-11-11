
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from research_agent import Perspectives

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Missing Gemini API key")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=api_key,
    temperature=0,
)

structured_llm = llm.with_structured_output(
    Perspectives,
    method="json_schema",
    include_raw=True,
)

result = structured_llm.invoke([
    HumanMessage(
        content=(
            "Generate exactly 3 AI analysts to research "
            "the impact of generative AI on software engineering. "
            "Give each analyst a distinct perspective."
        )
    )
])

print("Parsed result:", result.get("parsed"))
print("Parsing error:", result.get("parsing_error"))

parsed = result.get("parsed")

if not isinstance(parsed, Perspectives):
    raise ValueError("Gemini did not return a valid Perspectives object")

assert len(parsed.analysts) == 3

for analyst in parsed.analysts:
    print(analyst.model_dump())

print("Structured output test passed")
