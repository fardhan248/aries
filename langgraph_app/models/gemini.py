import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

# Google Gemini Free API
gemini = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=google_api_key,
    thinking_level="low",
    streaming=True,
)

gemini_instruct = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=google_api_key,
    thinking_level="medium",
    streaming=True,
)

gemini_thinking_reasoning = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=google_api_key,
    thinking_level="high",
    streaming=True,
)

gemini_embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=google_api_key,
)

# Image/video generation?