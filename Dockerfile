FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./langgraph_app ./langgraph_app

EXPOSE 8000

# CMD ["uvicorn", "app", "--host", "0.0.0.0", "--port", "8000", "--reload"]