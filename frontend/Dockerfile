# frontend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY ../frontend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ../frontend .

CMD ["streamlit", "run", "frontends.py", "--server.port=8501", "--server.address=0.0.0.0"]
