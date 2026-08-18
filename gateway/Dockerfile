
#GATEWAY
FROM python:3.11-slim

# set working dir 
WORKDIR /app

COPY requirements.txt .

#INSTAL DEPENDANCIES
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY evals/ ./evals/

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]