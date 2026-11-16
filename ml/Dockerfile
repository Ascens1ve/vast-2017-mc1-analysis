FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1 VAST_MODEL_PATH=/models/model_v3.joblib
EXPOSE 8767
CMD ["python", "-m", "vast2017_v3_prod.api", "--host", "0.0.0.0", "--port", "8767"]
