FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir numpy scipy scikit-learn xgboost joblib \
    fastapi "uvicorn[standard]" Pillow

COPY ml/ ml/
COPY model_server/ model_server/

EXPOSE 8001
CMD ["uvicorn", "model_server.main:app", "--host", "0.0.0.0", "--port", "8001"]
