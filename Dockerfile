FROM python:3.12-slim-bookworm

# OCR runtime (pytesseract) + OpenMP for faiss/torch wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ configs/
COPY src/ src/
COPY .streamlit/config.toml .streamlit/

EXPOSE 8501

ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    HF_HUB_VERBOSITY=error \
    TRANSFORMERS_VERBOSITY=error \
    TQDM_DISABLE=1

CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
