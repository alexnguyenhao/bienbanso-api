FROM python:3.10-slim

WORKDIR /app

# Cài đặt ffmpeg để xử lý âm thanh
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Cài đặt PyTorch CPU và các thư viện cần thiết
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Cổng mặc định của Hugging Face Spaces là 7860
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
