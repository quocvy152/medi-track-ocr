FROM python:3.11-slim

WORKDIR /app

# Cài đặt dependencies hệ thống
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables để tối ưu memory và CPU
ENV PADDLE_USE_GPU=0
ENV FLAGS_allocator_strategy=naive_best_fit
ENV FLAGS_fraction_of_gpu_memory_to_use=0.0
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Copy và cài đặt Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Pre-download PaddleOCR models trong quá trình build
# Điều này đảm bảo models được tải sẵn và không cần tải khi runtime
RUN python download_models.py

# Expose port
EXPOSE 8000

# Chạy ứng dụng
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
