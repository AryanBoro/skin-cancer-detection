FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# System dependencies are not needed because we use opencv-python-headless

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
