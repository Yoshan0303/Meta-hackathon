FROM python:3.11-slim

# Create a non-root user (Hugging Face Spaces run as user 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements and install (owned by user)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the code
COPY --chown=user . .

EXPOSE 7860

# Start the OpenEnv FastAPI server on port 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
