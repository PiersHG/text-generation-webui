FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git python3 python3-pip python-is-python3 curl && \
    apt-get clean

# Clone web UI
RUN git clone https://github.com/oobabooga/text-generation-webui .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose port and set entrypoint
EXPOSE 7860
CMD ["python3", "server.py", "--model", "models/Nous-Hermes-2-Mistral"]