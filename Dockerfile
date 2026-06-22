FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone Mammoth (the external CL framework this project extends)
RUN git clone --depth=1 https://github.com/aimagelab/mammoth.git /workspace/mammoth

# Install the project with tracking + viz extras
COPY . /workspace/symbioAI
RUN pip install --no-cache-dir -e "/workspace/symbioAI[tracking,viz]"

# Set PYTHONPATH so both this repo and mammoth are importable
ENV PYTHONPATH="/workspace/symbioAI:/workspace/mammoth"

# Default: drop into bash so the user can run any experiment script
CMD ["bash"]
