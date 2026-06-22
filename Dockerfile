FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Copy repo (mammoth submodule included via --recurse-submodules on the host)
COPY . /workspace/symbioAI

# Install the package (mammoth submodule is already present at symbioAI/mammoth/)
RUN pip install --no-cache-dir -e "/workspace/symbioAI[tracking,viz]"

ENV PYTHONPATH="/workspace/symbioAI:/workspace/symbioAI/mammoth"
WORKDIR /workspace/symbioAI

CMD ["bash"]
