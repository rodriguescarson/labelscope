# A container so the audit can be run without matching a Python environment.
#
#   docker build -t labelscope .
#   docker run --rm -v "$PWD/data:/data" labelscope scan --labels /data/labelsTr --out /data/audit
#
# CPU only by design: nothing here trains or needs a GPU, so the image stays
# small enough to pull on a laptop and the whole audit runs anywhere.
FROM python:3.12-slim

# tifffile reads the volumes; zarr is needed for the remote streaming reader
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir ".[zarr]"

# scripts are useful inside the container too (fleet sweeps, corpus manifests)
COPY scripts ./scripts

WORKDIR /data
ENTRYPOINT ["labelscope"]
CMD ["--help"]
