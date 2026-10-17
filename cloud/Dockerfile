FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_DEVICE_MAX_CONNECTIONS=1

WORKDIR /opt/kintsugi

COPY cloud/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

# Official state-spaces wheel: CUDA 12, PyTorch 2.6, Python 3.11, PyTorch wheel ABI.
RUN python -m pip install --no-cache-dir \
    "https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1%2Bcu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

COPY vesuvius_code ./vesuvius_code
COPY configs ./configs
COPY scripts ./scripts

RUN python -c "import lightning, mamba_ssm, torch, zarr; print(torch.__version__, torch.version.cuda)"

CMD ["python", "-m", "vesuvius_code.cloud_lightning_train", "--help"]
