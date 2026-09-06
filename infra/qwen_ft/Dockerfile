# qwen_ft training container for Vertex AI CustomJob (A100 40GB target).
#
# Unpinned by design for the first build: the implementation plan records the
# installed versions in the training log on the first run and pins them in a
# follow-up commit once the container builds and trains successfully.

# CUDA 12.x / PyTorch base image compatible with A100.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Upstream Qwen2.5-VL fine-tuning framework (never copied into this repo's
# committed source, same policy as the NPR external baseline). The clone is
# shallow; pin the commit after the first successful build.
RUN git clone --depth 1 https://github.com/QwenLM/Qwen2.5-VL.git /opt/qwen-vl-finetune \
    && pip install --no-cache-dir -r /opt/qwen-vl-finetune/qwen-vl-finetune/requirements.txt

COPY train_qwen_ft.py /app/train_qwen_ft.py

ENTRYPOINT ["python", "/app/train_qwen_ft.py"]
