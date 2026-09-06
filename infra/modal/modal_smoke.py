import os
from pathlib import Path

import modal

app = modal.App("aiforensics-modal-smoke")

volume = modal.Volume.from_name("aiforensics-qwen-ft")

image = modal.Image.debian_slim(python_version="3.10")


@app.function(
    image=image,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=300,
)
def smoke():
    Path("/vol/smoke").mkdir(parents=True, exist_ok=True)
    Path("/vol/smoke/hello.txt").write_text("modal volume ok\n", encoding="utf-8")
    volume.commit()

    token = os.environ.get("HF_TOKEN", "")
    return {
        "volume_write": Path("/vol/smoke/hello.txt").exists(),
        "hf_token_present": bool(token),
        "hf_token_prefix": token[:3] if token else "",
    }
