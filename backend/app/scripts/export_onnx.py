import typer
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.onnx import export
from transformers.onnx.features import FeaturesManager

from ..config import settings

app = typer.Typer(help="Export text-only Nemotron to ONNX under MODEL_DIR/onnx.")

@app.command()
def main(precision: str = typer.Option("bf16", help="bf16 or fp16")):
    model_id = settings.nemotron_text_model_id
    feature = "causal-lm"

    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True
    )

    onnx_dir = Path(settings.model_dir) / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_out = onnx_dir / "nemotron_12b_bf16.onnx"

    onnx_config = FeaturesManager.get_config(model, task=feature)
    onnx_config.default_onnx_opset = 17

    export(
        tokenizer=tokenizer,
        model=model,
        config=onnx_config,
        opset=onnx_config.default_onnx_opset,
        output=onnx_out,
    )

    print(f"Exported ONNX to {onnx_out}")

if __name__ == "__main__":
    typer.run(main)
