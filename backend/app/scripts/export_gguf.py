import subprocess
import typer
from pathlib import Path

from ...config import settings

app = typer.Typer(help="Export Nemotron 12B text-only to GGUF via llama.cpp into MODEL_DIR/gguf.")

@app.command()
def main(
    hf_id: str = typer.Option(None, help="HF repo id (defaults to settings.nemotron_text_model_id)"),
    llama_cpp_path: str = typer.Option("tools/llama.cpp", help="Path to llama.cpp checkout"),
    quant: str = typer.Option("q4_k_m", help="GGUF quantization type (e.g. q4_k_m, q5_k_m, q6_k)"),
):
    hf_id = hf_id or settings.nemotron_text_model_id
    llama_cpp = Path(llama_cpp_path)
    script = llama_cpp / "convert-hf-to-gguf.py"

    gguf_dir = Path(settings.model_dir) / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    out = gguf_dir / f"nemotron_12b_{quant}.gguf"

    cmd = [
        "python", str(script),
        "--model", hf_id,
        "--outfile", str(out),
        "--outtype", quant,
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Saved GGUF to {out}")

if __name__ == "__main__":
    main()
