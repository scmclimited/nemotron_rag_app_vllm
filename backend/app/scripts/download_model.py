import typer
from pathlib import Path
from huggingface_hub import snapshot_download

from ...config import settings

app = typer.Typer(help="Download Nemotron models from Hugging Face using MODEL_DIR.")

@app.command()
def main(variant: str = typer.Option("vl-fp8", help="vl-fp8 or text-bf16")):
    if variant == "vl-fp8":
        model_id = settings.nemotron_vl_model_id
        subdir = "nemotron_vl_fp8"
    elif variant == "text-bf16":
        model_id = settings.nemotron_text_model_id
        subdir = "nemotron_text_12b"
    else:
        raise typer.BadParameter("Unknown variant (use 'vl-fp8' or 'text-bf16').")

    model_dir = Path(settings.model_dir)
    out_dir = model_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Downloading {model_id} into {out_dir} ...")

    snapshot_download(
        repo_id=model_id,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )

    typer.echo("Done.")

if __name__ == "__main__":
    main()
