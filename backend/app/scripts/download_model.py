import typer
from huggingface_hub import snapshot_download

from ...config import settings

app = typer.Typer(help="Download Nemotron models from Hugging Face.")

@app.command()
def main(variant: str = typer.Option("vl-fp8", help="vl-fp8 or text-bf16")):
    if variant == "vl-fp8":
        model_id = settings.nemotron_vl_model_id
    elif variant == "text-bf16":
        model_id = settings.nemotron_text_model_id
    else:
        raise typer.BadParameter("Unknown variant")

    typer.echo(f"Downloading {model_id}...")
    snapshot_download(repo_id=model_id, local_dir="backend_models", local_dir_use_symlinks=False)
    typer.echo("Done.")

if __name__ == "__main__":
    main()
