import typer
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

app = typer.Typer(help="Quantize ONNX to INT8.")

@app.command()
def main(
    input_path: str = typer.Option("backend_models/nemotron_12b_bf16.onnx"),
    precision: str = typer.Option("int8", help="Currently only int8.")
):
    in_path = Path(input_path)
    out_path = Path("backend_models/nemotron_12b_int8.onnx")

    if precision != "int8":
        raise typer.BadParameter("Only int8 is currently supported.")

    quantize_dynamic(
        model_input=str(in_path),
        model_output=str(out_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )

    typer.echo(f"Quantized INT8 model saved to {out_path}")

if __name__ == "__main__":
    main()
