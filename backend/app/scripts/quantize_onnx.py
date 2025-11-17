import typer
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

from ...config import settings

app = typer.Typer(help="Quantize ONNX to INT8 under MODEL_DIR/onnx.")

@app.command()
def main(
    input_path: str = typer.Option("", help="Optional custom ONNX path (defaults to MODEL_DIR/onnx/nemotron_12b_bf16.onnx)"),
    precision: str = typer.Option("int8", help="Currently only int8.")
):
    onnx_dir = Path(settings.model_dir) / "onnx"
    default_in = onnx_dir / "nemotron_12b_bf16.onnx"
    in_path = Path(input_path) if input_path else default_in
    out_path = onnx_dir / "nemotron_12b_int8.onnx"

    if precision != "int8":
        raise typer.BadParameter("Only int8 is currently supported.")

    onnx_dir.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(in_path),
        model_output=str(out_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )

    print(f"Quantized INT8 model saved to {out_path}")

if __name__ == "__main__":
    main()
