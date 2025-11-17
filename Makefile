SHELL := /bin/bash

INFERENCE_MODE ?= vl_fp8        # vl_fp8 | text_4bit | cpu_gguf
GPU_OFFLOAD_FRACTION ?= 0.4     # 0.0–1.0

.PHONY: init-deep-rag download-model-vl-fp8 export-onnx-int8 export-gguf-q4 up down

init-deep-rag:
	chmod +x scripts/setup_deep_rag_assets.sh
	./scripts/setup_deep_rag_assets.sh

PYTHON ?= python  # default, can be overridden on the command line

download-model-vl-fp8:
	$(PYTHON) -m backend.app.scripts.download_model --variant vl-fp8


export-onnx-int8:
	python -m backend.app.scripts.export_onnx --precision bf16
	python -m backend.app.scripts.quantize_onnx --precision int8

export-gguf-q4:
	$(PYTHON) -m backend.app.scripts.export_gguf --quant q4_k_m

# Generate a requirements.txt file by scanning imported modules in the source
# tree.  This target calls the generate_requirements.py utility and writes
# requirements.txt into the project root.  The source directory can be
# overridden via ``SRC`` variable if needed (default: backend/app).
gen-reqs:
	python scripts/requirements.py --src backend/app --output ./requirements.txt

up:
	INFERENCE_MODE=$(INFERENCE_MODE) GPU_OFFLOAD_FRACTION=$(GPU_OFFLOAD_FRACTION) docker compose up --build

down:
	docker compose down
