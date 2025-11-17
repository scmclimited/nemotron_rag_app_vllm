SHELL := /bin/bash

INFERENCE_MODE ?= vl_fp8        # vl_fp8 | text_4bit | cpu_gguf
GPU_OFFLOAD_FRACTION ?= 0.4     # 0.0–1.0

.PHONY: init-deep-rag download-model-vl-fp8 export-onnx-int8 export-gguf-q4 up down

init-deep-rag:
	chmod +x scripts/setup_deep_rag_assets.sh
	./scripts/setup_deep_rag_assets.sh

download-model-vl-fp8:
	python -m backend.app.scripts.download_model --variant vl-fp8

export-onnx-int8:
	python -m backend.app.scripts.export_onnx --precision bf16
	python -m backend.app.scripts.quantize_onnx --precision int8

export-gguf-q4:
	python -m backend.app.scripts.export_gguf --quant q4_k_m

up:
	INFERENCE_MODE=$(INFERENCE_MODE) GPU_OFFLOAD_FRACTION=$(GPU_OFFLOAD_FRACTION) docker compose up --build

down:
	docker compose down
