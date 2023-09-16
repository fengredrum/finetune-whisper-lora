#!/usr/bin/env bash

python3 original_model_infer.py &&
    python3 bt_model_infer.py &&
    python3 onnx_model_infer.py &&
    python3 optimized_model_infer.py &&
    python3 quantized_model_infer.py
