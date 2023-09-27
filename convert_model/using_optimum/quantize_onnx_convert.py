from pathlib import Path
from optimum.onnxruntime import (
    AutoQuantizationConfig,
    ORTModelForSpeechSeq2Seq,
    ORTQuantizer
)


# model_id = "Oblivion208/whisper-base-cantonese"
model_id = "models/optimized_model"
save_dir = "models/quantized_model"

model = ORTModelForSpeechSeq2Seq.from_pretrained(model_id)
model_dir = model.model_save_dir
# Run quantization for all ONNX files of exported model
onnx_models = list(Path(model_dir).glob("*.onnx"))
print(f"Models: {onnx_models}")

quantizers = [ORTQuantizer.from_pretrained(
    "./", file_name=onnx_model) for onnx_model in onnx_models]

qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)

for quantizer in quantizers:
    # Apply dynamic quantization and save the resulting model
    quantizer.quantize(save_dir=save_dir, quantization_config=qconfig)
