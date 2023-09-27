
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq


model_name = "Oblivion208/whisper-base-cantonese"
output_path = "models/vanilla_onnx"

# Export model in ONNX
model = ORTModelForSpeechSeq2Seq.from_pretrained(model_name, export=True)

model.save_pretrained(output_path)
