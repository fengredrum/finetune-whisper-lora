import librosa

from datetime import datetime
from transformers import AutoProcessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq


onnx_dir = "models/vanilla_onnx"
task = "transcribe"
language = "zh"

num_inferences = 10
filepath = 'test.wav'

model_onnx = ORTModelForSpeechSeq2Seq.from_pretrained(onnx_dir)
processor_onnx = AutoProcessor.from_pretrained(onnx_dir)
model_onnx.config.forced_decoder_ids = processor_onnx.get_decoder_prompt_ids(
    language=language, task=task)

audio, sr = librosa.load(filepath, sr=16000, mono=True)

# Measure inference of onnx model
start_onnx = datetime.now()
for i in range(num_inferences):
    input_features = processor_onnx(audio, sampling_rate=sr,
                                    return_tensors="pt").input_features
    predicted_ids = model_onnx.generate(input_features, max_length=255)
    transcription = processor_onnx.batch_decode(
        predicted_ids, skip_special_tokens=True)
end_onnx = datetime.now()

onnx_inference_time = (
    end_onnx - start_onnx).total_seconds() / num_inferences
print(f"ONNX inference time: {onnx_inference_time} seconds.")

print(f"transcription: {transcription}")
