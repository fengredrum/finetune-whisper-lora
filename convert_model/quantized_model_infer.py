import librosa

from datetime import datetime
from transformers import AutoProcessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq


quantized_dir = "models/quantized_model"
task = "transcribe"
language = "zh"

num_inferences = 10
filepath = 'test.wav'

model_quantized = ORTModelForSpeechSeq2Seq.from_pretrained(quantized_dir)
processor_quantized = AutoProcessor.from_pretrained(quantized_dir)
model_quantized.config.forced_decoder_ids = processor_quantized.get_decoder_prompt_ids(
    language=language, task=task)

audio, sr = librosa.load(filepath, sr=16000, mono=True)

# Measure inference of quantized model
start_quantized = datetime.now()
for i in range(num_inferences):
    input_features = processor_quantized(audio, sampling_rate=sr,
                                         return_tensors="pt").input_features
    predicted_ids = model_quantized.generate(input_features, max_length=255)
    transcription = processor_quantized.batch_decode(
        predicted_ids, skip_special_tokens=True)
end_quantized = datetime.now()

quantized_inference_time = (
    end_quantized - start_quantized).total_seconds() / num_inferences
print(f"Quantized inference time: {quantized_inference_time} seconds.")

print(f"transcription: {transcription}")
