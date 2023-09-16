import librosa

from datetime import datetime
from transformers import AutoProcessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq


optimized_dir = "models/optimized_model"
task = "transcribe"
language = "zh"

num_inferences = 10
filepath = 'test.wav'

model_optimized = ORTModelForSpeechSeq2Seq.from_pretrained(optimized_dir)
processor_optimized = AutoProcessor.from_pretrained(optimized_dir)
model_optimized.config.forced_decoder_ids = processor_optimized.get_decoder_prompt_ids(
    language=language, task=task)

audio, sr = librosa.load(filepath, sr=16000, mono=True)

# Measure inference of optimized model
start_optimized = datetime.now()
for i in range(num_inferences):
    input_features = processor_optimized(audio, sampling_rate=sr,
                                         return_tensors="pt").input_features
    predicted_ids = model_optimized.generate(input_features, max_length=255)
    transcription = processor_optimized.batch_decode(
        predicted_ids, skip_special_tokens=True)
end_optimized = datetime.now()


optimized_inference_time = (
    end_optimized - start_optimized).total_seconds() / num_inferences
print(f"Optimized inference time: {optimized_inference_time} seconds.")

print(f"transcription: {transcription}")
