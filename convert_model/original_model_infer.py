import librosa

from datetime import datetime
from transformers import AutoProcessor, WhisperForConditionalGeneration


model_name = "Oblivion208/whisper-base-cantonese"
task = "transcribe"
language = "zh"

num_inferences = 10
filepath = 'test.wav'

processor_original = AutoProcessor.from_pretrained(model_name)
model_original = WhisperForConditionalGeneration.from_pretrained(model_name)
model_original.config.forced_decoder_ids = processor_original.get_decoder_prompt_ids(
    language=language, task=task)

audio, sr = librosa.load(filepath, sr=16000, mono=True)

# Measure inference of original model
start_original = datetime.now()
for i in range(num_inferences):
    input_features = processor_original(audio, sampling_rate=sr,
                                        return_tensors="pt").input_features
    predicted_ids = model_original.generate(input_features, max_length=255)
    transcription = processor_original.batch_decode(
        predicted_ids, skip_special_tokens=True)
end_original = datetime.now()

original_inference_time = (
    end_original - start_original).total_seconds() / num_inferences
print(f"Original inference time: {original_inference_time}  seconds.")

print(f"transcription: {transcription}")
