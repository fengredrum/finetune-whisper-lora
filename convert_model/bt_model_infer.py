import librosa

from datetime import datetime
from transformers import AutoProcessor, WhisperForConditionalGeneration
from optimum.bettertransformer import BetterTransformer


model_name = "Oblivion208/whisper-base-cantonese"
task = "transcribe"
language = "zh"

num_inferences = 10
filepath = 'test.wav'

processor_bt = AutoProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model_bt = BetterTransformer.transform(model, keep_original_model=True)
model_bt.config.forced_decoder_ids = processor_bt.get_decoder_prompt_ids(
    language=language, task=task)

audio, sr = librosa.load(filepath, sr=16000, mono=True)

# Measure inference of bt model
start_bt = datetime.now()
for i in range(num_inferences):
    input_features = processor_bt(audio, sampling_rate=sr,
                                  return_tensors="pt").input_features
    predicted_ids = model_bt.generate(input_features, max_length=255)
    transcription = processor_bt.batch_decode(
        predicted_ids, skip_special_tokens=True)
end_bt = datetime.now()

bt_inference_time = (
    end_bt - start_bt).total_seconds() / num_inferences
print(f"BetterTransformer inference time: {bt_inference_time} seconds.")

print(f"transcription: {transcription}")
