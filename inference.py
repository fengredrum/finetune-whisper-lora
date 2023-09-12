import librosa
from transformers import AutoProcessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

model_name = "./converted_onnx_model"
task = "transcribe"
language = "zh"

processor = AutoProcessor.from_pretrained(model_name)
model = ORTModelForSpeechSeq2Seq.from_pretrained(model_name)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language=language, task=task)
# model.save_pretrained("converted_onnx_model")

filepath = 'test.mp3'
audio, sr = librosa.load(filepath, sr=16000, mono=True)

input_features = processor(audio, sampling_rate=sr,
                           return_tensors="pt").input_features
predicted_ids = model.generate(input_features, max_length=255)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print(f"transcription: {transcription}")
