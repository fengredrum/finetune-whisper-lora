import os
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1
os.environ["inter_op_num_threads"] = "1"
os.environ["intra_op_num_threads"] = "1"

from datetime import datetime
import numpy as np
import librosa
import onnxruntime
from onnxruntime_extensions import get_library_path

audio_file = "data/test.wav"
model = "models/conversion-transformers_optimization-onnx_dynamic_quantization-insert_beam_search-prepost/whisper_cpu_int8_cpu-cpu_model.onnx"

audio, sr = librosa.load(audio_file, sr=16000, mono=True)

inputs = {
    "audio_pcm": np.asarray([audio]),
    "decoder_input_ids": np.asarray([[50258, 50260, 50359, 50363]], dtype=np.int32),
    "max_length": np.array([225], dtype=np.int32),
    "min_length": np.array([1], dtype=np.int32),
    "num_beams": np.array([3], dtype=np.int32),
    "num_return_sequences": np.array([1], dtype=np.int32),
    "length_penalty": np.array([1.0], dtype=np.float32),
    "repetition_penalty": np.array([1.0], dtype=np.float32),
    # "attention_mask": np.zeros((1, 80, 3000), dtype=np.int32),
}

num_threads = 2
options = onnxruntime.SessionOptions()
options.register_custom_ops_library(get_library_path())
options.intra_op_num_threads = num_threads
options.inter_op_num_threads = num_threads
options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session = onnxruntime.InferenceSession(model, options, providers=["CPUExecutionProvider"])

start_optimized = datetime.now()
num_inferences = 5
for i in range(num_inferences):
    outputs = session.run(None, inputs)[0]
end_optimized = datetime.now()

print(outputs)

optimized_inference_time = (
    end_optimized - start_optimized).total_seconds() / num_inferences
print(f"Inference time: {optimized_inference_time} seconds.")