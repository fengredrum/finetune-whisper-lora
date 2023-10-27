import os
import librosa
import numpy as np
import onnxruntime

from onnxruntime_extensions import get_library_path
from datetime import datetime

num_threads = 4
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

model = "models/mandarin/whisper_cpu_int8_cpu-cpu_model.onnx"

options = onnxruntime.SessionOptions()
options.register_custom_ops_library(get_library_path())
options.intra_op_num_threads = num_threads
options.inter_op_num_threads = num_threads
options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session = onnxruntime.InferenceSession(
    model, options, providers=["CPUExecutionProvider"])

filepath = "data/"
filenames = os.listdir(filepath)
for audio_file in filenames:

    audio, sr = librosa.load(filepath + audio_file, sr=16000, mono=True)
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

    start_time = datetime.now()
    num_inferences = 4
    for i in range(num_inferences):
        outputs = session.run(None, inputs)[0]
    end_time = datetime.now()

    print(f'{audio_file} transcript: {outputs}')
    inference_time = (
        end_time - start_time).total_seconds() / num_inferences
    print(f"Inference time: {inference_time} seconds.")
