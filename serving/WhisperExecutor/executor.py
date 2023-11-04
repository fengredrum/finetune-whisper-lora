import os
import onnxruntime
import numpy as np

from onnxruntime_extensions import get_library_path
from jina import Executor, requests, Flow
from docarray import BaseDoc, DocList
from docarray.typing import NdArray
from typing import Optional
from utils_vad import VADOnnxWrapper, read_audio, get_speech_timestamps

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1


class AudioURL(BaseDoc):
    audio_url: str = None
    language: str = "zh"
    task: str = "transcribe"
    min_speech_duration_ms: int = 500
    max_speech_duration_s: float = 30.0
    target_sr: Optional[int] = 16000


class AcousticFeatures(BaseDoc):
    sample_rate: int
    audio_duration: float = 0.0
    audio_timestamps: dict
    audio: dict


class Response(BaseDoc):
    text: dict
    audio_duration: float = 0.0


class AudioProcessor(Executor):
    def __init__(self, model_name_or_path: str = "models/silero_vad.onnx", *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load VAD Model
        self.model = VADOnnxWrapper(model_name_or_path)

    @requests
    async def load_process_audio(self, docs: DocList[AudioURL], **kwargs) -> DocList[AcousticFeatures]:
        feature_docs = DocList[AcousticFeatures]()

        for doc in docs:
            wav, sample_rate = read_audio(
                doc.audio_url, sampling_rate=doc.target_sr)
            duration = len(wav["channel_0"]) / sample_rate

            audio_timestamps = {"channel_0": None, "channel_1": None}
            for channel, audio in wav.items():
                if audio is not None:
                    audio_timestamps[channel] = get_speech_timestamps(
                        audio, self.model,
                        sampling_rate=sample_rate,
                        min_speech_duration_ms=doc.min_speech_duration_ms,
                        max_speech_duration_s=doc.max_speech_duration_s,
                    )

            feature_docs.append(AcousticFeatures(
                audio=wav,
                audio_timestamps=audio_timestamps,
                sample_rate=sample_rate,
                audio_duration=duration
            ))

        return feature_docs


class WhisperExecutor(Executor):
    def __init__(self,
                 model_name_or_path: str = "models/whisper_cpu_int8_cpu-cpu_model.onnx",
                 num_threads: int = 4,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        options = onnxruntime.SessionOptions()
        options.register_custom_ops_library(get_library_path())
        options.intra_op_num_threads = num_threads
        options.inter_op_num_threads = num_threads
        options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load ASR Model
        self.model = onnxruntime.InferenceSession(
            model_name_or_path, options, providers=["CPUExecutionProvider"])

    @requests
    async def transcribe(self, docs: DocList[AcousticFeatures], **kwargs) -> DocList[Response]:
        result_docs = DocList[Response]()

        for doc in docs:
            results = {"channel_0": "", "channel_1": ""}

            for channel in doc.audio_timestamps:
                transcripts = []

                if doc.audio[channel] is not None:
                    for tss in doc.audio_timestamps[channel]:
                        audio_clip = doc.audio[channel][tss["start"]:tss["end"]]

                        inputs = {
                            "audio_pcm": np.asarray([audio_clip]),
                            "decoder_input_ids": np.asarray([[50258, 50260, 50359, 50363]], dtype=np.int32),
                            "max_length": np.array([128], dtype=np.int32),
                            "min_length": np.array([1], dtype=np.int32),
                            "num_beams": np.array([3], dtype=np.int32),
                            "num_return_sequences": np.array([1], dtype=np.int32),
                            "length_penalty": np.array([1.0], dtype=np.float32),
                            "repetition_penalty": np.array([1.0], dtype=np.float32),
                            # "attention_mask": np.zeros((1, 80, 3000), dtype=np.int32),
                        }

                        outputs = self.model.run(None, inputs)[0][0][0]
                        if outputs != "":
                            transcripts.append(outputs)

                    results[channel] = ",".join(transcripts)

            result_docs.append(
                Response(text=results, audio_duration=doc.audio_duration))

        return result_docs


def main():
    f = Flow(port=12345).add(name='preprocessor', uses=AudioProcessor, replicas=2).add(
        name="transcriber", uses=WhisperExecutor, replicas=2)

    data = [AudioURL(
        audio_url="http://122.13.6.240:8881/4/4001012265/20231101/R009a339f10399_20231101134821.wav")]

    with f:
        docs = f.post(on='/', inputs=data,
                      return_type=DocList[Response])
        print(docs[0])


if __name__ == "__main__":
    main()
