Heavily borrowed from [Olive](https://github.com/microsoft/Olive).

```bash
python prepare_whisper_configs.py --model_name Oblivion208/whisper-base-cantonese --no_audio_decoder --multilingual
python -m olive.workflows.run --config whisper_cpu_int8.json --setup
python -m olive.workflows.run --config whisper_cpu_int8.json
```
