from optimum.onnxruntime import (
    ORTModelForSpeechSeq2Seq,
    AutoOptimizationConfig,
    ORTOptimizer,
)

model_id = "Oblivion208/whisper-base-cantonese"
save_dir = "models/optimized_model"

# Export model in ONNX
model = ORTModelForSpeechSeq2Seq.from_pretrained(model_id, export=True)

# Create the optimizer
optimizer = ORTOptimizer.from_pretrained(model)

# Define the optimization strategy by creating the appropriate configuration
optimization_config = AutoOptimizationConfig.O3()

# Optimize the model
optimizer.optimize(save_dir=save_dir, optimization_config=optimization_config)
