import torch
import ai_edge_torch

from qai_hub_models.models.controlnet_quantized import Model


model = Model.from_precompiled()

sample_inputs = (
    torch.randn(1, 64, 64, 4),
    torch.randn(1, 1280),
    torch.randn(1, 77, 768),
    torch.randn(1, 512, 512, 3),
)

edge_model = ai_edge_torch.convert(model, sample_inputs)
edge_model.export("controlnet_q.tflite")
