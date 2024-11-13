import torch
import ai_edge_torch

from qai_hub_models.models.convnext_tiny import Model
from qai_hub_models.utils.args import get_model_kwargs


model = Model.from_pretrained(**get_model_kwargs(Model, {}))

sample_inputs = (torch.randn(1, 3, 512, 512),)

edge_model = ai_edge_torch.convert(model.eval(), sample_inputs)
edge_model.export("convnext_tiny.tflite")
