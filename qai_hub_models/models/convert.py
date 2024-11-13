import argparse
import importlib
import pkgutil

import ai_edge_torch
import torch


"""
CONFIG = {
    "aotgan": {
        "constructor_name": "from_pretrained",
    },
    "convnext_tiny": {
        "constructor_name": "from_pretrained",
    },
}
"""


def convert_model(model_name):
    model_module = importlib.import_module("qai_hub_models.models." + model_name)
    model_cls = getattr(model_module, "Model")

    # constructor_name = CONFIG[model_name]["constructor_name"]
    # if constructor_name == "from_pretrained":
    constructor = getattr(model_cls, "from_pretrained")
    model = constructor()

    input_spec = model.get_input_spec()
    sample_kwargs = {}

    for arg_name, (shape, dtype) in input_spec.items():
        sample_kwargs[arg_name] = torch.randn(shape, dtype=getattr(torch, dtype))

    edge_model = ai_edge_torch.convert(model.eval(), sample_kwargs=sample_kwargs)
    edge_model.export(model_name + ".tflite")


def convert(args):
    if args.modelname == "all":
        for module_info in pkgutil.iter_modules(["qai_hub_models/models"]):
            if not module_info.ispkg:
                continue

            if module_info.name.startswith('_'):
                continue

            convert_model(module_info.name)
    else:
        model_name = args.modelname
        convert_model(model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts Qualcomm Models to TFLite using AI-Edge-Torch")

    parser.add_argument("modelname", help="Input modelname, can be all")

    args = parser.parse_args()
    convert(args)