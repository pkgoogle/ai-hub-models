import argparse
import importlib
import os
import pkgutil
import sys

import ai_edge_torch
import torch


SEGFAULT_EXCEPTIONS = ["aotgan", "ddrnet23_slim"]


def convert_model(model_name):
    model_module = importlib.import_module("qai_hub_models.models." + model_name)
    model_cls = getattr(model_module, "Model")

    constructor = getattr(model_cls, "from_pretrained")
    model = constructor()

    input_spec = model.get_input_spec()
    sample_kwargs = {}

    for arg_name, (shape, dtype) in input_spec.items():
        sample_kwargs[arg_name] = torch.randn(shape, dtype=getattr(torch, dtype))

    edge_model = ai_edge_torch.convert(model.eval(), sample_kwargs=sample_kwargs)
    edge_model.export("conversions/" + model_name + ".tflite")


def convert(args):
    os.makedirs("conversions", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    if args.modelname == "all":
        for module_info in pkgutil.iter_modules(["qai_hub_models/models"]):
            if not module_info.ispkg:
                continue

            if module_info.name.startswith('_'):
                continue

            model_name = module_info.name

            if model_name in SEGFAULT_EXCEPTIONS:
                continue # skip for now

            original_stdout = sys.stdout
            original_stderr = sys.stderr

            with open("logs/" + model_name + ".log", 'a') as f:
                sys.stdout = f
                sys.stderr = f

                try:
                    convert_model(model_name)
                except:
                    pass

            sys.stdout = original_stdout
            sys.stderr = original_stderr
    else:
        model_name = args.modelname
        convert_model(model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts Qualcomm Models to TFLite using AI-Edge-Torch")

    parser.add_argument("modelname", help="Input modelname, can be all")

    args = parser.parse_args()
    convert(args)