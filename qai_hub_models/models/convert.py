import argparse
import importlib
import os
import pkgutil
import traceback

import ai_edge_torch
import torch


SEGFAULT_EXCEPTIONS = ["aotgan", "ddrnet23_slim"]


def convert_model(model_name):
    model_module = importlib.import_module("qai_hub_models.models." + model_name)
    model_cls = getattr(model_module, "Model")

    constructor = getattr(model_cls, "from_pretrained")
    ctor_kwargs = {}
    if model_name.startswith("llama_v3"):
        ctor_kwargs["sequence_length"] = 128

    model = constructor(**ctor_kwargs)

    input_spec = model.get_input_spec()
    sample_kwargs = {}

    for arg_name, (shape, dtype) in input_spec.items():
        sample_kwargs[arg_name] = torch.randn(shape, dtype=getattr(torch, dtype))

    edge_model = ai_edge_torch.convert(model.eval(), sample_kwargs=sample_kwargs)
    edge_model.export("conversions/" + model_name + ".tflite")


def convert(args):
    os.makedirs("conversions", exist_ok=True)
    os.makedirs("conversion_logs", exist_ok=True)

    if args.modelname == "all":

        start_model = None
        if args.start:
            start_model = args.start
            started = False
        else:
            started = True

        for module_info in pkgutil.iter_modules(["qai_hub_models/models"]):
            if not module_info.ispkg:
                continue

            if module_info.name.startswith('_'):
                continue

            model_name = module_info.name

            if args.end and model_name == args.end:
                break

            if not started:
                if model_name == start_model:
                    started = True
                else:
                    continue

            if model_name in SEGFAULT_EXCEPTIONS:
                continue # skip for now

            with open("conversion_logs/" + model_name + ".log", 'a') as f:
                try:
                    convert_model(model_name)
                except Exception as e:
                    f.write(f"Error: {e}\n")
                    traceback.print_exc(file=f)
                    pass
    else:
        model_name = args.modelname
        convert_model(model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts Qualcomm Models to TFLite using AI-Edge-Torch")

    parser.add_argument("modelname", help="Input modelname, can be all")
    parser.add_argument("-s", "--start", help="If modelname==all, which model to start converting. Otherwise, ignored.")
    parser.add_argument("-e", "--end", help="If modelname==all, which model to end converting. Otherwise, ignored. Follows range convention. Do not use if you want to convert to the end")

    args = parser.parse_args()
    convert(args)
