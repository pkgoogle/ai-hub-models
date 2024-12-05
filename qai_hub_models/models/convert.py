import argparse
import importlib
import os
import pkgutil
import traceback

import ai_edge_torch
import torch


SEGFAULT_EXCEPTIONS = ["aotgan", "ddrnet23_slim", "mediapipe_selfie"]


def convert_model(model_cls, model_name):
    try:
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
    except Exception as e:
        with open("conversion_logs/" + model_name + ".log", 'a') as f:
            f.write(f"Error: {e}\n")
            traceback.print_exc(file=f)
        print(e)
        pass

def convert_module(module_name):
    model_module = importlib.import_module("qai_hub_models.models." + module_name)
    if module_name == "openai_clip":
        text_encoder_cls = getattr(model_module, "ClipTextEncoder")
        image_encoder_cls = getattr(model_module, "ClipImageEncoder")
        convert_model(text_encoder_cls, "ClipTextEncoder")
        convert_model(image_encoder_cls, "ClipImageEncoder")
    elif module_name.startswith("whisper_"):
        whisper_encoder_cls = getattr(model_module, "WhisperEncoderInf")
        whisper_decoder_cls = getattr(model_module, "WhisperDecoderInf")
        convert_model(whisper_encoder_cls, "WhisperEncoderInf")
        convert_model(whisper_decoder_cls, "WhisperDecoderInf")
    elif module_name == "llama_v2_7b_chat_quantized":
        llama2_cls = getattr(model_module, "Llama2_PromptProcessor_1_Quantized")
        convert_model(llama2_cls, "Llama2_PromptProcessor_1_Quantized")
    elif module_name == "mediapipe_face":
        face_detector_cls = getattr(model_module, "FaceDetector")
        face_landmark_detector_cls = getattr(model_module, "FaceLandmarkDetector")
        convert_model(face_detector_cls, "FaceDetector")
        convert_model(face_landmark_detector_cls, "FaceLandmarkDetector")
    elif module_name == "mediapipe_face_quantized":
        face_detector_cls = getattr(model_module, "FaceDetectorQuantizable")
        face_landmark_detector_cls = getattr(model_module, "FaceLandmarkDetectorQuantizable")
        convert_model(face_detector_cls, "FaceDetectorQuantizable")
        convert_model(face_landmark_detector_cls, "FaceLandmarkDetectorQuantizable")
    elif module_name == "mediapipe_hand":
        hand_detector_cls = getattr(model_module, "HandDetector")
        hand_landmark_detector_cls = getattr(model_module, "HandLandmarkDetector")
        convert_model(hand_detector_cls, "HandDetector")
        convert_model(hand_landmark_detector_cls, "HandLandmarkDetector")
    elif module_name == "mediapipe_pose":
        pose_detector_cls = getattr(model_module, "PoseDetector")
        pose_landmark_detector_cls = getattr(model_module, "PoseLandmarkDetector")
        convert_model(pose_detector_cls, "PoseDetector")
        convert_model(pose_landmark_detector_cls, "PoseLandmarkDetector")
    elif module_name == "trocr":
        encoder_cls = getattr(model_module, "TrOCREncoder")
        decoder_cls = getattr(model_module, "TrOCRDecoder")
        convert_model(encoder_cls, "TrOCREncoder")
        convert_model(decoder_cls, "TrOCRDecoder")
    else:
        model_cls = getattr(model_module, "Model")
        convert_model(model_cls, module_name)

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

            convert_module(model_name)
    else:
        model_name = args.modelname
        convert_module(model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts Qualcomm Models to TFLite using AI-Edge-Torch")

    parser.add_argument("modelname", help="Input modelname, can be all")
    parser.add_argument("-s", "--start", help="If modelname==all, which model to start converting. Otherwise, ignored.")
    parser.add_argument("-e", "--end", help="If modelname==all, which model to end converting. Otherwise, ignored. Follows range convention. Do not use if you want to convert to the end")

    args = parser.parse_args()
    convert(args)
