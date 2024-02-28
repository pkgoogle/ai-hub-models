[![Qualcomm® AI Hub Models](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/quic-logo.jpg)](../../README.md)


# [MediaPipe-Face-Detection: Detect faces and locate facial features in real-time video and image streams](https://aihub.qualcomm.com/models/mediapipe_face)

Designed for sub-millisecond processing, this model predicts bounding boxes and pose skeletons (left eye, right eye, nose tip, mouth, left eye tragion, and right eye tragion) of faces in an image.

This is based on the implementation of MediaPipe-Face-Detection found
[here](https://github.com/zmurez/MediaPipePyTorch/). This repository contains scripts for optimized on-device
export suitable to run on Qualcomm® devices. More details on model performance
accross various devices, can be found [here](https://aihub.qualcomm.com/models/mediapipe_face).

[Sign up](https://aihub.qualcomm.com/) for early access to run these models on
a hosted Qualcomm® device.


## Example & Usage

Install the package via pip:
```bash
pip install "qai_hub_models[mediapipe_face]"
```


Once installed, run the following simple CLI demo:

```bash
python -m qai_hub_models.models.mediapipe_face.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../#qai-hub-models) for more usage instructions.

## Export for on-device deployment

This repository contains export scripts that produce a model optimized for
on-device deployment. This can be run as follows:

```bash
python -m qai_hub_models.models.mediapipe_face.export
```
Additional options are documented with the `--help` option. Note that the above
script requires access to Deployment instructions for Qualcomm® AI Hub.

## License
- The license for the original implementation of MediaPipe-Face-Detection can be found
  [here](https://github.com/zmurez/MediaPipePyTorch/blob/master/LICENSE).
- The license for the compiled assets for on-device deployment can be found [here](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf).

## References
* [BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs](https://arxiv.org/abs/1907.05047)
* [Source Model Implementation](https://github.com/zmurez/MediaPipePyTorch/)

## Community
* Join [our AI Hub Slack community](https://join.slack.com/t/qualcomm-ai-hub/shared_invite/zt-2dgf95loi-CXHTDRR1rvPgQWPO~ZZZJg) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).

