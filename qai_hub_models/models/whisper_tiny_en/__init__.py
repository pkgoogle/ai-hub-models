# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.whisper.app import WhisperApp as App  # noqa: F401

from .model import MODEL_ID  # noqa: F401
from .model import WhisperTinyEn as Model  # noqa: F401
from qai_hub_models.models._shared.whisper.model import WhisperEncoderInf
from qai_hub_models.models._shared.whisper.model import WhisperDecoderInf
