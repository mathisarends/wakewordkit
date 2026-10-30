"""The version-pinned openWakeWord integration boundary (loaded on construction)."""

from collections.abc import Mapping
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray


class Backend(Protocol):
    def predict(self, audio: NDArray[np.int16]) -> Mapping[str, float]: ...


def _validate_session(session: Any, name: str, available_frames: int = 116) -> None:
    inputs, outputs = session.get_inputs(), session.get_outputs()
    message = (
        f"Model {name!r} must be a binary openWakeWord ONNX classifier: "
        f"one float32 input [batch, frames, 96] (1 <= frames <= {available_frames}) "
        "and one float32 output [batch, 1], with batch=1 or dynamic. "
        "General and multiclass ONNX models are unsupported."
    )
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(message)
    shape, output = inputs[0].shape, outputs[0].shape
    if (
        inputs[0].type != "tensor(float)"
        or outputs[0].type != "tensor(float)"
        or len(shape) != 3
        or not _supports_batch_one(shape[0])
        or not isinstance(shape[1], int)
        or not 1 <= shape[1] <= available_frames
        or shape[2] != 96
        or len(output) != 2
        or not _supports_batch_one(output[0])
        or output[1] != 1
    ):
        raise ValueError(message)


def create_model(wakeword_model_paths: list[str], vad_threshold: float) -> Backend:
    from openwakeword.model import Model

    try:
        model = Model(
            wakeword_model_paths=wakeword_model_paths, vad_threshold=vad_threshold
        )
    except Exception as exc:
        raise ValueError(
            "Could not load binary openWakeWord-compatible ONNX models: "
            + ", ".join(wakeword_model_paths)
        ) from exc
    for name, session in model.models.items():
        _validate_session(
            session, name, min(120, model.preprocessor.feature_buffer.shape[0])
        )
    return cast(Backend, model)


def _supports_batch_one(dimension: Any) -> bool:
    return dimension is None or isinstance(dimension, str) or dimension == 1
