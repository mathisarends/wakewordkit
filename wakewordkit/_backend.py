"""The version-pinned openWakeWord integration boundary (loaded on construction)."""

from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class Backend(Protocol):
    def predict(self, audio: NDArray[np.int16]) -> Mapping[str, float]: ...


def _validate_session(session: Any, name: str) -> None:
    inputs, outputs = session.get_inputs(), session.get_outputs()
    message = (
        f"Model {name!r} must be a binary openWakeWord ONNX classifier: "
        "one float32 input [1, frames, 96] (1 <= frames <= 120) and "
        "one float32 output [1, 1]. General and multiclass ONNX models are unsupported."
    )
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(message)
    shape, output = inputs[0].shape, outputs[0].shape
    if (
        inputs[0].type != "tensor(float)"
        or outputs[0].type != "tensor(float)"
        or len(shape) != 3
        or shape[0] != 1
        or not isinstance(shape[1], int)
        or not 1 <= shape[1] <= 120
        or shape[2] != 96
        or output != [1, 1]
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
        _validate_session(session, name)
    return model
