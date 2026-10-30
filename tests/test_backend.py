import warnings
from types import SimpleNamespace

import pytest

from wakewordkit._backend import _validate_session, create_model


def session(input_shape, output_shape, input_type="tensor(float)", extra=False):
    inputs = [SimpleNamespace(shape=input_shape, type=input_type)]
    if extra:
        inputs *= 2
    return SimpleNamespace(
        get_inputs=lambda: inputs,
        get_outputs=lambda: [SimpleNamespace(shape=output_shape, type="tensor(float)")],
    )


@pytest.mark.parametrize(
    "input_shape,output_shape,kind,extra",
    [
        ([1, 16, 96], [1, 2], "tensor(float)", False),
        ([1, 16, 95], [1, 1], "tensor(float)", False),
        ([1, 121, 96], [1, 1], "tensor(float)", False),
        ([1, 16, 96], [1, 1], "tensor(int16)", False),
        ([1, 16, 96], [1, 1], "tensor(float)", True),
    ],
)
def test_general_or_multiclass_models_are_rejected(
    input_shape, output_shape, kind, extra
):
    with pytest.raises(ValueError, match="binary openWakeWord"):
        _validate_session(session(input_shape, output_shape, kind, extra), "custom")


def test_backend_initialization_warnings_remain_visible(monkeypatch):
    import openwakeword.model

    def load(**kwargs):
        warnings.warn("provider warning", UserWarning, stacklevel=2)
        return SimpleNamespace(models={"custom": session([1, 16, 96], [1, 1])})

    monkeypatch.setattr(openwakeword.model, "Model", load)
    with pytest.warns(UserWarning, match="provider warning"):
        create_model(["custom.onnx"], 0.0)
