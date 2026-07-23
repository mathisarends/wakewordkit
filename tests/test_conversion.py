import numpy as np
import pytest

from wakewordkit.detector import _as_int16


def test_bytes_are_read_as_int16():
    result = _as_int16(b"\x01\x00\x02\x00")
    assert result.dtype == np.int16
    assert result.tolist() == [1, 2]


@pytest.mark.parametrize("wrapper", [bytes, bytearray, memoryview])
def test_bytes_like_inputs(wrapper):
    result = _as_int16(wrapper(b"\x05\x00"))
    assert result.tolist() == [5]


def test_int16_array_passes_through_unchanged():
    array = np.array([1, -2, 3], dtype=np.int16)
    assert _as_int16(array) is array


def test_float_array_is_scaled_to_int16():
    result = _as_int16(np.array([0.0, 1.0, -1.0], dtype=np.float32))
    assert result.dtype == np.int16
    assert result.tolist() == [0, 32767, -32767]


def test_other_integer_dtype_is_cast():
    result = _as_int16(np.array([7, 8], dtype=np.int32))
    assert result.dtype == np.int16
    assert result.tolist() == [7, 8]
