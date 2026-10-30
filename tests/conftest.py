import pytest


class FakeModel:
    def __init__(self, wakeword_model_paths, vad_threshold=0.0, **kwargs):
        self.paths = wakeword_model_paths
        self.vad_threshold = vad_threshold
        self.scores: dict[str, float] = {}
        self.score_sequence = []
        self.inputs = []
        self.reset_calls = 0

    def predict(self, audio):
        assert audio.dtype.name == "int16" and audio.shape == (1280,)
        self.inputs.append(audio.copy())
        if self.score_sequence:
            return self.score_sequence.pop(0)
        return self.scores

    def reset(self):
        self.reset_calls += 1


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr("wakewordkit.detector.create_model", FakeModel)
    return FakeModel
