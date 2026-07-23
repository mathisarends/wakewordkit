import pytest


class FakeModel:
    def __init__(self, wakeword_model_paths, vad_threshold=0.0, **kwargs):
        self.paths = wakeword_model_paths
        self.vad_threshold = vad_threshold
        self.scores: dict[str, float] = {}
        self.reset_calls = 0

    def predict(self, audio):
        return self.scores

    def reset(self):
        self.reset_calls += 1


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr("wakewordkit.detector.Model", FakeModel)
    return FakeModel
