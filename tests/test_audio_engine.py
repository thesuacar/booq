from pathlib import Path
from typing import List

import pytest

from audio_engine import (
    AudioClip,
    average_generation_time,
    export_generation_report,
    synthesize_captions,
)


def test_synthesize_captions_creates_audio_files(tmp_path: Path, stub_gtts: List[dict]) -> None:
    captions = ["hello world", "second caption"]

    clips = synthesize_captions(captions, output_dir=tmp_path, prefix="unit")

    assert len(clips) == 2
    for clip in clips:
        assert clip.path.exists()
        assert clip.caption in captions


def test_average_generation_time_computes_mean() -> None:
    clips = [
        AudioClip(caption="a", path=Path("a.mp3"), generation_time=0.5),
        AudioClip(caption="b", path=Path("b.mp3"), generation_time=1.0),
    ]
    assert average_generation_time(clips) == pytest.approx(0.75, rel=1e-3)


def test_average_generation_time_with_no_clips_returns_zero() -> None:
    assert average_generation_time([]) == 0.0


def test_export_generation_report_writes_expected_payload(tmp_path: Path) -> None:
    clips = [
        AudioClip(caption="c1", path=tmp_path / "c1.mp3", generation_time=0.1),
        AudioClip(caption="c2", path=tmp_path / "c2.mp3", generation_time=0.2),
    ]
    for clip in clips:
        clip.path.write_bytes(b"audio")

    report_path = tmp_path / "report.json"
    payload = export_generation_report(clips, output_path=report_path, extra_metadata={"foo": "bar"})

    assert report_path.exists()
    assert payload["foo"] == "bar"
    assert len(payload["clips"]) == 2
