"""``luna_core.media.video`` against real ffmpeg (skipped where it is absent)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from luna_core.media import video

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg not installed",
)


def _clip(path: Path, *, codec: str = "libx264", size: str = "320x240") -> Path:
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=duration=2:size={size}:rate=10",
            "-f", "lavfi", "-i", "sine=d=2",
            "-c:v", codec, "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
        ],
        check=True, capture_output=True,
    )
    return path


def test_probe_reads_the_streams(tmp_path):
    info = video.probe(_clip(tmp_path / "a.mp4"))
    assert info.codec == "h264" and info.audio_codec == "aac"
    assert info.container == "mp4" and info.pix_fmt == "yuv420p"
    assert (info.width, info.height) == (320, 240)
    assert info.duration and 1.5 < info.duration < 2.5


def test_poster_frame_is_a_jpeg(tmp_path):
    frame = video.poster_frame(_clip(tmp_path / "a.mp4"))
    assert frame[:2] == b"\xff\xd8"


def test_a_playable_small_mp4_needs_no_transcode(tmp_path):
    clip = _clip(tmp_path / "a.mp4")
    info = video.probe(clip)
    assert not video.needs_transcode(info, clip.stat().st_size)
    assert video.needs_transcode(info, 30 * 1024 * 1024)  # heavy → yes


def test_a_tall_or_odd_codec_clip_is_transcoded_to_a_playable_mp4(tmp_path):
    src = _clip(tmp_path / "big.mp4", size="2560x1440")
    info = video.probe(src)
    assert video.needs_transcode(info, src.stat().st_size)
    dst = tmp_path / "out.mp4"
    video.transcode(src, dst, max_long_edge=640, threads=1)
    out = video.probe(dst)
    assert out.codec == "h264" and out.container == "mp4"
    assert max(out.width, out.height) == 640
    assert not video.needs_transcode(out, dst.stat().st_size)


def test_probe_refuses_a_file_without_video(tmp_path):
    bad = tmp_path / "x.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(video.VideoToolError):
        video.probe(bad)
