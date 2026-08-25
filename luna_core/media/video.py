"""Video helpers: probe, poster frame, transcode — thin, honest wrappers over
``ffprobe``/``ffmpeg`` for a host's media pipeline.

Pure functions over files. They know nothing about a host's media rows or
storage: the host downloads to a temp path, calls these, uploads what comes
back. Every call shells out synchronously — run them in a thread
(``asyncio.to_thread``) or a worker, never on the event loop.

The playability opinion lives in ``needs_transcode``: an H.264 + AAC MP4 with
``yuv420p`` pixels at 1080p or less plays everywhere (iOS, Android, browsers,
``expo-video``); anything else (an iPhone's HEVC ``.mov``, 4K, 10-bit HDR,
WebM) is re-encoded once, in the background, and swapped in.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class VideoToolError(RuntimeError):
    """ffmpeg/ffprobe is missing, refused the file, or timed out."""


@dataclass(frozen=True)
class VideoInfo:
    """What ``ffprobe`` says about the first video stream, with the rotation
    tag applied so ``width``/``height`` are the DISPLAY dimensions — a phone
    video is often stored landscape with a 90° rotate flag."""

    codec: str | None  # "h264" | "hevc" | "vp9" | ...
    audio_codec: str | None  # "aac" | None (no audio) | other
    container: str  # "mp4" | "mov" | "webm" | "other"
    pix_fmt: str | None  # "yuv420p" is what plays everywhere
    width: int
    height: int
    duration: float | None


@dataclass(frozen=True)
class TranscodeThresholds:
    """Above either of these the file is re-encoded even if the codecs are
    fine: a 4K clip is heavy to serve and to play, and past the size limit a
    smaller copy is worth the CPU."""

    max_long_edge: int = 1920
    size_bytes: int = 25 * 1024 * 1024


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _run(cmd: list[str], *, timeout_s: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            cmd, check=True, capture_output=True, timeout=timeout_s
        )
    except FileNotFoundError as exc:
        raise VideoToolError(f"{cmd[0]} is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoToolError(f"{cmd[0]} timed out after {timeout_s:.0f}s") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        raise VideoToolError(
            f"{cmd[0]} failed ({exc.returncode}): {detail[-1] if detail else 'no output'}"
        ) from exc


def probe(path: Path, *, timeout_s: float = 60) -> VideoInfo:
    """Read the streams. Raises ``VideoToolError`` when the file has no video
    stream (an audio file with a video mime, a corrupt upload)."""
    out = _run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        timeout_s=timeout_s,
    )
    data = json.loads(out.stdout or b"{}")
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise VideoToolError("no video stream")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    rotation = _rotation(video)
    if rotation in (90, 270):
        width, height = height, width

    fmt = data.get("format") or {}
    duration = _float(fmt.get("duration")) or _float(video.get("duration"))
    return VideoInfo(
        codec=video.get("codec_name"),
        audio_codec=audio.get("codec_name") if audio else None,
        container=_container(fmt),
        pix_fmt=video.get("pix_fmt"),
        width=width,
        height=height,
        duration=duration,
    )


def _rotation(stream: dict) -> int:
    tag = (stream.get("tags") or {}).get("rotate")
    if tag is not None:
        try:
            return abs(int(float(tag))) % 360
        except (TypeError, ValueError):
            pass
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            try:
                return abs(int(float(side["rotation"]))) % 360
            except (TypeError, ValueError):
                continue
    return 0


def _container(fmt: dict) -> str:
    names = set((fmt.get("format_name") or "").split(","))
    brand = ((fmt.get("tags") or {}).get("major_brand") or "").strip().lower()
    if "webm" in names or "matroska" in names:
        return "webm"
    if brand.startswith("qt") or names == {"mov"}:
        return "mov"
    if names & {"mp4", "mov", "m4a", "3gp", "3g2", "mj2"}:
        # ffprobe reports the whole family for any of them; the brand decides.
        return "mov" if brand.startswith("qt") else "mp4"
    return "other"


def _float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def poster_frame(path: Path, at_seconds: float = 1.0, *, timeout_s: float = 60) -> bytes:
    """One JPEG frame (upright — ffmpeg applies the rotation) at ``at_seconds``,
    or at the very start when the clip is shorter than that. Capped at 1568px
    on the long edge, which is what a vision model gets anyway."""
    for at in (at_seconds, 0.0):
        out = _run(
            [
                "ffmpeg", "-v", "error", "-y", "-ss", f"{at:.3f}", "-i", str(path),
                "-frames:v", "1", "-vf", "scale='min(1568,iw)':-2", "-q:v", "3",
                "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
            ],
            timeout_s=timeout_s,
        )
        if out.stdout:
            return out.stdout
    raise VideoToolError("could not extract a poster frame")


def needs_transcode(
    info: VideoInfo, size_bytes: int, thresholds: TranscodeThresholds | None = None
) -> bool:
    """False only for a file that already plays everywhere and is not heavy;
    everything else earns a background re-encode."""
    t = thresholds or TranscodeThresholds()
    return not (
        info.container == "mp4"
        and info.codec == "h264"
        and info.pix_fmt == "yuv420p"
        and info.audio_codec in (None, "aac")
        and max(info.width, info.height) <= t.max_long_edge
        and size_bytes <= t.size_bytes
    )


def transcode(
    src: Path,
    dst: Path,
    *,
    max_long_edge: int = 1920,
    crf: int = 28,
    preset: str = "veryfast",
    threads: int = 2,
    nice: int = 15,
    timeout_s: float = 1800,
) -> None:
    """Re-encode ``src`` into an H.264/AAC MP4 at ``dst`` (``+faststart`` so it
    plays while downloading), scaled down to ``max_long_edge`` on the long side
    with the aspect kept. ``nice``d and thread-capped: on a small box this runs
    next to everything else and must never starve it."""
    scale = (
        f"scale='if(gt(a,1),min({max_long_edge},iw),-2)':"
        f"'if(gt(a,1),-2,min({max_long_edge},ih))'"
    )
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-i", str(src),
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-vf", scale,
        "-c:a", "aac", "-b:a", "128k", "-ac", "2",
        "-movflags", "+faststart", "-threads", str(threads),
        str(dst),
    ]
    if nice and shutil.which("nice"):
        cmd = ["nice", "-n", str(nice), *cmd]
    _run(cmd, timeout_s=timeout_s)
    if not dst.exists() or dst.stat().st_size == 0:
        raise VideoToolError("transcode produced no output")


__all__ = [
    "TranscodeThresholds",
    "VideoInfo",
    "VideoToolError",
    "ffmpeg_available",
    "needs_transcode",
    "poster_frame",
    "probe",
    "transcode",
]
