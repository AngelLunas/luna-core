"""Media helpers a host can build on — pure functions over files, no database.

``video``: probe, poster frame, "does this need transcoding", transcode — the
ffmpeg/ffprobe wrappers a host's media pipeline calls, so every app on
luna-core shares one opinion about what a playable video is.
"""
