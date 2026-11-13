#!/usr/bin/env python3
"""
VoicePanel - 4-Track Audio Engine
Fast NumPy-powered 4-track mixing, waveform peak decimation, microphone capture
(PipeWire/PulseAudio parec with PyAudio/arecord fallbacks), non-blocking playback,
range editing with Undo/Redo, and .wav/.mp3/.flac/.ogg/.vpanel I/O.
"""

import io
import json
import os
import shutil
import struct
import subprocess
import threading
import time
import wave
import zipfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


DEFAULT_SAMPLE_RATE = 48000
NUM_TRACKS = 4
TRACK_COLORS = [
    "#00f0ff",  # Track 1: Neon Cyan
    "#38bdf8",  # Track 2: Electric Sky
    "#a855f7",  # Track 3: Ultraviolet Neon
    "#10b981",  # Track 4: Emerald Pulse
]

SUPPORTED_AUDIO_EXTS = (".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".opus", ".wma")


def empty_stereo(num_samples: int = 0) -> np.ndarray:
    if num_samples <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.zeros((num_samples, 2), dtype=np.float32)


def ensure_stereo_float32(arr: np.ndarray) -> np.ndarray:
    """Normalize any 1D/2D audio array into contiguous float32 shape (N, 2)."""
    if arr is None:
        return empty_stereo(0)
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        a = np.column_stack((a, a))
    elif a.ndim == 2:
        if a.shape[1] == 1:
            a = np.repeat(a, 2, axis=1)
        elif a.shape[1] > 2:
            a = a[:, :2]
    else:
        return empty_stereo(0)
    return np.ascontiguousarray(a, dtype=np.float32)


def resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linearly resample stereo float32 audio from src_rate to dst_rate cleanly with NumPy."""
    samples = ensure_stereo_float32(samples)
    if src_rate <= 0 or dst_rate <= 0 or src_rate == dst_rate or len(samples) < 2:
        return samples
    new_len = max(1, int(round(len(samples) * float(dst_rate) / float(src_rate))))
    x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False, dtype=np.float64)
    left = np.interp(x_new, x_old, samples[:, 0]).astype(np.float32)
    right = np.interp(x_new, x_old, samples[:, 1]).astype(np.float32)
    return np.ascontiguousarray(np.column_stack((left, right)), dtype=np.float32)


def wav_bytes_from_float32(samples: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
    """Encode stereo float32 array as standard 16-bit PCM WAV bytes."""
    s = ensure_stereo_float32(samples)
    clipped = np.clip(s, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def float32_from_wav_bytes(data: bytes, target_rate: int = DEFAULT_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Decode uncompressed PCM WAV (8/16/24/32-bit, mono/stereo) into stereo float32."""
    with wave.open(io.BytesIO(data), "rb") as wf:
        n_ch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        src_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if n_frames == 0 or not raw:
        return empty_stereo(0), target_rate

    if sampwidth == 1:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        signed = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int8).astype(np.int32) << 16)
        )
        arr = signed.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth * 8}-bit")

    if n_ch > 1:
        usable = (len(arr) // n_ch) * n_ch
        arr = arr[:usable].reshape(-1, n_ch)
    stereo = ensure_stereo_float32(arr)
    if src_rate != target_rate:
        stereo = resample_linear(stereo, src_rate, target_rate)
    return stereo, target_rate


@dataclass
class AudioTrack:
    """Single audio track inside VoicePanel's 4-track timeline."""
    index: int
    name: str = ""
    samples: np.ndarray = field(default_factory=lambda: empty_stereo(0))
    offset_samples: int = 0
    volume: float = 1.0
    pan: float = 0.0
    muted: bool = False
    solo: bool = False
    armed: bool = False
    color_hex: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"Track {self.index + 1}"
        if not self.color_hex:
            self.color_hex = TRACK_COLORS[self.index % len(TRACK_COLORS)]
        self.samples = ensure_stereo_float32(self.samples)
        self.offset_samples = max(0, int(self.offset_samples))

    @property
    def duration_samples(self) -> int:
        return int(len(self.samples))

    @property
    def end_samples(self) -> int:
        return int(self.offset_samples + len(self.samples))

    def clone_state(self) -> dict:
        return {
            "name": self.name,
            "samples": self.samples.copy(),
            "offset_samples": int(self.offset_samples),
            "volume": float(self.volume),
            "pan": float(self.pan),
            "muted": bool(self.muted),
            "solo": bool(self.solo),
        }

    def restore_state(self, state: dict):
        self.name = str(state["name"])
        self.samples = ensure_stereo_float32(state["samples"].copy())
        self.offset_samples = max(0, int(state["offset_samples"]))
        self.volume = float(state["volume"])
        self.pan = float(state["pan"])
        self.muted = bool(state["muted"])
        self.solo = bool(state["solo"])

    def peak_envelope(
        self,
        view_start: int,
        view_end: int,
        num_bins: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-column (min, max) mono amplitude envelope for [view_start, view_end)."""
        num_bins = max(1, int(num_bins))
        mins = np.zeros(num_bins, dtype=np.float32)
        maxs = np.zeros(num_bins, dtype=np.float32)
        if len(self.samples) == 0 or view_end <= view_start:
            return mins, maxs

        t_start = self.offset_samples
        t_end = self.end_samples
        ov_start = max(view_start, t_start)
        ov_end = min(view_end, t_end)
        if ov_end <= ov_start:
            return mins, maxs

        span = float(view_end - view_start)
        bin_edges = np.linspace(view_start, view_end, num_bins + 1)

        # Convert stereo to crisp mono mix for waveform display
        mono = 0.5 * (self.samples[:, 0] + self.samples[:, 1])

        b0 = max(0, int(np.floor((ov_start - view_start) * num_bins / span)))
        b1 = min(num_bins, int(np.ceil((ov_end - view_start) * num_bins / span)))
        if b1 <= b0:
            return mins, maxs

        s_starts = np.floor(bin_edges[b0:b1]).astype(np.int64) - t_start
        s_ends = np.ceil(bin_edges[b0 + 1:b1 + 1]).astype(np.int64) - t_start
        s_starts = np.clip(s_starts, 0, len(mono) - 1)
        s_ends = np.clip(np.maximum(s_ends, s_starts + 1), 1, len(mono))

        # Vectorized uniform-block reshape when contiguous bins cover >= 2 samples each cleanly
        counts = s_ends - s_starts
        min_count = int(counts.min()) if len(counts) else 0
        if min_count >= 2 and s_ends[-1] > s_starts[0]:
            stride = min_count
            sub_n = b1 - b0
            # Sample up to 64 evenly distributed points per bin for O(width) fixed render speed
            step = max(1, stride // 32)
            idx_offsets = np.arange(0, stride, step, dtype=np.int64)[None, :]
            grid = np.clip(s_starts[:, None] + idx_offsets, 0, len(mono) - 1)
            sampled = mono[grid]
            mins[b0:b1] = sampled.min(axis=1)
            maxs[b0:b1] = sampled.max(axis=1)
        else:
            for bi in range(b0, b1):
                seg = mono[s_starts[bi - b0]:s_ends[bi - b0]]
                if len(seg):
                    mins[bi] = float(seg.min())
                    maxs[bi] = float(seg.max())

        return mins, maxs


class MultiTrackSession:
    """4-track timeline model supporting mixdown, range edits, Undo/Redo, and file/project I/O."""

    MAX_UNDO = 40

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE):
        self.sample_rate = int(sample_rate)
        self.tracks: List[AudioTrack] = [AudioTrack(index=i) for i in range(NUM_TRACKS)]
        self.tracks[0].armed = True
        self.active_track_index = 0
        self.playhead_sample = 0
        self.selection: Optional[Tuple[int, int]] = None  # (start_sample, end_sample) in timeline coords
        self.clipboard: Optional[np.ndarray] = None
        self.project_path: Optional[str] = None
        self.modified = False
        self._undo_stack: List[dict] = []
        self._redo_stack: List[dict] = []

    def get_active_track(self) -> AudioTrack:
        return self.tracks[self.active_track_index]

    def get_armed_tracks(self) -> List[AudioTrack]:
        return [t for t in self.tracks if t.armed]

    def arm_track(self, index: int, exclusive: bool = True):
        index = max(0, min(index, NUM_TRACKS - 1))
        if exclusive:
            for t in self.tracks:
                t.armed = (t.index == index)
        else:
            self.tracks[index].armed = not self.tracks[index].armed
        self.active_track_index = index

    def total_samples(self) -> int:
        ends = [t.end_samples for t in self.tracks if t.duration_samples > 0]
        return max(ends) if ends else 0

    def total_seconds(self) -> float:
        return self.total_samples() / float(self.sample_rate)

    def _snapshot(self) -> dict:
        return {
            "active_track": self.active_track_index,
            "playhead": self.playhead_sample,
            "selection": self.selection,
            "tracks": [t.clone_state() for t in self.tracks],
        }

    def _restore_snapshot(self, snap: dict):
        self.active_track_index = int(snap["active_track"])
        self.playhead_sample = int(snap["playhead"])
        self.selection = snap["selection"]
        for t, ts in zip(self.tracks, snap["tracks"]):
            t.restore_state(ts)

    def push_undo(self):
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > self.MAX_UNDO:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self.modified = True

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._snapshot())
        snap = self._undo_stack.pop()
        self._restore_snapshot(snap)
        self.modified = True
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._snapshot())
        snap = self._redo_stack.pop()
        self._restore_snapshot(snap)
        self.modified = True
        return True

    # ------------------------------------------------------------------
    # Recording & Importing
    # ------------------------------------------------------------------
    def commit_recording(self, recorded_samples: np.ndarray, start_sample: int, track_index: Optional[int] = None) -> int:
        """Write microphone recording into target track starting at start_sample (punch-in safe)."""
        rec = ensure_stereo_float32(recorded_samples)
        if len(rec) == 0:
            return start_sample
        if track_index is None:
            armed = self.get_armed_tracks()
            track = armed[0] if armed else self.get_active_track()
        else:
            track = self.tracks[max(0, min(track_index, NUM_TRACKS - 1))]

        start_sample = max(0, int(start_sample))
        self.push_undo()

        rec_end = start_sample + len(rec)
        if track.duration_samples == 0:
            track.offset_samples = start_sample
            track.samples = rec
        else:
            new_offset = min(track.offset_samples, start_sample)
            new_end = max(track.end_samples, rec_end)
            buf = empty_stereo(new_end - new_offset)
            old_rel = track.offset_samples - new_offset
            buf[old_rel:old_rel + len(track.samples)] = track.samples
            rec_rel = start_sample - new_offset
            buf[rec_rel:rec_rel + len(rec)] = rec
            track.offset_samples = new_offset
            track.samples = buf

        self.playhead_sample = rec_end
        return rec_end

    def clear_track(self, track_index: int):
        track = self.tracks[max(0, min(track_index, NUM_TRACKS - 1))]
        if track.duration_samples == 0:
            return
        self.push_undo()
        track.samples = empty_stereo(0)
        track.offset_samples = 0

    def import_audio_file(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load any WAV/MP3/OGG/FLAC/M4A file into stereo float32 at session sample_rate."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".wav":
            try:
                with open(file_path, "rb") as f:
                    return float32_from_wav_bytes(f.read(), target_rate=self.sample_rate)
            except Exception:
                # Fall through to ffmpeg for IEEE-float / ADPCM / compressed WAV variants
                pass

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required to decode non-PCM audio formats.")

        proc = subprocess.run(
            [
                ffmpeg,
                "-v", "error",
                "-i", file_path,
                "-f", "f32le",
                "-acodec", "pcm_f32le",
                "-ac", "2",
                "-ar", str(self.sample_rate),
                "pipe:1",
            ],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Failed to decode {os.path.basename(file_path)}: {err or 'ffmpeg error'}")

        n_floats = len(proc.stdout) // 4
        if n_floats < 2:
            return empty_stereo(0), self.sample_rate
        usable_bytes = (n_floats // 2) * 8
        arr = np.frombuffer(proc.stdout[:usable_bytes], dtype="<f4").reshape(-1, 2).copy()
        return ensure_stereo_float32(arr), self.sample_rate

    def load_file_to_track(self, file_path: str, track_index: int, offset_sample: int = 0) -> float:
        samples, _ = self.import_audio_file(file_path)
        track = self.tracks[max(0, min(track_index, NUM_TRACKS - 1))]
        self.push_undo()
        track.samples = samples
        track.offset_samples = max(0, int(offset_sample))
        base = os.path.splitext(os.path.basename(file_path))[0]
        if base:
            track.name = base[:24]
        self.active_track_index = track.index
        return len(samples) / float(self.sample_rate)

    # ------------------------------------------------------------------
    # Timeline Range Editing (Undoable)
    # ------------------------------------------------------------------
    def _resolve_track_local_range(
        self,
        track: AudioTrack,
        use_full_if_empty: bool = True,
    ) -> Optional[Tuple[int, int]]:
        """Convert timeline selection onto track-local [lo, hi) sample indices."""
        if track.duration_samples == 0:
            return None
        if self.selection is not None:
            s0, s1 = sorted((int(self.selection[0]), int(self.selection[1])))
            lo = max(0, s0 - track.offset_samples)
            hi = min(track.duration_samples, s1 - track.offset_samples)
            if hi > lo:
                return (lo, hi)
        if use_full_if_empty:
            return (0, track.duration_samples)
        return None

    def cut_selection(self, track_index: Optional[int] = None, copy_to_clipboard: bool = True) -> bool:
        track = self.tracks[self.active_track_index if track_index is None else track_index]
        rng = self._resolve_track_local_range(track, use_full_if_empty=False)
        if rng is None:
            return False
        lo, hi = rng
        self.push_undo()
        if copy_to_clipboard:
            self.clipboard = track.samples[lo:hi].copy()
        track.samples = np.ascontiguousarray(
            np.concatenate((track.samples[:lo], track.samples[hi:]), axis=0),
            dtype=np.float32,
        )
        self.playhead_sample = track.offset_samples + lo
        self.selection = None
        return True

    def silence_selection(self, track_index: Optional[int] = None) -> bool:
        track = self.tracks[self.active_track_index if track_index is None else track_index]
        rng = self._resolve_track_local_range(track, use_full_if_empty=False)
        if rng is None:
            return False
        lo, hi = rng
        self.push_undo()
        track.samples[lo:hi] = 0.0
        return True

    def copy_selection(self, track_index: Optional[int] = None) -> bool:
        track = self.tracks[self.active_track_index if track_index is None else track_index]
        rng = self._resolve_track_local_range(track, use_full_if_empty=True)
        if rng is None:
            return False
        lo, hi = rng
        self.clipboard = track.samples[lo:hi].copy()
        return True

    def paste_at_playhead(self, track_index: Optional[int] = None) -> bool:
        if self.clipboard is None or len(self.clipboard) == 0:
            return False
        track = self.tracks[self.active_track_index if track_index is None else track_index]
        clip = ensure_stereo_float32(self.clipboard)
        pos = max(0, int(self.playhead_sample))
        self.push_undo()
        if track.duration_samples == 0:
            track.offset_samples = pos
            track.samples = clip.copy()
        elif pos <= track.offset_samples:
            gap = empty_stereo(track.offset_samples - pos)
            track.samples = np.ascontiguousarray(np.concatenate((clip, gap, track.samples), axis=0))
            track.offset_samples = pos
        elif pos >= track.end_samples:
            gap = empty_stereo(pos - track.end_samples)
            track.samples = np.ascontiguousarray(np.concatenate((track.samples, gap, clip), axis=0))
        else:
            local_pos = pos - track.offset_samples
            track.samples = np.ascontiguousarray(
                np.concatenate((track.samples[:local_pos], clip, track.samples[local_pos:]), axis=0)
            )
        self.playhead_sample = pos + len(clip)
        return True

    def trim_to_selection(self, track_index: Optional[int] = None) -> bool:
        track = self.tracks[self.active_track_index if track_index is None else track_index]
        rng = self._resolve_track_local_range(track, use_full_if_empty=False)
        if rng is None:
            return False
        lo, hi = rng
        self.push_undo()
        track.samples = np.ascontiguousarray(track.samples[lo:hi].copy(), dtype=np.float32)
        track.offset_samples = track.offset_samples + lo
        self.selection = (track.offset_samples, track.end_samples)
        self.playhead_sample = track.offset_samples
        return True

    def fade_selection(self, fade_in: bool = True, track_index: Optional[int] = None) -> bool:
        track = self.tracks[self.active_track_index if track_index is None else track_index]
        rng = self._resolve_track_local_range(track, use_full_if_empty=True)
        if rng is None:
            return False
        lo, hi = rng
        length = hi - lo
        if length < 2:
            return False
        self.push_undo()
        ramp = np.linspace(0.0, 1.0, length, dtype=np.float32)[:, None]
        if not fade_in:
            ramp = 1.0 - ramp
        track.samples[lo:hi] *= ramp
        return True

    def normalize_track(self, target_db: float = -1.0, track_index: Optional[int] = None) -> bool:
        track = self.tracks[self.active_track_index if track_index is None else track_index]
        rng = self._resolve_track_local_range(track, use_full_if_empty=True)
        if rng is None:
            return False
        lo, hi = rng
        seg = track.samples[lo:hi]
        peak = float(np.max(np.abs(seg))) if len(seg) else 0.0
        if peak < 1e-6:
            return False
        target_amp = float(10.0 ** (target_db / 20.0))
        gain = target_amp / peak
        self.push_undo()
        track.samples[lo:hi] = np.clip(seg * gain, -1.0, 1.0)
        return True

    def reverse_selection(self, track_index: Optional[int] = None) -> bool:
        track = self.tracks[self.active_track_index if track_index is None else track_index]
        rng = self._resolve_track_local_range(track, use_full_if_empty=True)
        if rng is None:
            return False
        lo, hi = rng
        self.push_undo()
        track.samples[lo:hi] = track.samples[lo:hi][::-1].copy()
        return True

    def apply_gain_db(self, db: float, track_index: Optional[int] = None) -> bool:
        track = self.tracks[self.active_track_index if track_index is None else track_index]
        rng = self._resolve_track_local_range(track, use_full_if_empty=True)
        if rng is None:
            return False
        lo, hi = rng
        factor = float(10.0 ** (db / 20.0))
        self.push_undo()
        track.samples[lo:hi] = np.clip(track.samples[lo:hi] * factor, -1.0, 1.0)
        return True

    def nudge_track(self, track_index: int, delta_samples: int, push_undo: bool = True) -> int:
        track = self.tracks[max(0, min(track_index, NUM_TRACKS - 1))]
        new_off = max(0, int(track.offset_samples + delta_samples))
        if new_off != track.offset_samples:
            if push_undo:
                self.push_undo()
            track.offset_samples = new_off
            self.modified = True
        return track.offset_samples

    # ------------------------------------------------------------------
    # Multi-Track Real-Time & Offline Mixdown
    # ------------------------------------------------------------------
    def active_audible_tracks(self, exclude_track: Optional[int] = None) -> List[AudioTrack]:
        any_solo = any(t.solo for t in self.tracks if t.index != exclude_track)
        out = []
        for t in self.tracks:
            if exclude_track is not None and t.index == exclude_track:
                continue
            if t.muted or t.duration_samples == 0:
                continue
            if any_solo and not t.solo:
                continue
            out.append(t)
        return out

    def mixdown(
        self,
        start_sample: int = 0,
        end_sample: Optional[int] = None,
        exclude_track: Optional[int] = None,
    ) -> np.ndarray:
        """Mix active tracks cleanly across [start_sample, end_sample) into stereo float32."""
        start_sample = max(0, int(start_sample))
        if end_sample is None:
            end_sample = self.total_samples()
        end_sample = max(start_sample, int(end_sample))
        length = end_sample - start_sample
        if length <= 0:
            return empty_stereo(0)

        mix = empty_stereo(length)
        for t in self.active_audible_tracks(exclude_track=exclude_track):
            ov_start = max(start_sample, t.offset_samples)
            ov_end = min(end_sample, t.end_samples)
            if ov_end <= ov_start:
                continue
            src_lo = ov_start - t.offset_samples
            src_hi = ov_end - t.offset_samples
            dst_lo = ov_start - start_sample
            dst_hi = ov_end - start_sample

            pan = max(-1.0, min(1.0, float(t.pan)))
            vol = max(0.0, float(t.volume))
            g_l = np.float32(vol * min(1.0, 1.0 - pan))
            g_r = np.float32(vol * min(1.0, 1.0 + pan))
            seg = t.samples[src_lo:src_hi]
            mix[dst_lo:dst_hi, 0] += seg[:, 0] * g_l
            mix[dst_lo:dst_hi, 1] += seg[:, 1] * g_r

        return np.clip(mix, -1.0, 1.0)

    def export_mix(self, out_path: str, file_format: Optional[str] = None):
        """Export mixed 4-track project to WAV cleanly or MP3/FLAC/OGG/M4A via ffmpeg."""
        total = self.total_samples()
        if total <= 0:
            mix = empty_stereo(int(self.sample_rate * 0.25))
        else:
            mix = self.mixdown(0, total)

        ext = (file_format or os.path.splitext(out_path)[1].lstrip(".") or "wav").lower()
        wav_data = wav_bytes_from_float32(mix, self.sample_rate)

        if ext == "wav":
            with open(out_path, "wb") as f:
                f.write(wav_data)
            return

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(f"ffmpeg is required to export .{ext} audio.")

        proc = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-v", "error",
                "-f", "wav",
                "-i", "pipe:0",
                out_path,
            ],
            input=wav_data,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Export failed: {err or 'ffmpeg error'}")

    # ------------------------------------------------------------------
    # Project Save / Load (.vpanel zip bundle using ZIP_STORED)
    # ------------------------------------------------------------------
    def save_project(self, path: str):
        meta = {
            "app": "VoicePanel",
            "version": 1,
            "sample_rate": self.sample_rate,
            "active_track": self.active_track_index,
            "playhead_sample": self.playhead_sample,
            "tracks": [],
        }
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
            for t in self.tracks:
                wav_name = f"track_{t.index}.wav" if t.duration_samples > 0 else None
                if wav_name:
                    zf.writestr(wav_name, wav_bytes_from_float32(t.samples, self.sample_rate))
                meta["tracks"].append({
                    "index": t.index,
                    "name": t.name,
                    "offset_samples": t.offset_samples,
                    "volume": t.volume,
                    "pan": t.pan,
                    "muted": t.muted,
                    "solo": t.solo,
                    "armed": t.armed,
                    "color_hex": t.color_hex,
                    "wav_file": wav_name,
                })
            zf.writestr("project.json", json.dumps(meta, indent=2).encode("utf-8"))
        self.project_path = path
        self.modified = False

    def load_project(self, path: str):
        with zipfile.ZipFile(path, "r") as zf:
            meta = json.loads(zf.read("project.json").decode("utf-8"))
            self.sample_rate = int(meta.get("sample_rate", DEFAULT_SAMPLE_RATE))
            self.active_track_index = int(meta.get("active_track", 0))
            self.playhead_sample = int(meta.get("playhead_sample", 0))
            self.selection = None
            for t_meta in meta.get("tracks", []):
                idx = int(t_meta["index"])
                if not (0 <= idx < NUM_TRACKS):
                    continue
                t = self.tracks[idx]
                t.name = t_meta.get("name", f"Track {idx + 1}")
                t.offset_samples = int(t_meta.get("offset_samples", 0))
                t.volume = float(t_meta.get("volume", 1.0))
                t.pan = float(t_meta.get("pan", 0.0))
                t.muted = bool(t_meta.get("muted", False))
                t.solo = bool(t_meta.get("solo", False))
                t.armed = bool(t_meta.get("armed", idx == 0))
                t.color_hex = t_meta.get("color_hex", TRACK_COLORS[idx % len(TRACK_COLORS)])
                wav_name = t_meta.get("wav_file")
                if wav_name and wav_name in zf.namelist():
                    t.samples, _ = float32_from_wav_bytes(zf.read(wav_name), target_rate=self.sample_rate)
                else:
                    t.samples = empty_stereo(0)
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.project_path = path
        self.modified = False


class AudioDeviceManager:
    """Enumerate available microphone inputs cleanly via PulseAudio/PipeWire pactl."""

    @staticmethod
    def list_input_sources() -> List[Tuple[str, str]]:
        """Return [(device_id, human_label), ...], putting real mics first and monitors last."""
        pactl = shutil.which("pactl")
        if not pactl:
            return [("", "Default System Microphone")]
        try:
            proc = subprocess.run(
                [pactl, "list", "sources"],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=3,
                check=False,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                return [("", "Default System Microphone")]

            mics: List[Tuple[str, str]] = []
            monitors: List[Tuple[str, str]] = []
            cur_name = None
            cur_desc = None

            def flush():
                nonlocal cur_name, cur_desc
                if cur_name:
                    label = cur_desc or cur_name
                    if ".monitor" in cur_name:
                        monitors.append((cur_name, f"[System Monitor] {label}"))
                    else:
                        mics.append((cur_name, label))
                cur_name = None
                cur_desc = None

            for raw_line in proc.stdout.splitlines():
                line = raw_line.strip()
                if line.startswith("Source #"):
                    flush()
                elif line.startswith("Name:"):
                    cur_name = line.split(":", 1)[1].strip()
                elif line.startswith("Description:"):
                    cur_desc = line.split(":", 1)[1].strip()
            flush()

            ordered = mics + monitors
            if ordered:
                return [("", "Default System Microphone")] + ordered
        except Exception:
            pass
        return [("", "Default System Microphone")]


class AudioRecorder:
    """Low-latency microphone capture streaming stereo float32 chunks into memory."""

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, device_name: str = ""):
        self.sample_rate = int(sample_rate)
        self.device_name = device_name or ""
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._chunks: List[np.ndarray] = []
        self._total_frames = 0
        self.live_peak = 0.0
        self.backend_name = "none"
        self.last_error: Optional[str] = None

    def _build_capture_cmd(self) -> Optional[Tuple[str, List[str], int]]:
        parec = shutil.which("parec")
        if parec:
            cmd = [
                parec,
                "--format=float32le",
                f"--rate={self.sample_rate}",
                "--channels=2",
                "--latency-msec=20",
            ]
            if self.device_name:
                cmd.append(f"--device={self.device_name}")
            return "parec", cmd, 4  # 4 bytes per float32 sample

        arecord = shutil.which("arecord")
        if arecord:
            cmd = [
                arecord,
                "-q",
                "-t", "raw",
                "-f", "S16_LE",
                "-c", "2",
                "-r", str(self.sample_rate),
            ]
            return "arecord", cmd, 2

        return None

    def start(self) -> bool:
        if self._running:
            return True
        spec = self._build_capture_cmd()
        if spec is None:
            self.last_error = "No microphone capture backend found (parec / arecord missing)."
            return False

        backend, cmd, samp_bytes = spec
        self.backend_name = backend
        self.last_error = None
        with self._lock:
            self._chunks = []
            self._total_frames = 0
            self.live_peak = 0.0

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except Exception as e:
            self.last_error = str(e)
            self._proc = None
            return False

        self._running = True
        frames_per_read = 1024
        bytes_per_frame = 2 * samp_bytes
        read_size = frames_per_read * bytes_per_frame

        def reader():
            leftover = b""
            stdout = self._proc.stdout
            try:
                while self._running and stdout is not None:
                    data = stdout.read(read_size)
                    if not data:
                        break
                    buf = leftover + data
                    rem = len(buf) % bytes_per_frame
                    if rem:
                        leftover = buf[-rem:]
                        buf = buf[:-rem]
                    else:
                        leftover = b""
                    if not buf:
                        continue
                    if samp_bytes == 4:
                        chunk = np.frombuffer(buf, dtype="<f4").reshape(-1, 2).copy()
                    else:
                        chunk = (np.frombuffer(buf, dtype="<i2").astype(np.float32) / 32768.0).reshape(-1, 2).copy()
                    peak = float(np.max(np.abs(chunk))) if len(chunk) else 0.0
                    with self._lock:
                        self._chunks.append(chunk)
                        self._total_frames += len(chunk)
                        self.live_peak = max(peak, self.live_peak * 0.72)
            except Exception as e:
                self.last_error = str(e)

        self._thread = threading.Thread(target=reader, daemon=True)
        self._thread.start()
        return True

    def is_recording(self) -> bool:
        return self._running

    def get_recorded_frames_count(self) -> int:
        with self._lock:
            return self._total_frames

    def get_live_preview_samples(self) -> np.ndarray:
        with self._lock:
            if not self._chunks:
                return empty_stereo(0)
            return np.concatenate(self._chunks, axis=0)

    def stop(self) -> np.ndarray:
        self._running = False
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        with self._lock:
            if not self._chunks:
                self.live_peak = 0.0
                return empty_stereo(0)
            result = np.ascontiguousarray(np.concatenate(self._chunks, axis=0), dtype=np.float32)
            self._chunks = []
            self._total_frames = 0
            self.live_peak = 0.0
            return result


class AudioPlayer:
    """Non-blocking real-time playback streamer for the mixed 4-track timeline."""

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE):
        self.sample_rate = int(sample_rate)
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.current_sample = 0
        self.live_peak = 0.0
        self.on_finished: Optional[Callable[[], None]] = None

    def _build_play_cmd(self) -> Optional[List[str]]:
        paplay = shutil.which("paplay")
        if paplay:
            return [
                paplay,
                "--raw",
                "--format=s16le",
                f"--rate={self.sample_rate}",
                "--channels=2",
                "--latency-msec=30",
            ]
        aplay = shutil.which("aplay")
        if aplay:
            return [
                aplay,
                "-q",
                "-t", "raw",
                "-f", "S16_LE",
                "-c", "2",
                "-r", str(self.sample_rate),
            ]
        return None

    def start(
        self,
        session: MultiTrackSession,
        start_sample: int = 0,
        loop: bool = False,
        exclude_track: Optional[int] = None,
        on_finished: Optional[Callable[[], None]] = None,
    ) -> bool:
        self.stop()
        cmd = self._build_play_cmd()
        if cmd is None:
            return False

        self.on_finished = on_finished
        self.current_sample = max(0, int(start_sample))
        self.live_peak = 0.0

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except Exception:
            self._proc = None
            return False

        self._running = True
        chunk_frames = 2048

        def writer():
            pos = max(0, int(start_sample))
            proc = self._proc
            try:
                while self._running and proc is not None and proc.stdin is not None:
                    total = session.total_samples()
                    if total <= 0:
                        if exclude_track is not None:
                            # Silent keep-alive clock during solo overdub when no backing track exists
                            silence = np.zeros((chunk_frames, 2), dtype="<i2").tobytes()
                            proc.stdin.write(silence)
                            pos += chunk_frames
                            self.current_sample = pos
                            continue
                        break

                    if pos >= total:
                        if exclude_track is not None:
                            silence = np.zeros((chunk_frames, 2), dtype="<i2").tobytes()
                            proc.stdin.write(silence)
                            pos += chunk_frames
                            self.current_sample = pos
                            continue
                        if loop:
                            pos = 0
                        else:
                            break

                    end = min(total, pos + chunk_frames)
                    if exclude_track is not None and end <= pos:
                        chunk = empty_stereo(chunk_frames)
                    else:
                        chunk = session.mixdown(pos, end, exclude_track=exclude_track)
                        if len(chunk) < chunk_frames and exclude_track is not None:
                            pad = empty_stereo(chunk_frames - len(chunk))
                            chunk = np.concatenate((chunk, pad), axis=0)

                    self.live_peak = float(np.max(np.abs(chunk))) if len(chunk) else 0.0
                    pcm16 = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype("<i2")
                    proc.stdin.write(pcm16.tobytes())
                    proc.stdin.flush()
                    pos += len(chunk)
                    self.current_sample = pos
            except Exception:
                pass
            finally:
                was_running = self._running
                self._running = False
                if proc is not None and proc.stdin is not None:
                    try:
                        proc.stdin.close()
                    except Exception:
                        pass
                if proc is not None:
                    try:
                        proc.wait(timeout=1.5)
                    except Exception:
                        pass
                self.live_peak = 0.0
                if was_running and self.on_finished:
                    try:
                        self.on_finished()
                    except Exception:
                        pass

        self._thread = threading.Thread(target=writer, daemon=True)
        self._thread.start()
        return True

    def is_playing(self) -> bool:
        return self._running

    def stop(self):
        self._running = False
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=0.8)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=0.8)
            self._thread = None
        self.live_peak = 0.0
