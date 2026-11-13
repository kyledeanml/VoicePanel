#!/usr/bin/env python3
"""
Unit and Integration Tests for VoicePanel
Tests 4-track mixing, range editing, Undo/Redo, WAV & .vpanel I/O,
microphone recorder streaming, SVG watermark assets, and GTK 3 UI.
"""

import io
import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import numpy as np

sys.dont_write_bytecode = True

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk

from audio_engine import (
    AudioDeviceManager,
    AudioRecorder,
    AudioTrack,
    DEFAULT_SAMPLE_RATE,
    MultiTrackSession,
    NUM_TRACKS,
    float32_from_wav_bytes,
    wav_bytes_from_float32,
)
from voicepanel import VoicePanelWindow


def sine_stereo(freq: float, seconds: float, amp: float = 0.5, sr: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(round(seconds * sr)), dtype=np.float32) / float(sr)
    wave_1d = (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    return np.ascontiguousarray(np.column_stack((wave_1d, wave_1d)), dtype=np.float32)


class TestAudioEngine(unittest.TestCase):
    """Test 4-track data model, mixing, editing, Undo/Redo, and I/O."""

    def test_four_tracks_init_and_exclusive_arm(self):
        sess = MultiTrackSession()
        self.assertEqual(len(sess.tracks), NUM_TRACKS)
        self.assertTrue(sess.tracks[0].armed)
        self.assertFalse(any(t.armed for t in sess.tracks[1:]))

        sess.arm_track(2)
        self.assertTrue(sess.tracks[2].armed)
        self.assertFalse(sess.tracks[0].armed)
        self.assertEqual(sess.active_track_index, 2)

    def test_wav_encode_decode_and_resample(self):
        sr_24k = 24000
        tone = sine_stereo(440.0, 0.2, amp=0.6, sr=sr_24k)
        raw = wav_bytes_from_float32(tone, sample_rate=sr_24k)
        decoded, out_sr = float32_from_wav_bytes(raw, target_rate=48000)
        self.assertEqual(out_sr, 48000)
        self.assertEqual(decoded.shape[1], 2)
        self.assertAlmostEqual(len(decoded), 9600, delta=4)
        self.assertGreater(float(np.max(np.abs(decoded))), 0.5)

    def test_mixdown_volume_pan_mute_solo(self):
        sess = MultiTrackSession(sample_rate=48000)
        sess.tracks[0].samples = np.full((100, 2), 0.4, dtype=np.float32)
        sess.tracks[1].samples = np.full((100, 2), 0.2, dtype=np.float32)
        sess.tracks[1].offset_samples = 50
        sess.tracks[1].pan = -1.0  # Full Left

        mix = sess.mixdown()
        self.assertEqual(mix.shape, (150, 2))
        # First 50 frames: Track 0 only
        self.assertTrue(np.allclose(mix[:50], 0.4, atol=1e-5))
        # Overlap [50:100]: Track 0 + Track 1 left only
        self.assertTrue(np.allclose(mix[50:100, 0], 0.6, atol=1e-5))
        self.assertTrue(np.allclose(mix[50:100, 1], 0.4, atol=1e-5))

        # Mute track 0
        sess.tracks[0].muted = True
        mix_muted = sess.mixdown()
        self.assertTrue(np.allclose(mix_muted[:50], 0.0, atol=1e-5))

        # Solo track 0 (while unmuting track 0) isolates track 0
        sess.tracks[0].muted = False
        sess.tracks[0].solo = True
        mix_solo = sess.mixdown()
        self.assertTrue(np.allclose(mix_solo[50:100, 0], 0.4, atol=1e-5))

    def test_range_editing_and_undo_redo(self):
        sess = MultiTrackSession(sample_rate=48000)
        t0 = sess.tracks[0]
        t0.samples = sine_stereo(440.0, 1.0, amp=0.25)

        # Normalize -> peak scales close to -1 dBFS (~0.891)
        self.assertTrue(sess.normalize_track(target_db=-1.0, track_index=0))
        peak_norm = float(np.max(np.abs(t0.samples)))
        self.assertAlmostEqual(peak_norm, 10 ** (-1.0 / 20.0), places=2)

        # Cut first 0.25s
        sess.selection = (0, 12000)
        self.assertTrue(sess.cut_selection(track_index=0))
        self.assertEqual(t0.duration_samples, 36000)

        # Paste back at sample 0
        sess.playhead_sample = 0
        self.assertTrue(sess.paste_at_playhead(track_index=0))
        self.assertEqual(t0.duration_samples, 48000)

        # Fade in & Fade out
        sess.selection = (0, 4800)
        self.assertTrue(sess.fade_selection(fade_in=True, track_index=0))
        self.assertAlmostEqual(float(np.abs(t0.samples[0, 0])), 0.0, places=5)

        # Silence selection & Undo restore
        sess.selection = (1000, 2000)
        self.assertTrue(sess.silence_selection(track_index=0))
        self.assertTrue(np.allclose(t0.samples[1000:2000], 0.0))
        self.assertTrue(sess.undo())
        self.assertGreater(float(np.max(np.abs(t0.samples[1000:2000]))), 0.05)
        self.assertTrue(sess.redo())
        self.assertTrue(np.allclose(t0.samples[1000:2000], 0.0))

    def test_commit_recording_punch_in_and_reverse(self):
        sess = MultiTrackSession(sample_rate=48000)
        rec1 = np.full((1000, 2), 0.3, dtype=np.float32)
        end_pos = sess.commit_recording(rec1, start_sample=500, track_index=1)
        self.assertEqual(end_pos, 1500)
        self.assertEqual(sess.tracks[1].offset_samples, 500)
        self.assertEqual(sess.tracks[1].duration_samples, 1000)

        # Punch in later on same track preserves earlier audio
        rec2 = np.full((500, 2), -0.5, dtype=np.float32)
        sess.commit_recording(rec2, start_sample=2000, track_index=1)
        self.assertEqual(sess.tracks[1].duration_samples, 2000)  # 500..2500
        self.assertAlmostEqual(float(sess.tracks[1].samples[0, 0]), 0.3, places=5)
        self.assertAlmostEqual(float(sess.tracks[1].samples[-1, 0]), -0.5, places=5)

        sess.selection = None
        sess.reverse_selection(track_index=1)
        self.assertAlmostEqual(float(sess.tracks[1].samples[0, 0]), -0.5, places=5)
        self.assertAlmostEqual(float(sess.tracks[1].samples[-1, 0]), 0.3, places=5)

    def test_project_save_load_and_export_wav(self):
        tmp_dir = tempfile.mkdtemp(prefix="voicepanel_proj_")
        vpanel_path = os.path.join(tmp_dir, "demo.vpanel")
        wav_path = os.path.join(tmp_dir, "mix.wav")
        try:
            sess = MultiTrackSession(sample_rate=48000)
            sess.tracks[0].name = "Vocals"
            sess.tracks[0].samples = sine_stereo(523.25, 0.25, amp=0.5)
            sess.tracks[3].name = "Bass"
            sess.tracks[3].samples = sine_stereo(110.0, 0.25, amp=0.3)
            sess.tracks[3].offset_samples = 2400
            sess.save_project(vpanel_path)
            sess.export_mix(wav_path)

            self.assertTrue(os.path.isfile(vpanel_path))
            self.assertTrue(os.path.isfile(wav_path))

            loaded = MultiTrackSession()
            loaded.load_project(vpanel_path)
            self.assertEqual(loaded.tracks[0].name, "Vocals")
            self.assertEqual(loaded.tracks[3].name, "Bass")
            self.assertEqual(loaded.tracks[3].offset_samples, 2400)
            self.assertGreater(loaded.tracks[0].duration_samples, 10000)

            imp, sr = loaded.import_audio_file(wav_path)
            self.assertEqual(sr, 48000)
            self.assertEqual(len(imp), loaded.total_samples())
        finally:
            for p in (vpanel_path, wav_path):
                if os.path.exists(p):
                    os.remove(p)
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass

    def test_peak_envelope_bounds(self):
        track = AudioTrack(index=0, samples=sine_stereo(220.0, 0.5, amp=0.8), offset_samples=1000)
        mins, maxs = track.peak_envelope(0, 48000, 120)
        self.assertEqual(len(mins), 120)
        self.assertEqual(len(maxs), 120)
        # First bin starts before offset_samples=1000? Bin width is 400 samples, so bin 0..1 are 0
        self.assertEqual(float(maxs[0]), 0.0)
        self.assertGreater(float(np.max(maxs)), 0.7)
        self.assertLess(float(np.min(mins)), -0.7)

    def test_device_list_parsing_and_mocked_recorder(self):
        fake_pactl = (
            "Source #10\n\tName: alsa_output.pci.monitor\n\tDescription: Monitor of Speakers\n"
            "Source #11\n\tName: alsa_input.usb-mic\n\tDescription: USB Condenser Mic\n"
        )
        with patch("shutil.which", return_value="/usr/bin/pactl"), patch(
            "subprocess.run",
            return_value=MagicMock(returncode=0, stdout=fake_pactl),
        ):
            sources = AudioDeviceManager.list_input_sources()
        self.assertEqual(sources[0][0], "")
        # Real mic precedes .monitor source
        self.assertEqual(sources[1][0], "alsa_input.usb-mic")
        self.assertIn("[System Monitor]", sources[2][1])

        # Test AudioRecorder chunk assembly with mocked parec subprocess
        fake_pcm = np.full((2048, 2), 0.25, dtype="<f4").tobytes()
        mock_proc = MagicMock()
        mock_proc.stdout = io.BytesIO(fake_pcm)
        rec = AudioRecorder(sample_rate=48000)
        with patch("shutil.which", side_effect=lambda name: "/usr/bin/parec" if name == "parec" else None), patch(
            "subprocess.Popen", return_value=mock_proc
        ):
            self.assertTrue(rec.start())
            if rec._thread:
                rec._thread.join(timeout=1.0)
            captured = rec.stop()
        self.assertEqual(captured.shape, (2048, 2))
        self.assertTrue(np.allclose(captured, 0.25, atol=1e-5))


class TestVoicePanelUI(unittest.TestCase):
    """Test SVG assets, watermark loading, and VoicePanel 4-track GTK UI."""

    def test_svg_and_watermark_validity(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        svg_path = os.path.join(base_dir, "voicepanel.svg")
        self.assertTrue(os.path.isfile(svg_path))
        tree = ET.parse(svg_path)
        self.assertTrue(tree.getroot().tag.endswith("svg"))
        self.assertEqual(tree.getroot().attrib.get("viewBox"), "0 0 512 512")
        self.assertIsNotNone(VoicePanelWindow.get_watermark_pixbuf(128))

    def test_window_tracks_dark_mode_and_editing(self):
        has_display, _ = Gtk.init_check()
        if not has_display:
            self.skipTest("Display unavailable")

        tmp_dir = tempfile.mkdtemp(prefix="voicepanel_ui_")
        wav_path = os.path.join(tmp_dir, "vocal.wav")
        try:
            with open(wav_path, "wb") as f:
                f.write(wav_bytes_from_float32(sine_stereo(330.0, 0.25, amp=0.5), 48000))

            win = VoicePanelWindow(None, initial_files=[wav_path])
            try:
                self.assertEqual(len(win.track_rows), 4)
                self.assertGreater(win.session.tracks[0].duration_samples, 10000)

                # Arm Track 2 -> exclusive arm switches off Track 1
                win.track_rows[1].btn_arm.set_active(True)
                self.assertTrue(win.session.tracks[1].armed)
                self.assertFalse(win.session.tracks[0].armed)

                # Toggle Dark / Light Mode via headerbar button
                self.assertTrue(win.dark_mode)
                win.btn_dark_mode.set_active(False)
                self.assertFalse(win.dark_mode)
                self.assertFalse(win.chk_dark_mode.get_active())
                win.toggle_dark_mode()
                self.assertTrue(win.dark_mode)

                # Edit + Undo via UI
                win.select_active_track(0)
                win.session.selection = (0, 2000)
                win.run_edit("cut")
                self.assertEqual(win.session.tracks[0].duration_samples, 10000)
                win.on_undo()
                self.assertEqual(win.session.tracks[0].duration_samples, 12000)

                # Shift + > moves marker to end of active track; Shift + < moves to beginning
                from gi.repository import Gdk
                win.set_focus(None)
                win.set_playhead(3000)
                ev_gt = Gdk.Event.new(Gdk.EventType.KEY_PRESS)
                ev_gt.key.keyval = Gdk.KEY_greater
                ev_gt.key.state = Gdk.ModifierType.SHIFT_MASK
                self.assertTrue(win._on_window_key_press(win, ev_gt))
                self.assertEqual(win.session.playhead_sample, 12000)

                ev_lt = Gdk.Event.new(Gdk.EventType.KEY_PRESS)
                ev_lt.key.keyval = Gdk.KEY_less
                ev_lt.key.state = Gdk.ModifierType.SHIFT_MASK
                self.assertTrue(win._on_window_key_press(win, ev_lt))
                self.assertEqual(win.session.playhead_sample, 0)

                # With clip offset > 0, Shift + < lands at clip start first, then 0 on second press
                win.session.tracks[0].offset_samples = 2400
                win.set_playhead(8000)
                self.assertTrue(win._on_window_key_press(win, ev_lt))
                self.assertEqual(win.session.playhead_sample, 2400)
                self.assertTrue(win._on_window_key_press(win, ev_lt))
                self.assertEqual(win.session.playhead_sample, 0)
            finally:
                win.stop_all()
                win.destroy()
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
