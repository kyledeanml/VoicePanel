#!/usr/bin/env python3
"""
VoicePanel - 4-Track Audio Editor & Microphone Studio
Simple, fast 4-track recorder and waveform editor with live mic capture,
overdub playback, range editing, Undo/Redo, Drag-and-Drop import, and
PythonPanel-style dark-neon UI + translucent yoga-pad logo watermark.
"""

import os
import sys
import urllib.parse
from typing import List, Optional, Tuple

import cairo
import gi
import numpy as np

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("GdkPixbuf", "2.0")

from gi.repository import Gdk, GdkPixbuf, GLib, Gio, Gtk

from audio_engine import (
    AudioDeviceManager,
    AudioPlayer,
    AudioRecorder,
    AudioTrack,
    DEFAULT_SAMPLE_RATE,
    MultiTrackSession,
    NUM_TRACKS,
    SUPPORTED_AUDIO_EXTS,
    TRACK_COLORS,
)


def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    mins = int(seconds // 60)
    secs = seconds - (mins * 60)
    return f"{mins:02d}:{secs:05.2f}"


class PeakMeter(Gtk.DrawingArea):
    """Compact neon LED peak meter widget."""

    def __init__(self):
        super().__init__()
        self.level = 0.0
        self.dark_mode = True
        self.set_size_request(92, 12)
        self.connect("draw", self._on_draw)

    def set_level(self, level: float):
        level = max(0.0, min(1.0, float(level)))
        if abs(level - self.level) > 0.01 or (level == 0.0 and self.level != 0.0):
            self.level = level
            self.queue_draw()

    def _on_draw(self, widget, cr: cairo.Context):
        alloc = self.get_allocation()
        w = max(1, alloc.width)
        h = max(1, alloc.height)

        if self.dark_mode:
            cr.set_source_rgb(0.04, 0.06, 0.12)
        else:
            cr.set_source_rgb(0.88, 0.91, 0.96)
        cr.rectangle(0, 0, w, h)
        cr.fill()

        if self.level > 0.005:
            fill_w = w * min(1.0, self.level)
            grad = cairo.LinearGradient(0, 0, w, 0)
            grad.add_color_stop_rgb(0.0, 0.0, 0.94, 1.0)
            grad.add_color_stop_rgb(0.65, 0.06, 0.73, 0.51)
            grad.add_color_stop_rgb(0.85, 0.96, 0.62, 0.04)
            grad.add_color_stop_rgb(1.0, 0.96, 0.25, 0.37)
            cr.set_source(grad)
            cr.rectangle(1, 1, max(0, fill_w - 2), max(1, h - 2))
            cr.fill()

        cr.set_source_rgba(0.22, 0.74, 0.97, 0.35)
        cr.set_line_width(1.0)
        cr.rectangle(0.5, 0.5, w - 1, h - 1)
        cr.stroke()
        return False


class TimelineRuler(Gtk.DrawingArea):
    """Interactive time ruler synced with the 4-track waveform viewport."""

    def __init__(self, app_window):
        super().__init__()
        self.app_window = app_window
        self.set_size_request(-1, 26)
        self.add_events(
            Gdk.EventMask.BUTTON_PRESS_MASK
            | Gdk.EventMask.BUTTON1_MOTION_MASK
        )
        self.connect("draw", self._on_draw)
        self.connect("button-press-event", self._on_mouse)
        self.connect("motion-notify-event", self._on_mouse)

    def _on_mouse(self, widget, event):
        if event.type == Gdk.EventType.BUTTON_PRESS and event.button != 1:
            return False
        alloc = self.get_allocation()
        sample = self.app_window.x_to_sample(event.x, alloc.width)
        self.app_window.set_playhead(sample)
        return True

    def _on_draw(self, widget, cr: cairo.Context):
        alloc = self.get_allocation()
        w, h = alloc.width, alloc.height
        dark = self.app_window.dark_mode
        if dark:
            cr.set_source_rgb(0.05, 0.08, 0.14)
        else:
            cr.set_source_rgb(0.92, 0.95, 0.98)
        cr.rectangle(0, 0, w, h)
        cr.fill()

        view_start, view_end = self.app_window.get_view_range()
        sr = self.app_window.session.sample_rate
        span_sec = max(0.05, (view_end - view_start) / float(sr))

        # Choose clean tick step in seconds
        for cand in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0):
            if span_sec / cand <= 12:
                step_sec = cand
                break
        else:
            step_sec = 300.0

        start_sec = view_start / float(sr)
        end_sec = view_end / float(sr)
        first_tick = int(start_sec / step_sec) * step_sec

        cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        cr.set_font_size(10.5)

        t = first_tick
        while t <= end_sec + step_sec * 0.5:
            s_idx = int(round(t * sr))
            x = self.app_window.sample_to_x(s_idx, w)
            if 0 <= x <= w:
                cr.set_source_rgba(0.22, 0.74, 0.97, 0.55 if dark else 0.7)
                cr.move_to(x, h - 9)
                cr.line_to(x, h)
                cr.stroke()

                lbl = f"{int(t // 60)}:{int(t % 60):02d}" if step_sec >= 1.0 else f"{t:.1f}s"
                cr.set_source_rgba(0.88, 0.95, 1.0, 0.85) if dark else cr.set_source_rgba(0.06, 0.09, 0.16, 0.85)
                cr.move_to(x + 4, 13)
                cr.show_text(lbl)
            t += step_sec

        # Selection band on ruler
        sel = self.app_window.session.selection
        if sel is not None:
            s0, s1 = sorted(sel)
            x0 = max(0.0, self.app_window.sample_to_x(s0, w))
            x1 = min(float(w), self.app_window.sample_to_x(s1, w))
            if x1 > x0:
                cr.set_source_rgba(0.0, 0.94, 1.0, 0.25)
                cr.rectangle(x0, 0, x1 - x0, h)
                cr.fill()

        # Playhead triangle marker
        px = self.app_window.sample_to_x(self.app_window.session.playhead_sample, w)
        if -8 <= px <= w + 8:
            cr.set_source_rgb(0.0, 0.94, 1.0)
            cr.move_to(px - 6, 0)
            cr.line_to(px + 6, 0)
            cr.line_to(px, 11)
            cr.close_path()
            cr.fill()
            cr.set_line_width(1.5)
            cr.move_to(px, 10)
            cr.line_to(px, h)
            cr.stroke()

        cr.set_source_rgba(0.0, 0.94, 1.0, 0.3)
        cr.set_line_width(1.0)
        cr.move_to(0, h - 0.5)
        cr.line_to(w, h - 0.5)
        cr.stroke()
        return False


class TrackCanvas(Gtk.DrawingArea):
    """Cairo waveform & clip canvas for a single VoicePanel track."""

    def __init__(self, track_index: int, app_window):
        super().__init__()
        self.track_index = track_index
        self.app_window = app_window
        self._drag_mode: Optional[str] = None
        self._anchor_sample = 0
        self._last_nudge_x = 0.0
        self._nudge_pushed_undo = False

        self.set_size_request(420, 118)
        self.add_events(
            Gdk.EventMask.BUTTON_PRESS_MASK
            | Gdk.EventMask.BUTTON_RELEASE_MASK
            | Gdk.EventMask.POINTER_MOTION_MASK
            | Gdk.EventMask.BUTTON1_MOTION_MASK
            | Gdk.EventMask.SCROLL_MASK
            | Gdk.EventMask.SMOOTH_SCROLL_MASK
        )
        self.connect("draw", self._on_draw)
        self.connect("button-press-event", self._on_button_press)
        self.connect("motion-notify-event", self._on_motion_notify)
        self.connect("button-release-event", self._on_button_release)
        self.connect("scroll-event", self._on_scroll)

        # Drag-and-drop audio files directly onto track waveform
        self.drag_dest_set(
            Gtk.DestDefaults.ALL,
            [Gtk.TargetEntry.new("text/uri-list", 0, 0)],
            Gdk.DragAction.COPY,
        )
        self.connect("drag-data-received", self._on_drag_data_received)

    def _on_drag_data_received(self, widget, drag_context, x, y, data, info, time_val):
        uris = data.get_uris() or []
        loaded = False
        for uri in uris:
            if uri.startswith("file://"):
                path = urllib.parse.unquote(urllib.parse.urlparse(uri).path)
                if os.path.isfile(path):
                    self.app_window.import_file_to_track(path, self.track_index)
                    loaded = True
                    break
        drag_context.finish(loaded, False, time_val)

    def _on_scroll(self, widget, event):
        state = getattr(event, "state", 0)
        if state & Gdk.ModifierType.CONTROL_MASK:
            dy = 0.0
            if event.direction == Gdk.ScrollDirection.UP:
                dy = -1.0
            elif event.direction == Gdk.ScrollDirection.DOWN:
                dy = 1.0
            elif event.direction == Gdk.ScrollDirection.SMOOTH:
                _, _, dy = event.get_scroll_deltas()
            if dy < 0:
                self.app_window.zoom_timeline(factor=0.75)
                return True
            elif dy > 0:
                self.app_window.zoom_timeline(factor=1.35)
                return True
        return False

    def _on_button_press(self, widget, event):
        if event.button != 1:
            return False
        alloc = self.get_allocation()
        self.app_window.select_active_track(self.track_index)
        sample = self.app_window.x_to_sample(event.x, alloc.width)

        shift = bool(getattr(event, "state", 0) & Gdk.ModifierType.SHIFT_MASK)
        track = self.app_window.session.tracks[self.track_index]
        if shift and track.duration_samples > 0:
            self._drag_mode = "nudge"
            self._last_nudge_x = event.x
            self._nudge_pushed_undo = False
        else:
            self._drag_mode = "select"
            self._anchor_sample = sample
            self.app_window.session.selection = None
            self.app_window.set_playhead(sample)
        return True

    def _on_motion_notify(self, widget, event):
        if not self._drag_mode:
            return False
        alloc = self.get_allocation()
        w = max(1, alloc.width)
        if self._drag_mode == "select":
            cur_s = self.app_window.x_to_sample(event.x, w)
            if abs(cur_s - self._anchor_sample) > int(self.app_window.session.sample_rate * 0.02):
                self.app_window.session.selection = (
                    min(self._anchor_sample, cur_s),
                    max(self._anchor_sample, cur_s),
                )
            self.app_window.queue_all_redraws()
            self.app_window.update_time_readout()
            return True
        elif self._drag_mode == "nudge":
            dx = event.x - self._last_nudge_x
            if abs(dx) >= 1.5:
                view_start, view_end = self.app_window.get_view_range()
                samples_per_px = max(1.0, (view_end - view_start) / float(w))
                delta = int(round(dx * samples_per_px))
                if delta != 0:
                    push = not self._nudge_pushed_undo
                    self.app_window.session.nudge_track(self.track_index, delta, push_undo=push)
                    self._nudge_pushed_undo = True
                    self._last_nudge_x = event.x
                    self.app_window.update_undo_redo_ui()
                    self.app_window.queue_all_redraws()
            return True
        return False

    def _on_button_release(self, widget, event):
        if self._drag_mode == "nudge" and self._nudge_pushed_undo:
            self.app_window.update_time_readout()
        self._drag_mode = None
        self._nudge_pushed_undo = False
        return False

    def _on_draw(self, widget, cr: cairo.Context):
        alloc = self.get_allocation()
        w = max(1, alloc.width)
        h = max(1, alloc.height)
        mid_y = h * 0.5
        session = self.app_window.session
        track: AudioTrack = session.tracks[self.track_index]
        is_active = (session.active_track_index == self.track_index)
        dark = self.app_window.dark_mode
        r, g, b = hex_to_rgb(track.color_hex)

        # 1. Background
        if dark:
            cr.set_source_rgb(0.055, 0.082, 0.145) if is_active else cr.set_source_rgb(0.039, 0.055, 0.102)
        else:
            cr.set_source_rgb(0.95, 0.98, 1.0) if is_active else cr.set_source_rgb(1.0, 1.0, 1.0)
        cr.rectangle(0, 0, w, h)
        cr.fill()

        # Subtle center line
        cr.set_source_rgba(r, g, b, 0.18 if dark else 0.25)
        cr.set_line_width(1.0)
        cr.move_to(0, mid_y)
        cr.line_to(w, mid_y)
        cr.stroke()

        # 2. PythonPanel-style VoicePanel Yoga-Pad Watermark
        has_live = (
            self.app_window.recorder.is_recording()
            and track.armed
        )
        wm_alpha = 0.22 if (track.duration_samples == 0 and not has_live) else 0.07
        wm_pix = self.app_window.get_watermark_pixbuf(max(64, int(h * 0.86)))
        if wm_pix is not None:
            pw, ph = wm_pix.get_width(), wm_pix.get_height()
            wx = w - pw - 20 if track.duration_samples > 0 else (w - pw) * 0.5
            wy = (h - ph) * 0.5
            cr.save()
            Gdk.cairo_set_source_pixbuf(cr, wm_pix, wx, wy)
            cr.paint_with_alpha(wm_alpha)
            cr.restore()

        view_start, view_end = self.app_window.get_view_range()

        # 3. Render track audio clip envelope
        if track.duration_samples > 0:
            x0 = self.app_window.sample_to_x(track.offset_samples, w)
            x1 = self.app_window.sample_to_x(track.end_samples, w)
            cx0 = max(0.0, x0)
            cx1 = min(float(w), x1)
            if cx1 > cx0:
                cr.set_source_rgba(r, g, b, 0.09 if dark else 0.10)
                cr.rectangle(cx0, 4, cx1 - cx0, h - 8)
                cr.fill()

                cr.set_source_rgba(r, g, b, 0.45)
                cr.set_line_width(1.0)
                cr.rectangle(cx0, 4, cx1 - cx0, h - 8)
                cr.stroke()

            mins, maxs = track.peak_envelope(view_start, view_end, w)
            amp_scale = (h - 14) * 0.5
            alpha_fill = 0.42 if not track.muted else 0.16
            alpha_edge = 0.95 if not track.muted else 0.35

            # Filled symmetric waveform silhouette
            cr.new_path()
            cr.move_to(0, mid_y - maxs[0] * amp_scale)
            for xi in range(1, w):
                cr.line_to(xi, mid_y - maxs[xi] * amp_scale)
            for xi in range(w - 1, -1, -1):
                cr.line_to(xi, mid_y - mins[xi] * amp_scale)
            cr.close_path()
            cr.set_source_rgba(r, g, b, alpha_fill)
            cr.fill_preserve()
            cr.set_source_rgba(r, g, b, alpha_edge)
            cr.set_line_width(1.1)
            cr.stroke()
        elif not has_live:
            cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            cr.set_font_size(11.5)
            hint = f"{track.name} — Press REC (Ctrl+R) to record mic or drop an audio file here"
            ext = cr.text_extents(hint)
            tx = max(12.0, (w - ext.width) * 0.5)
            ty = mid_y + 4
            if dark:
                cr.set_source_rgba(0.58, 0.64, 0.72, 0.72)
            else:
                cr.set_source_rgba(0.28, 0.33, 0.41, 0.78)
            cr.move_to(tx, ty)
            cr.show_text(hint)

        # 4. Live recording preview overlay if recording into this armed track
        if has_live:
            live_samples = self.app_window.recorder.get_live_preview_samples()
            if len(live_samples) > 0:
                rec_start = self.app_window.record_start_sample
                rec_track = AudioTrack(index=self.track_index, samples=live_samples, offset_samples=rec_start)
                mins, maxs = rec_track.peak_envelope(view_start, view_end, w)
                amp_scale = (h - 14) * 0.5
                cr.new_path()
                for xi in range(w):
                    y_top = mid_y - maxs[xi] * amp_scale
                    y_bot = mid_y - mins[xi] * amp_scale
                    if abs(y_bot - y_top) > 0.5:
                        cr.move_to(xi + 0.5, y_top)
                        cr.line_to(xi + 0.5, y_bot)
                cr.set_source_rgba(0.96, 0.25, 0.37, 0.90)
                cr.set_line_width(1.3)
                cr.stroke()

        # 5. Time selection highlight
        if session.selection is not None:
            s0, s1 = sorted(session.selection)
            sx0 = max(0.0, self.app_window.sample_to_x(s0, w))
            sx1 = min(float(w), self.app_window.sample_to_x(s1, w))
            if sx1 > sx0:
                cr.set_source_rgba(0.0, 0.94, 1.0, 0.16)
                cr.rectangle(sx0, 0, sx1 - sx0, h)
                cr.fill()
                cr.set_source_rgba(0.0, 0.94, 1.0, 0.7)
                cr.set_line_width(1.0)
                cr.move_to(sx0, 0)
                cr.line_to(sx0, h)
                cr.move_to(sx1, 0)
                cr.line_to(sx1, h)
                cr.stroke()

        # 6. Shared playhead line
        px = self.app_window.sample_to_x(session.playhead_sample, w)
        if 0 <= px <= w:
            cr.set_source_rgba(0.96, 0.25, 0.37, 0.95) if has_live else cr.set_source_rgba(0.0, 0.94, 1.0, 0.92)
            cr.set_line_width(1.6)
            cr.move_to(px, 0)
            cr.line_to(px, h)
            cr.stroke()

        # Outer border highlighting active track
        if is_active:
            cr.set_source_rgba(r, g, b, 0.72)
            cr.set_line_width(1.5)
        else:
            cr.set_source_rgba(0.22, 0.74, 0.97, 0.22)
            cr.set_line_width(1.0)
        cr.rectangle(0.5, 0.5, w - 1, h - 1)
        cr.stroke()
        return False


class TrackRow(Gtk.Box):
    """Complete control strip + waveform lane for one of the 4 tracks."""

    def __init__(self, track_index: int, app_window):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        self.track_index = track_index
        self.app_window = app_window
        self._syncing = False
        track = app_window.session.tracks[track_index]

        # Left Controls Strip
        ctrl_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        ctrl_box.set_size_request(256, 118)
        ctrl_box.set_margin_start(6)
        ctrl_box.set_margin_end(6)
        ctrl_box.set_margin_top(4)
        ctrl_box.set_margin_bottom(4)
        Gtk.StyleContext.add_class(ctrl_box.get_style_context(), "track-controls")

        # Row 1: Track badge, Editable Name, Import, Clear
        top_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        lbl_idx = Gtk.Label()
        lbl_idx.set_markup(f"<span foreground='{track.color_hex}'><b>T{track_index + 1}</b></span>")
        lbl_idx.set_width_chars(3)
        top_row.pack_start(lbl_idx, False, False, 0)

        self.entry_name = Gtk.Entry()
        self.entry_name.set_text(track.name)
        self.entry_name.set_width_chars(12)
        self.entry_name.connect("changed", self._on_name_changed)
        top_row.pack_start(self.entry_name, True, True, 0)

        btn_import = Gtk.Button.new_from_icon_name("document-open-symbolic", Gtk.IconSize.MENU)
        btn_import.set_tooltip_text(f"Import audio file into Track {track_index + 1}")
        btn_import.connect("clicked", lambda b: self.app_window.prompt_import_to_track(self.track_index))
        top_row.pack_start(btn_import, False, False, 0)

        btn_clear = Gtk.Button.new_from_icon_name("edit-clear-symbolic", Gtk.IconSize.MENU)
        btn_clear.set_tooltip_text(f"Clear Track {track_index + 1}")
        btn_clear.connect("clicked", lambda b: self.app_window.clear_track(self.track_index))
        top_row.pack_start(btn_clear, False, False, 0)

        ctrl_box.pack_start(top_row, False, False, 0)

        # Row 2: ARM, M, S + PeakMeter
        mid_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self.btn_arm = Gtk.ToggleButton(label="● ARM")
        self.btn_arm.set_active(track.armed)
        self.btn_arm.set_tooltip_text("Arm Track for Microphone Recording")
        Gtk.StyleContext.add_class(self.btn_arm.get_style_context(), "arm-btn")
        self.btn_arm.connect("toggled", self._on_arm_toggled)
        mid_row.pack_start(self.btn_arm, False, False, 0)

        self.btn_mute = Gtk.ToggleButton(label="M")
        self.btn_mute.set_active(track.muted)
        self.btn_mute.set_tooltip_text("Mute Track")
        Gtk.StyleContext.add_class(self.btn_mute.get_style_context(), "mute-btn")
        self.btn_mute.connect("toggled", self._on_mute_toggled)
        mid_row.pack_start(self.btn_mute, False, False, 0)

        self.btn_solo = Gtk.ToggleButton(label="S")
        self.btn_solo.set_active(track.solo)
        self.btn_solo.set_tooltip_text("Solo Track")
        Gtk.StyleContext.add_class(self.btn_solo.get_style_context(), "solo-btn")
        self.btn_solo.connect("toggled", self._on_solo_toggled)
        mid_row.pack_start(self.btn_solo, False, False, 0)

        self.meter = PeakMeter()
        mid_row.pack_end(self.meter, True, True, 2)

        ctrl_box.pack_start(mid_row, False, False, 0)

        # Row 3: Volume slider + Pan slider
        mix_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        lbl_vol = Gtk.Label(label="Vol")
        lbl_vol.set_width_chars(3)
        mix_row.pack_start(lbl_vol, False, False, 0)

        self.scale_vol = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.0, 1.5, 0.05)
        self.scale_vol.set_draw_value(False)
        self.scale_vol.set_value(track.volume)
        self.scale_vol.set_tooltip_text("Track Volume (0% - 150%)")
        self.scale_vol.connect("value-changed", self._on_vol_changed)
        mix_row.pack_start(self.scale_vol, True, True, 0)

        lbl_pan = Gtk.Label(label="Pan")
        lbl_pan.set_width_chars(3)
        mix_row.pack_start(lbl_pan, False, False, 0)

        self.scale_pan = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, -1.0, 1.0, 0.05)
        self.scale_pan.set_draw_value(False)
        self.scale_pan.set_value(track.pan)
        self.scale_pan.set_tooltip_text("Stereo Pan (Left / Center / Right)")
        self.scale_pan.connect("value-changed", self._on_pan_changed)
        mix_row.pack_start(self.scale_pan, True, True, 0)

        ctrl_box.pack_start(mix_row, False, False, 0)

        self.pack_start(ctrl_box, False, False, 0)

        # Right Waveform Canvas
        self.canvas = TrackCanvas(track_index, app_window)
        self.pack_start(self.canvas, True, True, 0)

    def sync_from_model(self):
        track = self.app_window.session.tracks[self.track_index]
        self._syncing = True
        try:
            if self.entry_name.get_text() != track.name:
                self.entry_name.set_text(track.name)
            self.btn_arm.set_active(track.armed)
            self.btn_mute.set_active(track.muted)
            self.btn_solo.set_active(track.solo)
            self.scale_vol.set_value(track.volume)
            self.scale_pan.set_value(track.pan)
            self.meter.dark_mode = self.app_window.dark_mode
        finally:
            self._syncing = False

    def _on_name_changed(self, entry):
        if self._syncing:
            return
        self.app_window.session.tracks[self.track_index].name = entry.get_text()

    def _on_arm_toggled(self, btn):
        if self._syncing:
            return
        if btn.get_active():
            self.app_window.arm_track(self.track_index)
        else:
            self.app_window.session.tracks[self.track_index].armed = False
            self.app_window.queue_all_redraws()

    def _on_mute_toggled(self, btn):
        if self._syncing:
            return
        self.app_window.session.tracks[self.track_index].muted = btn.get_active()
        self.app_window.queue_all_redraws()

    def _on_solo_toggled(self, btn):
        if self._syncing:
            return
        self.app_window.session.tracks[self.track_index].solo = btn.get_active()
        self.app_window.queue_all_redraws()

    def _on_vol_changed(self, scale):
        if self._syncing:
            return
        self.app_window.session.tracks[self.track_index].volume = scale.get_value()

    def _on_pan_changed(self, scale):
        if self._syncing:
            return
        self.app_window.session.tracks[self.track_index].pan = scale.get_value()


class VoicePanelWindow(Gtk.ApplicationWindow):
    """Main VoicePanel 4-Track Audio Studio Window."""

    APP_NAME = "VoicePanel"
    APP_VERSION = "1.0.0"

    _THEME_PALETTES = {
        True: {
            "win_bg": "#0b0f19",
            "win_fg": "#f8fafc",
            "hb_grad": "linear-gradient(to bottom, #111827, #0b0f19)",
            "hb_border": "rgba(0, 240, 255, 0.35)",
            "subtitle_fg": "#38bdf8",
            "btn_bg": "#161f33",
            "btn_fg": "#f8fafc",
            "btn_border": "rgba(56, 189, 248, 0.35)",
            "btn_hover_bg": "#1e2d4a",
            "btn_hover_fg": "#00f0ff",
            "btn_hover_border": "#00f0ff",
            "btn_checked_bg": "#0284c7",
            "btn_checked_fg": "#ffffff",
            "panel_bg": "#0f172a",
            "entry_bg": "#0d1322",
            "entry_fg": "#f8fafc",
            "entry_border": "rgba(56, 189, 248, 0.45)",
            "pop_bg": "#0f172a",
            "pop_fg": "#f8fafc",
            "pop_border": "rgba(0, 240, 255, 0.4)",
            "sel_bg": "#0284c7",
            "sel_fg": "#ffffff",
            "trough_bg": "#162038",
        },
        False: {
            "win_bg": "#f1f5f9",
            "win_fg": "#0f172a",
            "hb_grad": "linear-gradient(to bottom, #ffffff, #e2e8f0)",
            "hb_border": "rgba(2, 132, 199, 0.35)",
            "subtitle_fg": "#0284c7",
            "btn_bg": "#ffffff",
            "btn_fg": "#0f172a",
            "btn_border": "#cbd5e1",
            "btn_hover_bg": "#e0f2fe",
            "btn_hover_fg": "#0369a1",
            "btn_hover_border": "#0284c7",
            "btn_checked_bg": "#0284c7",
            "btn_checked_fg": "#ffffff",
            "panel_bg": "#e2e8f0",
            "entry_bg": "#ffffff",
            "entry_fg": "#0f172a",
            "entry_border": "#94a3b8",
            "pop_bg": "#ffffff",
            "pop_fg": "#0f172a",
            "pop_border": "#cbd5e1",
            "sel_bg": "#0284c7",
            "sel_fg": "#ffffff",
            "trough_bg": "#cbd5e1",
        },
    }

    _watermark_pixbufs: dict = {}

    def __init__(self, app=None, initial_files: Optional[List[str]] = None):
        super().__init__(application=app, title=self.APP_NAME)
        self.set_wmclass("voicepanel", "VoicePanel")
        self.set_default_size(1160, 740)

        self.dark_mode = True
        self._theme_css_provider = None
        self._updating_dark_toggle = False
        self._ui_timer_id = None

        self.session = MultiTrackSession(sample_rate=DEFAULT_SAMPLE_RATE)
        self.recorder = AudioRecorder(sample_rate=self.session.sample_rate)
        self.player = AudioPlayer(sample_rate=self.session.sample_rate)
        self.record_start_sample = 0
        self.overdub_enabled = True
        self.loop_enabled = False

        # Timeline viewport in seconds (at least 15s initial grid)
        self.view_start_sec = 0.0
        self.view_span_sec = 16.0

        self._load_app_icon()
        self._apply_theme(self.dark_mode)
        self._build_headerbar()
        self._build_main_ui()
        self._setup_shortcuts()
        self.reload_audio_devices()

        self.connect("key-press-event", self._on_window_key_press)
        self.connect("delete-event", self._on_delete_event)

        if initial_files:
            self._open_initial_files(initial_files)

        self.update_undo_redo_ui()
        self.update_time_readout()
        self.show_all()

    def _load_app_icon(self):
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voicepanel.svg")
        if os.path.isfile(icon_path):
            try:
                self.set_icon_from_file(icon_path)
                pixbufs = [
                    GdkPixbuf.Pixbuf.new_from_file_at_scale(icon_path, sz, sz, True)
                    for sz in (16, 24, 32, 48, 64, 128, 256)
                ]
                Gtk.Window.set_default_icon_list(pixbufs)
                self.set_icon_list(pixbufs)
            except Exception:
                pass

    @classmethod
    def get_watermark_pixbuf(cls, size: int = 128) -> Optional[GdkPixbuf.Pixbuf]:
        bucket = max(48, min(320, int(round(size / 16.0) * 16)))
        if bucket not in cls._watermark_pixbufs:
            svg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voicepanel.svg")
            pix = None
            if os.path.isfile(svg_path):
                try:
                    pix = GdkPixbuf.Pixbuf.new_from_file_at_scale(svg_path, bucket, bucket, True)
                except Exception:
                    pix = None
            cls._watermark_pixbufs[bucket] = pix
        return cls._watermark_pixbufs[bucket]

    def _apply_theme(self, dark_mode: bool):
        dark_mode = bool(dark_mode)
        self.dark_mode = dark_mode
        settings = Gtk.Settings.get_default()
        if settings:
            settings.set_property("gtk-application-prefer-dark-theme", dark_mode)

        p = self._THEME_PALETTES[dark_mode]
        css = f"""
        window, dialog, .background, scrolledwindow, viewport {{
            background-color: {p["win_bg"]};
            color: {p["win_fg"]};
        }}
        headerbar {{
            background: {p["hb_grad"]};
            border-bottom: 1px solid {p["hb_border"]};
            color: {p["win_fg"]};
            padding: 4px 6px;
        }}
        headerbar .title {{
            color: {p["win_fg"]};
            font-weight: 700;
        }}
        headerbar .subtitle {{
            color: {p["subtitle_fg"]};
            font-size: 90%;
        }}
        headerbar button, button {{
            background-image: none;
            background-color: {p["btn_bg"]};
            color: {p["btn_fg"]};
            border: 1px solid {p["btn_border"]};
            border-radius: 5px;
        }}
        headerbar button:hover, button:hover {{
            background-color: {p["btn_hover_bg"]};
            border-color: {p["btn_hover_border"]};
            color: {p["btn_hover_fg"]};
        }}
        headerbar button:checked, headerbar button:active,
        button:checked, button:active {{
            background-color: {p["btn_checked_bg"]};
            border-color: {p["btn_hover_border"]};
            color: {p["btn_checked_fg"]};
        }}
        button.arm-btn:checked, button.record-btn:checked {{
            background-color: #e11d48;
            border-color: #fb7185;
            color: #ffffff;
            box-shadow: 0 0 10px rgba(244, 63, 94, 0.55);
        }}
        button.mute-btn:checked {{
            background-color: #d97706;
            border-color: #fbbf24;
            color: #ffffff;
        }}
        button.solo-btn:checked {{
            background-color: #0284c7;
            border-color: #00f0ff;
            color: #ffffff;
        }}
        entry {{
            background-color: {p["entry_bg"]};
            color: {p["entry_fg"]};
            border: 1px solid {p["entry_border"]};
            border-radius: 4px;
            padding: 2px 6px;
        }}
        combobox button, combobox window, combobox menu {{
            background-color: {p["btn_bg"]};
            color: {p["btn_fg"]};
        }}
        scale trough {{
            background-color: {p["trough_bg"]};
            border-radius: 4px;
            min-height: 6px;
        }}
        scale highlight {{
            background-color: #0284c7;
            border-radius: 4px;
        }}
        scale slider {{
            background-color: #38bdf8;
            border: 1px solid #00f0ff;
            border-radius: 8px;
            min-width: 12px;
            min-height: 12px;
        }}
        popover, menu, .popup, treeview.view {{
            background-color: {p["pop_bg"]};
            color: {p["pop_fg"]};
            border: 1px solid {p["pop_border"]};
        }}
        treeview.view:selected, menuitem:hover, modelbutton:hover {{
            background-color: {p["sel_bg"]};
            color: {p["sel_fg"]};
        }}
        checkbutton, checkbutton label, label {{
            color: {p["win_fg"]};
        }}
        headerbar button:checked label, button:checked label {{
            color: {p["btn_checked_fg"]};
        }}
        .transport-clock {{
            font-family: 'JetBrains Mono', 'DejaVu Sans Mono', monospace;
            font-weight: 700;
            color: #00f0ff;
            background-color: {p["panel_bg"]};
            border: 1px solid {p["hb_border"]};
            border-radius: 5px;
            padding: 3px 10px;
        }}
        """.encode("utf-8")

        if self._theme_css_provider is None:
            self._theme_css_provider = Gtk.CssProvider()
            screen = Gdk.Screen.get_default()
            if screen:
                Gtk.StyleContext.add_provider_for_screen(
                    screen,
                    self._theme_css_provider,
                    Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
                )
        try:
            self._theme_css_provider.load_from_data(css)
        except Exception:
            pass

    def _build_headerbar(self):
        hb = Gtk.HeaderBar()
        hb.set_show_close_button(True)
        hb.set_title(self.APP_NAME)
        hb.set_subtitle("4-Track Audio Studio")
        self.header_bar = hb
        self.set_titlebar(hb)

        # Left: File & History controls
        file_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        Gtk.StyleContext.add_class(file_box.get_style_context(), "linked")

        btn_new = Gtk.Button.new_from_icon_name("document-new-symbolic", Gtk.IconSize.BUTTON)
        btn_new.set_tooltip_text("New 4-Track Project (Ctrl+N)")
        btn_new.connect("clicked", lambda b: self.on_new_project())
        file_box.pack_start(btn_new, False, False, 0)

        btn_open = Gtk.Button.new_from_icon_name("document-open-symbolic", Gtk.IconSize.BUTTON)
        btn_open.set_tooltip_text("Open Project (.vpanel) or Import Audio (Ctrl+O)")
        btn_open.connect("clicked", lambda b: self.on_open_or_import())
        file_box.pack_start(btn_open, False, False, 0)

        btn_save = Gtk.Button.new_from_icon_name("document-save-symbolic", Gtk.IconSize.BUTTON)
        btn_save.set_tooltip_text("Save VoicePanel Project (.vpanel) (Ctrl+S)")
        btn_save.connect("clicked", lambda b: self.on_save_project())
        file_box.pack_start(btn_save, False, False, 0)

        btn_export = Gtk.Button(label="Export")
        btn_export.set_tooltip_text("Export Mixed 4-Track Audio (.wav / .mp3 / .flac / .ogg) (Ctrl+E)")
        btn_export.connect("clicked", lambda b: self.on_export_mix())
        file_box.pack_start(btn_export, False, False, 0)

        hb.pack_start(file_box)

        hist_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        Gtk.StyleContext.add_class(hist_box.get_style_context(), "linked")
        hist_box.set_margin_start(6)

        self.btn_undo = Gtk.Button.new_from_icon_name("edit-undo-symbolic", Gtk.IconSize.BUTTON)
        self.btn_undo.set_tooltip_text("Undo (Ctrl+Z)")
        self.btn_undo.connect("clicked", lambda b: self.on_undo())
        hist_box.pack_start(self.btn_undo, False, False, 0)

        self.btn_redo = Gtk.Button.new_from_icon_name("edit-redo-symbolic", Gtk.IconSize.BUTTON)
        self.btn_redo.set_tooltip_text("Redo (Ctrl+Y)")
        self.btn_redo.connect("clicked", lambda b: self.on_redo())
        hist_box.pack_start(self.btn_redo, False, False, 0)

        hb.pack_start(hist_box)

        # Center-Left: Transport Bar
        trans_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        trans_box.set_margin_start(10)

        btn_rew = Gtk.Button.new_from_icon_name("media-skip-backward-symbolic", Gtk.IconSize.BUTTON)
        btn_rew.set_tooltip_text("Move Marker to Beginning (Home / Shift+<)")
        btn_rew.connect("clicked", lambda b: self.on_rewind())
        trans_box.pack_start(btn_rew, False, False, 0)

        btn_end = Gtk.Button.new_from_icon_name("media-skip-forward-symbolic", Gtk.IconSize.BUTTON)
        btn_end.set_tooltip_text("Move Marker to End (End / Shift+>)")
        btn_end.connect("clicked", lambda b: self.on_jump_to_end())
        trans_box.pack_start(btn_end, False, False, 0)

        self.btn_play = Gtk.ToggleButton()
        self.btn_play.set_image(Gtk.Image.new_from_icon_name("media-playback-start-symbolic", Gtk.IconSize.BUTTON))
        self.btn_play.set_tooltip_text("Play / Pause Mix (Space)")
        self.btn_play.connect("toggled", self._on_play_toggled)
        trans_box.pack_start(self.btn_play, False, False, 0)

        btn_stop = Gtk.Button.new_from_icon_name("media-playback-stop-symbolic", Gtk.IconSize.BUTTON)
        btn_stop.set_tooltip_text("Stop Playback / Recording (Esc)")
        btn_stop.connect("clicked", lambda b: self.stop_all())
        trans_box.pack_start(btn_stop, False, False, 0)

        self.btn_record = Gtk.ToggleButton(label="● REC")
        self.btn_record.set_tooltip_text("Record Microphone into Armed Track (Ctrl+R)")
        Gtk.StyleContext.add_class(self.btn_record.get_style_context(), "record-btn")
        self.btn_record.connect("toggled", self._on_record_toggled)
        trans_box.pack_start(self.btn_record, False, False, 0)

        self.btn_loop = Gtk.ToggleButton()
        self.btn_loop.set_image(Gtk.Image.new_from_icon_name("media-playlist-repeat-symbolic", Gtk.IconSize.BUTTON))
        self.btn_loop.set_tooltip_text("Loop Playback")
        self.btn_loop.connect("toggled", lambda b: setattr(self, "loop_enabled", b.get_active()))
        trans_box.pack_start(self.btn_loop, False, False, 0)

        hb.pack_start(trans_box)

        # Right: Menu, Dark Mode, Zoom, Mic Input
        btn_menu = Gtk.MenuButton()
        btn_menu.set_image(Gtk.Image.new_from_icon_name("open-menu-symbolic", Gtk.IconSize.BUTTON))
        btn_menu.set_tooltip_text("VoicePanel Options")
        btn_menu.set_popover(self._build_menu_popover(btn_menu))
        hb.pack_end(btn_menu)

        self.btn_dark_mode = Gtk.ToggleButton()
        self._dark_icon = Gtk.Image.new_from_icon_name("weather-clear-night-symbolic", Gtk.IconSize.BUTTON)
        self.btn_dark_mode.set_image(self._dark_icon)
        self.btn_dark_mode.set_active(True)
        self.btn_dark_mode.set_tooltip_text("Toggle Dark / Light Mode (Ctrl+Shift+D)")
        self.btn_dark_mode.connect("toggled", self._on_dark_toggled)
        hb.pack_end(self.btn_dark_mode)

        zoom_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        Gtk.StyleContext.add_class(zoom_box.get_style_context(), "linked")

        btn_z_out = Gtk.Button.new_from_icon_name("zoom-out-symbolic", Gtk.IconSize.BUTTON)
        btn_z_out.set_tooltip_text("Zoom Out Timeline (Ctrl+-)")
        btn_z_out.connect("clicked", lambda b: self.zoom_timeline(factor=1.4))
        zoom_box.pack_start(btn_z_out, False, False, 0)

        btn_z_fit = Gtk.Button.new_from_icon_name("zoom-fit-best-symbolic", Gtk.IconSize.BUTTON)
        btn_z_fit.set_tooltip_text("Fit All Tracks to Window")
        btn_z_fit.connect("clicked", lambda b: self.zoom_fit())
        zoom_box.pack_start(btn_z_fit, False, False, 0)

        btn_z_in = Gtk.Button.new_from_icon_name("zoom-in-symbolic", Gtk.IconSize.BUTTON)
        btn_z_in.set_tooltip_text("Zoom In Timeline (Ctrl++)")
        btn_z_in.connect("clicked", lambda b: self.zoom_timeline(factor=0.7))
        zoom_box.pack_start(btn_z_in, False, False, 0)

        hb.pack_end(zoom_box)

        self.combo_mic = Gtk.ComboBoxText()
        self.combo_mic.set_tooltip_text("Select Microphone Input Source")
        self.combo_mic.connect("changed", self._on_mic_changed)
        hb.pack_end(self.combo_mic)

    def _build_menu_popover(self, parent_btn):
        pop = Gtk.Popover.new(parent_btn)
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        vbox.set_margin_top(10)
        vbox.set_margin_bottom(10)
        vbox.set_margin_start(12)
        vbox.set_margin_end(12)

        items = [
            ("Reload Audio Devices", self.reload_audio_devices),
            ("Import Audio to Armed Track...", self.prompt_import_to_armed),
            ("Export Mixed Audio...", self.on_export_mix),
            ("Clear All Tracks", self.clear_all_tracks),
        ]
        for label_text, cb in items:
            btn = Gtk.ModelButton(label=label_text)
            btn.connect("clicked", lambda b, fn=cb: (pop.popdown(), fn()))
            vbox.pack_start(btn, False, False, 0)

        vbox.pack_start(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL), False, False, 4)

        self.chk_dark_mode = Gtk.CheckButton(label="Dark Mode (Ctrl+Shift+D)")
        self.chk_dark_mode.set_active(True)
        self.chk_dark_mode.connect("toggled", self._on_dark_toggled)
        vbox.pack_start(self.chk_dark_mode, False, False, 0)

        btn_about = Gtk.ModelButton(label="About VoicePanel")
        btn_about.connect("clicked", lambda b: (pop.popdown(), self.on_about()))
        vbox.pack_start(btn_about, False, False, 0)

        vbox.show_all()
        pop.add(vbox)
        return pop

    def _build_main_ui(self):
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.add(root)

        # Compact Edit Toolbar + Time Clock
        edit_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        edit_bar.set_margin_start(10)
        edit_bar.set_margin_end(10)
        edit_bar.set_margin_top(6)
        edit_bar.set_margin_bottom(2)

        edit_actions = [
            ("Cut", "Cut selected time region (Del)", lambda: self.run_edit("cut")),
            ("Silence", "Mute/Zero selected time region", lambda: self.run_edit("silence")),
            ("Trim", "Keep only selected region on active track", lambda: self.run_edit("trim")),
            ("Copy", "Copy selection (Ctrl+C)", lambda: self.run_edit("copy")),
            ("Paste", "Paste clipboard at playhead (Ctrl+V)", lambda: self.run_edit("paste")),
            ("Fade In", "Apply linear Fade In to selection or track", lambda: self.run_edit("fade_in")),
            ("Fade Out", "Apply linear Fade Out to selection or track", lambda: self.run_edit("fade_out")),
            ("Normalize", "Normalize active track/selection to -1 dBFS", lambda: self.run_edit("normalize")),
            ("Reverse", "Reverse active track/selection", lambda: self.run_edit("reverse")),
        ]
        for label, tip, cb in edit_actions:
            btn = Gtk.Button(label=label)
            btn.set_tooltip_text(tip)
            btn.connect("clicked", lambda b, fn=cb: fn())
            edit_bar.pack_start(btn, False, False, 0)

        self.chk_overdub = Gtk.CheckButton(label="Overdub Playback")
        self.chk_overdub.set_active(True)
        self.chk_overdub.set_tooltip_text("Play existing tracks while recording new vocals/audio into the armed track")
        self.chk_overdub.connect("toggled", lambda b: setattr(self, "overdub_enabled", b.get_active()))
        edit_bar.pack_start(self.chk_overdub, False, False, 8)

        self.lbl_clock = Gtk.Label(label="00:00.00 / 00:00.00")
        Gtk.StyleContext.add_class(self.lbl_clock.get_style_context(), "transport-clock")
        edit_bar.pack_end(self.lbl_clock, False, False, 0)

        root.pack_start(edit_bar, False, False, 0)

        # Ruler Row (aligned above waveform lanes, 268px left spacer matching TrackRow control strip)
        ruler_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        ruler_spacer = Gtk.Label(label="4-TRACK RACK  (Shift+Drag Clip)")
        ruler_spacer.set_size_request(268, 26)
        ruler_spacer.set_xalign(0.5)
        ruler_row.pack_start(ruler_spacer, False, False, 0)

        self.ruler = TimelineRuler(self)
        ruler_row.pack_start(self.ruler, True, True, 0)
        root.pack_start(ruler_row, False, False, 0)

        # Scrollable 4-Track Container
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        tracks_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        tracks_box.set_margin_start(4)
        tracks_box.set_margin_end(4)
        tracks_box.set_margin_bottom(4)

        self.track_rows: List[TrackRow] = []
        for i in range(NUM_TRACKS):
            row = TrackRow(i, self)
            self.track_rows.append(row)
            tracks_box.pack_start(row, True, True, 0)

        scroll.add(tracks_box)
        root.pack_start(scroll, True, True, 0)

        # Status bar
        status_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        status_bar.set_margin_start(12)
        status_bar.set_margin_end(12)
        status_bar.set_margin_bottom(4)
        self.lbl_status = Gtk.Label(label="Ready | Track 1 armed for microphone recording | 48000 Hz Stereo")
        self.lbl_status.set_xalign(0.0)
        status_bar.pack_start(self.lbl_status, True, True, 0)
        root.pack_end(status_bar, False, False, 0)

    def _setup_shortcuts(self):
        accel = Gtk.AccelGroup()
        self.add_accel_group(accel)
        bindings = [
            ("r", Gdk.ModifierType.CONTROL_MASK, self.toggle_record),
            ("n", Gdk.ModifierType.CONTROL_MASK, self.on_new_project),
            ("o", Gdk.ModifierType.CONTROL_MASK, self.on_open_or_import),
            ("s", Gdk.ModifierType.CONTROL_MASK, self.on_save_project),
            ("e", Gdk.ModifierType.CONTROL_MASK, self.on_export_mix),
            ("z", Gdk.ModifierType.CONTROL_MASK, self.on_undo),
            ("y", Gdk.ModifierType.CONTROL_MASK, self.on_redo),
            ("c", Gdk.ModifierType.CONTROL_MASK, lambda: self.run_edit("copy")),
            ("v", Gdk.ModifierType.CONTROL_MASK, lambda: self.run_edit("paste")),
            ("d", Gdk.ModifierType.CONTROL_MASK | Gdk.ModifierType.SHIFT_MASK, self.toggle_dark_mode),
            ("plus", Gdk.ModifierType.CONTROL_MASK, lambda: self.zoom_timeline(0.7)),
            ("equal", Gdk.ModifierType.CONTROL_MASK, lambda: self.zoom_timeline(0.7)),
            ("minus", Gdk.ModifierType.CONTROL_MASK, lambda: self.zoom_timeline(1.4)),
        ]
        for key, mod, cb in bindings:
            kv = Gdk.keyval_from_name(key)
            if kv != 0:
                accel.connect(kv, mod, Gtk.AccelFlags.VISIBLE, lambda ag, w, k, m, fn=cb: fn() or True)

    def _on_window_key_press(self, widget, event):
        focused = self.get_focus()
        kv = getattr(event, "keyval", 0)
        if isinstance(focused, Gtk.Entry):
            if kv == Gdk.KEY_Escape:
                self.set_focus(None)
                return True
            return False
        shift = bool(getattr(event, "state", 0) & Gdk.ModifierType.SHIFT_MASK)
        if kv == Gdk.KEY_space:
            self.toggle_play()
            return True
        if kv == Gdk.KEY_Escape:
            self.stop_all()
            return True
        if kv == Gdk.KEY_Home or kv == Gdk.KEY_less or (shift and kv == Gdk.KEY_comma):
            self.on_rewind()
            return True
        if kv == Gdk.KEY_End or kv == Gdk.KEY_greater or (shift and kv == Gdk.KEY_period):
            self.on_jump_to_end()
            return True
        if kv == Gdk.KEY_Delete:
            self.run_edit("cut")
            return True
        return False

    # ------------------------------------------------------------------
    # Coordinate / Viewport Helpers
    # ------------------------------------------------------------------
    def get_view_range(self) -> Tuple[int, int]:
        sr = self.session.sample_rate
        s0 = max(0, int(round(self.view_start_sec * sr)))
        s1 = max(s0 + sr // 10, int(round((self.view_start_sec + self.view_span_sec) * sr)))
        return s0, s1

    def sample_to_x(self, sample: int, width: int) -> float:
        s0, s1 = self.get_view_range()
        return ((sample - s0) / float(max(1, s1 - s0))) * float(max(1, width))

    def x_to_sample(self, x: float, width: int) -> int:
        s0, s1 = self.get_view_range()
        frac = max(0.0, min(1.0, float(x) / float(max(1, width))))
        return int(round(s0 + frac * (s1 - s0)))

    def zoom_timeline(self, factor: float = 1.0):
        self.view_span_sec = max(0.5, min(600.0, self.view_span_sec * factor))
        play_sec = self.session.playhead_sample / float(self.session.sample_rate)
        if play_sec < self.view_start_sec or play_sec > self.view_start_sec + self.view_span_sec:
            self.view_start_sec = max(0.0, play_sec - self.view_span_sec * 0.25)
        self.queue_all_redraws()

    def zoom_fit(self):
        total_sec = max(4.0, self.session.total_seconds() * 1.08)
        self.view_start_sec = 0.0
        self.view_span_sec = total_sec
        self.queue_all_redraws()

    def _ensure_sample_visible(self, sample: int):
        sec = sample / float(self.session.sample_rate)
        if sec > self.view_start_sec + self.view_span_sec * 0.94:
            self.view_start_sec = max(0.0, sec - self.view_span_sec * 0.5)
        elif sec < self.view_start_sec:
            self.view_start_sec = max(0.0, sec - self.view_span_sec * 0.1)

    def queue_all_redraws(self):
        self.ruler.queue_draw()
        for row in self.track_rows:
            row.canvas.queue_draw()

    def update_time_readout(self):
        cur_sec = self.session.playhead_sample / float(self.session.sample_rate)
        tot_sec = self.session.total_seconds()
        txt = f"{format_time(cur_sec)} / {format_time(tot_sec)}"
        if self.session.selection is not None:
            s0, s1 = sorted(self.session.selection)
            sel_sec = (s1 - s0) / float(self.session.sample_rate)
            txt += f"  [Sel: {sel_sec:.2f}s]"
        self.lbl_clock.set_text(txt)

    def update_undo_redo_ui(self):
        self.btn_undo.set_sensitive(self.session.can_undo())
        self.btn_redo.set_sensitive(self.session.can_redo())

    # ------------------------------------------------------------------
    # Audio Devices & Dark Mode
    # ------------------------------------------------------------------
    def reload_audio_devices(self):
        sources = AudioDeviceManager.list_input_sources()
        self._mic_ids = [dev for dev, _ in sources]
        self.combo_mic.remove_all()
        for dev, label in sources:
            self.combo_mic.append_text(label[:42])
        if self._mic_ids:
            self.combo_mic.set_active(0)
            self.recorder.device_name = self._mic_ids[0]
        self.set_status(f"Audio devices reloaded ({len(sources)} source(s) detected).")

    def _on_mic_changed(self, combo):
        idx = combo.get_active()
        if 0 <= idx < len(getattr(self, "_mic_ids", [])):
            self.recorder.device_name = self._mic_ids[idx]

    def set_dark_mode(self, enabled: bool):
        enabled = bool(enabled)
        self._apply_theme(enabled)
        self._updating_dark_toggle = True
        try:
            if hasattr(self, "btn_dark_mode"):
                self.btn_dark_mode.set_active(enabled)
            if hasattr(self, "chk_dark_mode"):
                self.chk_dark_mode.set_active(enabled)
            if hasattr(self, "_dark_icon"):
                self._dark_icon.set_from_icon_name(
                    "weather-clear-night-symbolic" if enabled else "weather-clear-symbolic",
                    Gtk.IconSize.BUTTON,
                )
        finally:
            self._updating_dark_toggle = False
        for row in self.track_rows:
            row.sync_from_model()
        self.queue_all_redraws()

    def _on_dark_toggled(self, btn):
        if self._updating_dark_toggle:
            return
        self.set_dark_mode(btn.get_active())

    def toggle_dark_mode(self):
        self.set_dark_mode(not self.dark_mode)

    def set_status(self, msg: str):
        self.lbl_status.set_text(msg)

    # ------------------------------------------------------------------
    # Track Selection, Arming, Recording & Playback
    # ------------------------------------------------------------------
    def select_active_track(self, index: int):
        self.session.active_track_index = max(0, min(index, NUM_TRACKS - 1))
        self.queue_all_redraws()

    def arm_track(self, index: int):
        self.session.arm_track(index, exclusive=True)
        for row in self.track_rows:
            row.sync_from_model()
        self.set_status(f"Track {index + 1} armed for microphone recording.")
        self.queue_all_redraws()

    def set_playhead(self, sample: int):
        was_playing = self.player.is_playing()
        self.session.playhead_sample = max(0, int(sample))
        if was_playing and not self.recorder.is_recording():
            self.player.start(
                self.session,
                start_sample=self.session.playhead_sample,
                loop=self.loop_enabled,
                on_finished=lambda: GLib.idle_add(self._on_playback_finished),
            )
        self.update_time_readout()
        self.queue_all_redraws()

    def on_rewind(self):
        track = self.session.get_active_track()
        if track.duration_samples > 0 and track.offset_samples > 0 and self.session.playhead_sample > track.offset_samples:
            target = track.offset_samples
        else:
            target = 0
        self.set_playhead(target)
        if target == 0:
            self.view_start_sec = 0.0
        else:
            self._ensure_sample_visible(target)
        self.queue_all_redraws()

    def on_jump_to_end(self):
        track = self.session.get_active_track()
        total = self.session.total_samples()
        if track.duration_samples > 0 and self.session.playhead_sample < track.end_samples:
            target = track.end_samples
        else:
            target = total
        self.set_playhead(target)
        self._ensure_sample_visible(target)
        self.queue_all_redraws()

    def toggle_play(self):
        self.btn_play.set_active(not self.btn_play.get_active())

    def _on_play_toggled(self, btn):
        if getattr(self, "_updating_transport", False):
            return
        if btn.get_active():
            self.start_playback()
        else:
            self.stop_all()

    def start_playback(self):
        if self.recorder.is_recording():
            return
        if self.session.total_samples() <= 0:
            self._updating_transport = True
            self.btn_play.set_active(False)
            self._updating_transport = False
            self.set_status("Nothing to play yet — record mic input or import an audio file first.")
            return

        if self.session.playhead_sample >= self.session.total_samples():
            self.session.playhead_sample = 0

        ok = self.player.start(
            self.session,
            start_sample=self.session.playhead_sample,
            loop=self.loop_enabled,
            on_finished=lambda: GLib.idle_add(self._on_playback_finished),
        )
        if not ok:
            self._updating_transport = True
            self.btn_play.set_active(False)
            self._updating_transport = False
            self.set_status("Audio playback backend unavailable.")
            return
        self._start_ui_timer()
        self.set_status("Playing 4-track mix...")

    def _on_playback_finished(self):
        if self.recorder.is_recording():
            return False
        self._updating_transport = True
        self.btn_play.set_active(False)
        self._updating_transport = False
        self._stop_ui_timer()
        for row in self.track_rows:
            row.meter.set_level(0.0)
        self.queue_all_redraws()
        self.set_status("Playback finished.")
        return False

    def toggle_record(self):
        self.btn_record.set_active(not self.btn_record.get_active())

    def _on_record_toggled(self, btn):
        if getattr(self, "_updating_transport", False):
            return
        if btn.get_active():
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if self.player.is_playing():
            self.player.stop()
        armed = self.session.get_armed_tracks()
        target = armed[0] if armed else self.session.get_active_track()
        target.armed = True
        self.session.active_track_index = target.index
        self.record_start_sample = self.session.playhead_sample

        if not self.recorder.start():
            self._updating_transport = True
            self.btn_record.set_active(False)
            self._updating_transport = False
            self.set_status(f"Recording failed: {self.recorder.last_error or 'microphone error'}")
            return

        if self.overdub_enabled and self.session.total_samples() > 0:
            self.player.start(
                self.session,
                start_sample=self.record_start_sample,
                loop=False,
                exclude_track=target.index,
            )

        self._start_ui_timer()
        self.set_status(f"Recording microphone into Track {target.index + 1} ({target.name})...")

    def stop_recording(self):
        if not self.recorder.is_recording():
            return
        self.player.stop()
        recorded = self.recorder.stop()
        self._stop_ui_timer()

        armed = self.session.get_armed_tracks()
        target_idx = armed[0].index if armed else self.session.active_track_index

        if len(recorded) > 0:
            self.session.commit_recording(recorded, self.record_start_sample, track_index=target_idx)
            dur = len(recorded) / float(self.session.sample_rate)
            self.set_status(f"Recorded {dur:.2f}s into Track {target_idx + 1}.")
            if self.session.total_seconds() > self.view_span_sec:
                self.view_span_sec = max(self.view_span_sec, self.session.total_seconds() * 1.15)
        else:
            self.set_status("Recording stopped (no audio captured).")

        for row in self.track_rows:
            row.meter.set_level(0.0)
        self.update_undo_redo_ui()
        self.update_time_readout()
        self.queue_all_redraws()

    def stop_all(self):
        was_rec = self.recorder.is_recording()
        self._updating_transport = True
        try:
            self.btn_play.set_active(False)
            self.btn_record.set_active(False)
        finally:
            self._updating_transport = False

        if was_rec:
            self.stop_recording()
        else:
            self.player.stop()
            self._stop_ui_timer()
            for row in self.track_rows:
                row.meter.set_level(0.0)
            self.queue_all_redraws()
            self.set_status("Stopped.")

    def _start_ui_timer(self):
        if self._ui_timer_id is None:
            self._ui_timer_id = GLib.timeout_add(33, self._on_ui_tick)

    def _stop_ui_timer(self):
        if self._ui_timer_id is not None:
            try:
                GLib.source_remove(self._ui_timer_id)
            except Exception:
                pass
            self._ui_timer_id = None

    def _on_ui_tick(self):
        if self.recorder.is_recording():
            rec_frames = self.recorder.get_recorded_frames_count()
            self.session.playhead_sample = self.record_start_sample + rec_frames
            self._ensure_sample_visible(self.session.playhead_sample)
            armed = self.session.get_armed_tracks()
            armed_idx = armed[0].index if armed else -1
            for row in self.track_rows:
                row.meter.set_level(self.recorder.live_peak if row.track_index == armed_idx else 0.0)
            self.update_time_readout()
            self.queue_all_redraws()
            return True

        if self.player.is_playing():
            self.session.playhead_sample = self.player.current_sample
            self._ensure_sample_visible(self.session.playhead_sample)
            audible = {t.index for t in self.session.active_audible_tracks()}
            for row in self.track_rows:
                row.meter.set_level(self.player.live_peak * row.app_window.session.tracks[row.track_index].volume if row.track_index in audible else 0.0)
            self.update_time_readout()
            self.queue_all_redraws()
            return True

        self._ui_timer_id = None
        return False

    # ------------------------------------------------------------------
    # Editing, File I/O & Dialogs
    # ------------------------------------------------------------------
    def run_edit(self, op: str):
        ok = False
        idx = self.session.active_track_index
        if op == "cut":
            ok = self.session.cut_selection(idx)
        elif op == "silence":
            ok = self.session.silence_selection(idx)
        elif op == "trim":
            ok = self.session.trim_to_selection(idx)
        elif op == "copy":
            ok = self.session.copy_selection(idx)
        elif op == "paste":
            ok = self.session.paste_at_playhead(idx)
        elif op == "fade_in":
            ok = self.session.fade_selection(True, idx)
        elif op == "fade_out":
            ok = self.session.fade_selection(False, idx)
        elif op == "normalize":
            ok = self.session.normalize_track(-1.0, idx)
        elif op == "reverse":
            ok = self.session.reverse_selection(idx)

        if ok:
            self.update_undo_redo_ui()
            self.update_time_readout()
            self.queue_all_redraws()
            self.set_status(f"Applied '{op}' to Track {idx + 1}.")
        else:
            self.set_status(f"Select a range or load audio on Track {idx + 1} for '{op}'.")

    def on_undo(self):
        if self.session.undo():
            for row in self.track_rows:
                row.sync_from_model()
            self.update_undo_redo_ui()
            self.update_time_readout()
            self.queue_all_redraws()
            self.set_status("Undone.")

    def on_redo(self):
        if self.session.redo():
            for row in self.track_rows:
                row.sync_from_model()
            self.update_undo_redo_ui()
            self.update_time_readout()
            self.queue_all_redraws()
            self.set_status("Redone.")

    def clear_track(self, track_index: int):
        self.session.clear_track(track_index)
        self.update_undo_redo_ui()
        self.update_time_readout()
        self.queue_all_redraws()
        self.set_status(f"Cleared Track {track_index + 1}.")

    def clear_all_tracks(self):
        self.stop_all()
        self.session.push_undo()
        for t in self.session.tracks:
            t.samples = np.zeros((0, 2), dtype=np.float32)
            t.offset_samples = 0
        self.session.playhead_sample = 0
        self.session.selection = None
        for row in self.track_rows:
            row.sync_from_model()
        self.update_undo_redo_ui()
        self.update_time_readout()
        self.queue_all_redraws()
        self.set_status("All 4 tracks cleared.")

    def on_new_project(self):
        self.stop_all()
        self.session = MultiTrackSession(sample_rate=DEFAULT_SAMPLE_RATE)
        self.view_start_sec = 0.0
        self.view_span_sec = 16.0
        for row in self.track_rows:
            row.sync_from_model()
        self.update_undo_redo_ui()
        self.update_time_readout()
        self.queue_all_redraws()
        self.set_status("New 4-track VoicePanel project ready.")

    def import_file_to_track(self, path: str, track_index: int):
        try:
            dur = self.session.load_file_to_track(path, track_index)
            self.track_rows[track_index].sync_from_model()
            self.zoom_fit()
            self.update_undo_redo_ui()
            self.update_time_readout()
            self.set_status(f"Loaded {os.path.basename(path)} ({dur:.2f}s) into Track {track_index + 1}.")
        except Exception as e:
            self._show_error(str(e))

    def _open_initial_files(self, paths: List[str]):
        track_slot = 0
        for p in paths:
            if not os.path.isfile(p):
                continue
            if p.lower().endswith(".vpanel"):
                try:
                    self.session.load_project(p)
                    for row in self.track_rows:
                        row.sync_from_model()
                    self.zoom_fit()
                    self.set_status(f"Opened project: {os.path.basename(p)}")
                    return
                except Exception as e:
                    self._show_error(str(e))
                    return
            elif track_slot < NUM_TRACKS:
                self.import_file_to_track(p, track_slot)
                track_slot += 1

    def prompt_import_to_armed(self):
        armed = self.session.get_armed_tracks()
        idx = armed[0].index if armed else self.session.active_track_index
        self.prompt_import_to_track(idx)

    def prompt_import_to_track(self, track_index: int):
        dlg = Gtk.FileChooserDialog(
            title=f"Import Audio to Track {track_index + 1}",
            parent=self,
            action=Gtk.FileChooserAction.OPEN,
        )
        dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
        flt = Gtk.FileFilter()
        flt.set_name("Audio Files (WAV, MP3, OGG, FLAC, M4A)")
        for ext in SUPPORTED_AUDIO_EXTS:
            flt.add_pattern(f"*{ext}")
            flt.add_pattern(f"*{ext.upper()}")
        dlg.add_filter(flt)

        res = dlg.run()
        path = dlg.get_filename()
        dlg.destroy()
        if res == Gtk.ResponseType.OK and path:
            self.import_file_to_track(path, track_index)

    def on_open_or_import(self):
        dlg = Gtk.FileChooserDialog(
            title="Open Project (.vpanel) or Import Audio",
            parent=self,
            action=Gtk.FileChooserAction.OPEN,
        )
        dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
        flt = Gtk.FileFilter()
        flt.set_name("VoicePanel Projects & Audio")
        flt.add_pattern("*.vpanel")
        for ext in SUPPORTED_AUDIO_EXTS:
            flt.add_pattern(f"*{ext}")
        dlg.add_filter(flt)

        res = dlg.run()
        path = dlg.get_filename()
        dlg.destroy()
        if res == Gtk.ResponseType.OK and path:
            if path.lower().endswith(".vpanel"):
                try:
                    self.stop_all()
                    self.session.load_project(path)
                    for row in self.track_rows:
                        row.sync_from_model()
                    self.zoom_fit()
                    self.update_undo_redo_ui()
                    self.update_time_readout()
                    self.set_status(f"Opened project: {os.path.basename(path)}")
                except Exception as e:
                    self._show_error(str(e))
            else:
                self.import_file_to_track(path, self.session.active_track_index)

    def on_save_project(self):
        if self.session.project_path:
            out_path = self.session.project_path
        else:
            dlg = Gtk.FileChooserDialog(
                title="Save VoicePanel Project",
                parent=self,
                action=Gtk.FileChooserAction.SAVE,
            )
            dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_SAVE, Gtk.ResponseType.OK)
            dlg.set_do_overwrite_confirmation(True)
            dlg.set_current_name("session.vpanel")
            res = dlg.run()
            out_path = dlg.get_filename()
            dlg.destroy()
            if res != Gtk.ResponseType.OK or not out_path:
                return
            if not out_path.lower().endswith(".vpanel"):
                out_path += ".vpanel"

        try:
            self.session.save_project(out_path)
            self.set_status(f"Saved project: {os.path.basename(out_path)}")
        except Exception as e:
            self._show_error(str(e))

    def on_export_mix(self):
        dlg = Gtk.FileChooserDialog(
            title="Export 4-Track Mixdown",
            parent=self,
            action=Gtk.FileChooserAction.SAVE,
        )
        dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_SAVE, Gtk.ResponseType.OK)
        dlg.set_do_overwrite_confirmation(True)
        dlg.set_current_name("voicepanel_mix.wav")
        res = dlg.run()
        out_path = dlg.get_filename()
        dlg.destroy()
        if res == Gtk.ResponseType.OK and out_path:
            try:
                self.session.export_mix(out_path)
                self.set_status(f"Exported mixdown: {os.path.basename(out_path)}")
            except Exception as e:
                self._show_error(str(e))

    def on_about(self):
        dlg = Gtk.AboutDialog(transient_for=self, modal=True)
        dlg.set_program_name(self.APP_NAME)
        dlg.set_version(self.APP_VERSION)
        dlg.set_comments("4-Track Audio Editor & Microphone Studio on the PythonPanel Neon Pad.")
        pix = self.get_watermark_pixbuf(128)
        if pix:
            dlg.set_logo(pix)
        dlg.run()
        dlg.destroy()

    def _show_error(self, msg: str):
        dlg = Gtk.MessageDialog(
            transient_for=self,
            flags=0,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text=msg,
        )
        dlg.run()
        dlg.destroy()

    def _on_delete_event(self, widget, event):
        self.stop_all()
        return False


class VoicePanelApp(Gtk.Application):
    def __init__(self):
        super().__init__(
            application_id="org.linuxmint.voicepanel",
            flags=Gio.ApplicationFlags.HANDLES_OPEN | Gio.ApplicationFlags.NON_UNIQUE,
        )
        self.initial_files: List[str] = []

    def do_activate(self):
        win = VoicePanelWindow(self, initial_files=self.initial_files)
        win.present()

    def do_open(self, files, n_files, hint):
        paths = [f.get_path() for f in files if f.get_path()]
        win = VoicePanelWindow(self, initial_files=paths)
        win.present()


def main():
    GLib.set_prgname("voicepanel")
    GLib.set_application_name("VoicePanel")
    app = VoicePanelApp()
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        app.initial_files = [os.path.abspath(p) for p in sys.argv[1:]]
    return app.run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
