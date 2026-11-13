# VoicePanel

A clean, fast **4-Track Audio Editor & Microphone Studio** for Linux Mint Cinnamon, built in the PythonPanel / PDFpanel dark-neon family style with the matching studio mic + 4-bar waveform yoga-pad logo and canvas watermark.

## Features
- **4 Independent Stereo Tracks**:
  - **ARM (`● ARM`)**, **Mute (`M`)**, **Solo (`S`)**, **Volume (`0–150%`)**, **Stereo Pan (`L/C/R`)**, editable track names, and live neon LED peak meters.
- **Direct Microphone Recording + Overdubbing**:
  - Records directly from selectable system microphones (`parec` / PipeWire / ALSA fallback) into whichever track is armed, with real-time red waveform growth and optional headphone overdub playback of existing tracks.
- **Interactive Waveform Editing (with Undo/Redo)**:
  - Click to scrub the playhead, drag to select ranges, `Shift+Drag` to align/slide clips in time, `Ctrl+Scroll` to zoom.
  - One-click **Cut**, **Silence**, **Trim**, **Copy**, **Paste**, **Fade In**, **Fade Out**, **Normalize (-1 dBFS)**, and **Reverse**.
- **Drag-and-Drop & Multi-Format Export**:
  - Drop `.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`, or `.opus` files onto any track lane.
  - Save/reopen `.vpanel` 4-track sessions or export full 4-track stereo mixdowns (`.wav`, `.mp3`, `.flac`, `.ogg`).
- **Standardized Dark-Neon UI + Light Mode Toggle**:
  - Neon blue & `#0b0f19` dark studio UI with translucent VoicePanel yoga-pad watermark and `Ctrl+Shift+D` Dark/Light mode switching.

## Installation & Launch
```bash
./install.sh
voicepanel
```

## Testing
```bash
python3 -B test_voicepanel.py -v
```
