#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$HOME/.local/share/applications"
ICONS_DIR="$HOME/.local/share/icons/hicolor/scalable/apps"
BIN_DIR="$HOME/.local/bin"

echo "=========================================================="
echo " Installing VoicePanel for Linux Mint Cinnamon"
echo " (4-Track Audio Editor & Microphone Studio)"
echo "=========================================================="

chmod +x "$DIR/run.sh" "$DIR/voicepanel.py"

mkdir -p "$APP_DIR" "$ICONS_DIR" "$BIN_DIR" "$HOME/.icons" "$HOME/.local/share/pixmaps"

# 1. Install high-DPI vector icon & multi-size PNG icons
cp -f "$DIR/voicepanel.svg" "$ICONS_DIR/voicepanel.svg"
cp -f "$DIR/voicepanel.svg" "$HOME/.icons/voicepanel.svg"
cp -f "$DIR/voicepanel.svg" "$HOME/.local/share/pixmaps/voicepanel.svg"

python3 -B -c "
import os, gi
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import GdkPixbuf
src = '$DIR/voicepanel.svg'
base = os.path.expanduser('~/.local/share/icons/hicolor')
for sz in (16, 24, 32, 48, 64, 128, 256, 512):
    out_dir = os.path.join(base, f'{sz}x{sz}', 'apps')
    os.makedirs(out_dir, exist_ok=True)
    pb = GdkPixbuf.Pixbuf.new_from_file_at_scale(src, sz, sz, True)
    pb.savev(os.path.join(out_dir, 'voicepanel.png'), 'png', [], [])
" 2>/dev/null || true

echo "[+] Installed VoicePanel SVG & panel icon set: $ICONS_DIR/voicepanel.svg"

# 2. Install executable launcher in ~/.local/bin
ln -sf "$DIR/run.sh" "$BIN_DIR/voicepanel"
echo "[+] Created CLI executable link: $BIN_DIR/voicepanel"

# 3. Install desktop launcher
cat <<EOF > "$APP_DIR/voicepanel.desktop"
[Desktop Entry]
Name=VoicePanel
Comment=4-track audio editor and microphone recorder in PythonPanel neon style
GenericName=4-Track Audio Studio
Exec=$DIR/run.sh %F
Icon=$ICONS_DIR/voicepanel.svg
Terminal=false
Type=Application
Categories=AudioVideo;Audio;Recorder;GTK;
MimeType=audio/wav;audio/x-wav;audio/mpeg;audio/ogg;audio/flac;
StartupNotify=true
StartupWMClass=voicepanel
Keywords=audio;recorder;microphone;multitrack;4-track;voice;editor;wav;mp3;voicepanel;
Actions=NewWindow;

[Desktop Action NewWindow]
Name=Open a New Window
Exec=$DIR/run.sh
EOF

chmod +x "$APP_DIR/voicepanel.desktop"
echo "[+] Installed desktop entry: $APP_DIR/voicepanel.desktop"

if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APP_DIR" || true
fi

if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" 2>/dev/null || true
fi

if command -v gsettings >/dev/null 2>&1; then
    python3 -B -c "
import gi
from gi.repository import Gio
try:
    settings = Gio.Settings.new('org.cinnamon')
    favs = list(settings.get_strv('favorite-apps'))
    if 'voicepanel.desktop' not in favs:
        favs.append('voicepanel.desktop')
    settings.set_strv('favorite-apps', favs)
    print('[+] Added VoicePanel to Cinnamon favorite apps')
except Exception:
    pass
" || true
fi

python3 -B -c "
import os, glob, json
for cfg_path in glob.glob(os.path.expanduser('~/.config/cinnamon/spices/grouped-window-list@cinnamon.org/*.json')):
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        pinned = data.get('pinned-apps', {}).get('value')
        if isinstance(pinned, list) and 'voicepanel.desktop' not in pinned:
            pinned.append('voicepanel.desktop')
            with open(cfg_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            print('[+] Pinned VoicePanel to Cinnamon grouped-window-list panel')
    except Exception:
        pass
" || true

echo ""
echo "=========================================================="
echo " VoicePanel Installation Complete!"
echo "=========================================================="
echo "You can launch VoicePanel anytime from your Cinnamon Menu"
echo "(under Sound & Video / Favorites) or from terminal by typing:"
echo "  voicepanel"
echo "=========================================================="
