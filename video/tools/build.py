"""Assemble the demo video: cards (PIL) + dashboard stills (Playwright) + ElevenLabs voice-over -> out MP4 via ffmpeg."""
import json, pathlib, subprocess
from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
FR, VO, WK = ROOT / "frames", ROOT / "audio" / "vo", ROOT / "work"
WK.mkdir(exist_ok=True)
W, H, FPS = 1920, 1080, 30
F = lambda n, s: ImageFont.truetype(f"C:/Windows/Fonts/{n}.ttf", s)
BG, INK, TEAL, MUTE = (16, 24, 32), (232, 237, 242), (53, 182, 194), (154, 166, 180)


def card(path, lines, sub=None, foot=None):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 14, H], fill=TEAL)
    y = 300
    for text, size, col, font in lines:
        d.text((150, y), text, font=F(font, size), fill=col); y += int(size * 1.35)
    if sub:
        y += 20
        for t in sub:
            d.text((150, y), t, font=F("segoeui", 34), fill=MUTE); y += 52
    if foot:
        d.text((150, H - 120), foot, font=F("segoeui", 28), fill=MUTE)
    im.save(path)


card(FR / "card_title.png",
     [("PRODROME", 150, TEAL, "segoeuib"), ("Population & workforce health early warning", 54, INK, "segoeui")],
     sub=["From scattered signals to proportionate, explainable, privacy-preserving action.",
          "Future Health Digital Innovation Challenge  ·  Challenge 06  ·  Population Health"])
card(FR / "card_end.png",
     [("PRODROME", 110, TEAL, "segoeuib"), ("From signals to action, before the burden grows.", 50, INK, "segoeui")],
     sub=["Multi-signal  ·  Explainable  ·  Alert-budgeted  ·  Privacy by design  ·  Back-tested",
          "Real weather and air-quality data (Open-Meteo, CC BY 4.0)  ·  Health and workplace data are SIMULATED"],
     foot="Voice-over: ElevenLabs  ·  Built solo for the Future Health Digital Innovation Challenge")

dur = json.loads((VO / "durations.json").read_text())
# scene -> list of (frame, share of segment duration)
PLAN = {
    "s1": [("card_title", 0.60), ("f01_live", 0.40)],
    "s2": [("f01_live", 0.62), ("f02_detail", 0.38)],
    "s3": [("f02_detail", 0.35), ("f03_action", 0.65)],
    "s4": [("f04_falsealarm", 1.0)],
    "s5": [("f05_backtest", 1.0)],
    "s6": [("f07_missing", 1.0)],
    "s7": [("f08_privacy", 0.5), ("f09_employer", 0.5)],
    "s8": [("f10_impact", 1.0)],
    "s9": [("card_end", 1.0)],
}
clips = []
for sid, parts in PLAN.items():
    for k, (fr, share) in enumerate(parts):
        t = dur[sid] * share + (0.35 if k == len(parts) - 1 else 0)   # tiny tail so sentences breathe
        out = WK / f"{sid}_{k}.mp4"; n = int(t * FPS)
        zoom = "min(zoom+0.00035,1.05)"
        vf = (f"scale=3840:2160,zoompan=z='{zoom}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={n}:s={W}x{H}:fps={FPS},"
              f"fade=t=in:st=0:d=0.25,format=yuv420p")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(FR / f"{fr}.png"), "-vf", vf,
                        "-frames:v", str(n), "-c:v", "libx264", "-crf", "18", "-preset", "medium", str(out)], check=True)
        clips.append(out)
(WK / "list.txt").write_text("".join(f"file '{c.as_posix()}'\n" for c in clips))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(WK / "list.txt"), "-c", "copy", str(WK / "video_only.mp4")], check=True)
# audio: concat segments with a short pad matching the tail added above
alist = []
for sid in PLAN:
    pad = WK / f"{sid}_pad.mp3"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(VO / f"{sid}.mp3"), "-af", "apad=pad_dur=0.35", "-ar", "44100", "-ac", "2", str(pad)], check=True)
    alist.append(pad)
(WK / "alist.txt").write_text("".join(f"file '{a.as_posix()}'\n" for a in alist))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(WK / "alist.txt"), "-c:a", "pcm_s16le", str(WK / "voice.wav")], check=True)
final = ROOT / "Prodrome_demo.mp4"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(WK / "video_only.mp4"), "-i", str(WK / "voice.wav"), "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", str(final)], check=True)
out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,size", "-of", "csv=p=0", str(final)], capture_output=True, text=True).stdout
print("built", final, out.strip())
