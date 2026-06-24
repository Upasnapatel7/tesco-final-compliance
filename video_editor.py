"""
video_editor.py  —  AI-Powered Video Editor
============================================
Trim, enhance, adjust, apply filters, change background,
add overlays, and export — all using OpenCV + imageio + ffmpeg.
No moviepy required.
"""

import os, io, subprocess, tempfile, math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
import imageio

# ─────────────────────────────────────────────────────────────────────────────
#  VIDEO INFO
# ─────────────────────────────────────────────────────────────────────────────
def get_video_info(video_path: str) -> dict:
    """Get video metadata using ffprobe."""
    try:
        cmd = [
            r"C:\ffmpeg\bin\ffprobe.exe", "-v", "quiet","-print_format", "json",
            "-show_streams", "-show_format", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        import json
        data = json.loads(result.stdout)
        video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
        audio_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
        duration = float(data.get("format", {}).get("duration", 0))
        fps_str = video_stream.get("r_frame_rate", "25/1")
        fps_parts = fps_str.split("/")
        fps = float(fps_parts[0]) / max(float(fps_parts[1]), 1) if len(fps_parts) == 2 else 25.0
        return {
            "width":    int(video_stream.get("width",  0)),
            "height":   int(video_stream.get("height", 0)),
            "fps":      round(fps, 2),
            "duration": round(duration, 2),
            "codec":    video_stream.get("codec_name", "unknown"),
            "has_audio": audio_stream is not None,
            "audio_codec": audio_stream.get("codec_name", "none") if audio_stream else "none",
            "size_mb":  round(os.path.getsize(video_path) / 1024**2, 2) if os.path.exists(video_path) else 0,
        }
    except Exception as e:
        return {"error": str(e), "width":0,"height":0,"fps":25,"duration":0,"has_audio":False}

# ─────────────────────────────────────────────────────────────────────────────
#  EXTRACT FRAMES FOR PREVIEW
# ─────────────────────────────────────────────────────────────────────────────
def extract_frames(video_path: str, n_frames: int = 8) -> list:
    """Extract N evenly-spaced frames from video for preview grid."""
    frames = []
    try:
        reader = imageio.get_reader(video_path)
        total  = reader.count_frames()
        if total == 0 or total == float("inf"):
            # fallback: read first 8
            for i, frame in enumerate(reader):
                if i >= n_frames: break
                frames.append(Image.fromarray(frame))
        else:
            indices = [int(i * total / n_frames) for i in range(n_frames)]
            for idx in indices:
                try:
                    frame = reader.get_data(min(idx, total-1))
                    frames.append(Image.fromarray(frame))
                except Exception:
                    pass
        reader.close()
    except Exception as e:
        pass
    return frames

def extract_thumbnail(video_path: str, time_sec: float = 1.0) -> Optional[Image.Image]:
    """Extract a single thumbnail at given timestamp."""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_num = int(time_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception:
        pass
    return None

# ─────────────────────────────────────────────────────────────────────────────
#  FFMPEG WRAPPER — core processing
# ─────────────────────────────────────────────────────────────────────────────
def _run_ffmpeg(args: list, timeout: int = 120) -> tuple:
    """Run ffmpeg command, return (success, stderr)."""
    cmd = [r"C:\ffmpeg\bin\ffmpeg.exe", "-y"] + args
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.returncode == 0, result.stderr

# ─────────────────────────────────────────────────────────────────────────────
#  TRIM
# ─────────────────────────────────────────────────────────────────────────────
def trim_video(input_path: str, output_path: str,
               start_sec: float, end_sec: float) -> tuple:
    """Trim video to [start_sec, end_sec]."""
    duration = max(0.1, end_sec - start_sec)
    ok, err = _run_ffmpeg([
        "-ss", str(start_sec),
        "-i", input_path,
        "-t", str(duration),
        "-c", "copy",
        output_path
    ])
    return ok, err

# ─────────────────────────────────────────────────────────────────────────────
#  VIDEO QUALITY & RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────
def change_resolution(input_path: str, output_path: str,
                       preset: str = "1080p") -> tuple:
    RESOLUTIONS = {
        "4K (3840×2160)":  "3840:2160",
        "1080p (1920×1080)": "1920:1080",
        "720p (1280×720)":   "1280:720",
        "480p (854×480)":    "854:480",
        "360p (640×360)":    "640:360",
        "Square (1080×1080)":"1080:1080",
        "Stories (1080×1920)":"1080:1920",
        "Landscape (1200×628)":"1200:628",
    }
    scale = RESOLUTIONS.get(preset, "1280:720")
    ok, err = _run_ffmpeg([
        "-i", input_path,
        "-vf", f"scale={scale}:force_original_aspect_ratio=decrease,pad={scale}:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        output_path
    ])
    return ok, err

def change_quality(input_path: str, output_path: str, quality: str = "High") -> tuple:
    """Change output quality/bitrate."""
    crf_map = {"Ultra (best)": "16", "High": "20", "Medium": "26", "Low (smallest)": "34"}
    crf = crf_map.get(quality, "20")
    ok, err = _run_ffmpeg([
        "-i", input_path,
        "-c:v", "libx264", "-crf", crf, "-preset", "medium",
        "-c:a", "aac", "-b:a", "128k",
        output_path
    ])
    return ok, err

# ─────────────────────────────────────────────────────────────────────────────
#  VIDEO ADJUSTMENTS  (brightness, contrast, saturation, sharpness, exposure)
# ─────────────────────────────────────────────────────────────────────────────
def adjust_video(input_path: str, output_path: str,
                 brightness: float = 0.0,   # -1.0 to 1.0
                 contrast: float = 1.0,     # 0.0 to 3.0
                 saturation: float = 1.0,   # 0.0 to 3.0
                 gamma: float = 1.0,        # exposure: 0.1 to 3.0
                 sharpness: float = 0.0,    # 0.0 to 5.0
                 ) -> tuple:
    """Apply colour/light/sharpness adjustments using ffmpeg eq filter."""
    # ffmpeg eq filter: brightness (-1..1), contrast (0..100 clamped), saturation (0..3), gamma (0.1..10)
    filters = [
        f"eq=brightness={brightness:.2f}:contrast={contrast:.2f}:saturation={saturation:.2f}:gamma={gamma:.2f}"
    ]
    if sharpness > 0:
        # unsharp mask: luma_amount
        filters.append(f"unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount={sharpness:.1f}")
    vf = ",".join(filters)
    ok, err = _run_ffmpeg([
        "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "20", "-preset", "fast",
        "-c:a", "copy",
        output_path
    ])
    return ok, err

# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR FILTERS
# ─────────────────────────────────────────────────────────────────────────────
_FILTER_PRESETS = {
    "None":       "",
    "Warm":       "eq=saturation=1.2,colorbalance=rs=0.05:gs=0:bs=-0.05",
    "Cool":       "eq=saturation=1.1,colorbalance=rs=-0.05:gs=0:bs=0.08",
    "Vibrant":    "eq=contrast=1.2:saturation=1.5",
    "Matte":      "eq=contrast=0.85:saturation=0.75:brightness=0.05",
    "Black & White": "hue=s=0,eq=contrast=1.2",
    "Vintage":    "eq=brightness=-0.02:contrast=0.9:saturation=0.8,colorbalance=rs=0.08:gs=0.02:bs=-0.1",
    "Cinematic":  "eq=contrast=1.1:saturation=0.85,colorbalance=rs=0.03:gs=0:bs=-0.03,vignette",
    "HDR":        "eq=contrast=1.3:saturation=1.4:brightness=0.02",
    "Faded":      "eq=contrast=0.7:saturation=0.6:brightness=0.08",
}

def apply_filter(input_path: str, output_path: str, filter_name: str = "None") -> tuple:
    """Apply a colour filter preset to the video."""
    vf = _FILTER_PRESETS.get(filter_name, "")
    if not vf:
        # Just copy
        ok, err = _run_ffmpeg(["-i", input_path, "-c", "copy", output_path])
    else:
        ok, err = _run_ffmpeg([
            "-i", input_path,
            "-vf", vf,
            "-c:v", "libx264", "-crf", "20", "-preset", "fast",
            "-c:a", "copy",
            output_path
        ])
    return ok, err

# ─────────────────────────────────────────────────────────────────────────────
#  SPEED
# ─────────────────────────────────────────────────────────────────────────────
def change_speed(input_path: str, output_path: str, speed: float = 1.0) -> tuple:
    """Change playback speed. 0.5 = half speed, 2.0 = double speed."""
    speed = max(0.25, min(4.0, speed))
    pts_factor = 1.0 / speed
    atempo = speed if 0.5 <= speed <= 2.0 else speed  # clamp for atempo
    ok, err = _run_ffmpeg([
        "-i", input_path,
        "-filter_complex",
        f"[0:v]setpts={pts_factor:.4f}*PTS[v];[0:a]atempo={atempo:.4f}[a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-crf", "20",
        output_path
    ])
    return ok, err

# ─────────────────────────────────────────────────────────────────────────────
#  FLIP / ROTATE
# ─────────────────────────────────────────────────────────────────────────────
def flip_rotate(input_path: str, output_path: str,
                flip_h: bool = False, flip_v: bool = False,
                rotate: int = 0) -> tuple:
    filters = []
    if flip_h: filters.append("hflip")
    if flip_v: filters.append("vflip")
    if rotate == 90:  filters.append("transpose=1")
    elif rotate == 180: filters.append("transpose=1,transpose=1")
    elif rotate == 270: filters.append("transpose=2")
    if not filters:
        return _run_ffmpeg(["-i", input_path, "-c", "copy", output_path])
    vf = ",".join(filters)
    return _run_ffmpeg([
        "-i", input_path, "-vf", vf,
        "-c:v", "libx264", "-crf", "20", "-c:a", "copy", output_path
    ])

# ─────────────────────────────────────────────────────────────────────────────
#  AUDIO PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def audio_enhance(input_path: str, output_path: str,
                  volume: float = 1.0,
                  noise_reduce: bool = False,
                  normalize: bool = False,
                  bass_boost: bool = False,
                  ) -> tuple:
    """Audio enhancement using ffmpeg filters."""
    audio_filters = []
    if volume != 1.0:
        audio_filters.append(f"volume={volume:.2f}")
    if normalize:
        audio_filters.append("loudnorm")
    if noise_reduce:
        # High-pass filter to reduce low-frequency noise
        audio_filters.append("highpass=f=80,lowpass=f=8000")
    if bass_boost:
        audio_filters.append("bass=g=5:f=110")
    if audio_filters:
        af = ",".join(audio_filters)
        ok, err = _run_ffmpeg([
            "-i", input_path,
            "-af", af,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            output_path
        ])
    else:
        ok, err = _run_ffmpeg(["-i", input_path, "-c", "copy", output_path])
    return ok, err

def mute_audio(input_path: str, output_path: str) -> tuple:
    return _run_ffmpeg(["-i", input_path, "-an", "-c:v", "copy", output_path])

def add_background_music(video_path: str, audio_path: str, output_path: str,
                          music_volume: float = 0.3) -> tuple:
    """Mix background music with original video audio."""
    return _run_ffmpeg([
        "-i", video_path,
        "-i", audio_path,
        "-filter_complex",
        f"[0:a]volume=1.0[a1];[1:a]volume={music_volume:.2f}[a2];[a1][a2]amix=inputs=2:duration=first[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        output_path
    ])

def replace_audio(video_path: str, audio_path: str, output_path: str) -> tuple:
    """Replace video audio with a new audio file."""
    return _run_ffmpeg([
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-c:a", "aac",
        "-shortest",
        output_path
    ])

# ─────────────────────────────────────────────────────────────────────────────
#  TEXT / WATERMARK OVERLAY
# ─────────────────────────────────────────────────────────────────────────────
def add_text_overlay(input_path: str, output_path: str,
                     text: str, position: str = "bottom",
                     font_size: int = 48, colour: str = "white",
                     bg_box: bool = True) -> tuple:
    """Burn text overlay into video using ffmpeg drawtext."""
    pos_map = {
        "top":    "x=(w-text_w)/2:y=30",
        "center": "x=(w-text_w)/2:y=(h-text_h)/2",
        "bottom": "x=(w-text_w)/2:y=h-text_h-40",
        "top-left": "x=20:y=20",
        "top-right":"x=w-text_w-20:y=20",
    }
    xy  = pos_map.get(position, pos_map["bottom"])
    box = ":box=1:boxcolor=black@0.5:boxborderw=8" if bg_box else ""
    safe_text = text.replace("'", "'\\''")
    vf = (f"drawtext=text='{safe_text}':fontsize={font_size}:"
          f"fontcolor={colour}:{xy}{box}:line_spacing=8")
    return _run_ffmpeg([
        "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "20", "-c:a", "copy",
        output_path
    ])

def add_watermark_image(video_path: str, watermark_path: str,
                         output_path: str, position: str = "bottom-right",
                         opacity: float = 0.7, scale: float = 0.15) -> tuple:
    """Add an image watermark to video."""
    pos_map = {
        "top-left":     "10:10",
        "top-right":    "main_w-overlay_w-10:10",
        "bottom-left":  "10:main_h-overlay_h-10",
        "bottom-right": "main_w-overlay_w-10:main_h-overlay_h-10",
        "center":       "(main_w-overlay_w)/2:(main_h-overlay_h)/2",
    }
    xy = pos_map.get(position, pos_map["bottom-right"])
    return _run_ffmpeg([
        "-i", video_path,
        "-i", watermark_path,
        "-filter_complex",
        f"[1:v]format=rgba,colorchannelmixer=aa={opacity:.2f},scale=iw*{scale}:-1[wm];"
        f"[0:v][wm]overlay={xy}[out]",
        "-map", "[out]", "-map", "0:a?",
        "-c:v", "libx264", "-crf", "20", "-c:a", "copy",
        output_path
    ])

# ─────────────────────────────────────────────────────────────────────────────
#  BACKGROUND CHANGE  (chroma-key / colour replacement)
# ─────────────────────────────────────────────────────────────────────────────
def chroma_key(video_path: str, background_path: str, output_path: str,
               key_colour: str = "green",
               similarity: float = 0.35,
               blend: float = 0.1) -> tuple:
    """
    Green/blue screen removal and background replacement.
    key_colour: 'green' or 'blue'
    """
    color_hex = "0x00ff00" if key_colour == "green" else "0x0000ff"
    return _run_ffmpeg([
        "-i", video_path,
        "-i", background_path,
        "-filter_complex",
        f"[0:v]chromakey={color_hex}:{similarity:.2f}:{blend:.2f}[fg];"
        f"[1:v]scale=iw:ih[bg];"
        f"[bg][fg]overlay[out]",
        "-map", "[out]", "-map", "0:a?",
        "-c:v", "libx264", "-crf", "20", "-c:a", "copy",
        output_path
    ])

def replace_background_blur(input_path: str, output_path: str,
                              blur_strength: int = 20) -> tuple:
    """Blur the background (simple — applies to whole frame then overlays)."""
    # Note: true AI background separation needs a segmentation model.
    # This applies a cinematic blur + slight centre focus vignette.
    return _run_ffmpeg([
        "-i", input_path,
        "-vf", f"gblur=sigma={blur_strength},vignette=PI/4",
        "-c:v", "libx264", "-crf", "20", "-c:a", "copy",
        output_path
    ])

# ─────────────────────────────────────────────────────────────────────────────
#  VIGNETTE / BORDERS / FADE
# ─────────────────────────────────────────────────────────────────────────────
def add_vignette(input_path: str, output_path: str, strength: float = 0.5) -> tuple:
    angle = strength * math.pi / 2
    return _run_ffmpeg([
        "-i", input_path,
        "-vf", f"vignette=angle={angle:.3f}",
        "-c:v", "libx264", "-crf", "20", "-c:a", "copy",
        output_path
    ])

def add_fade(input_path: str, output_path: str,
             fade_in: float = 0.5, fade_out: float = 0.5,
             duration: float = 10.0) -> tuple:
    """Add fade-in and fade-out effects."""
    filters = []
    if fade_in > 0:
        filters.append(f"fade=in:st=0:d={fade_in:.2f}")
    if fade_out > 0 and duration > fade_out:
        filters.append(f"fade=out:st={duration-fade_out:.2f}:d={fade_out:.2f}")
    vf = ",".join(filters) if filters else "null"
    return _run_ffmpeg([
        "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "20", "-c:a", "copy",
        output_path
    ])

# ─────────────────────────────────────────────────────────────────────────────
#  FULL PROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def process_video(
    input_path: str,
    output_path: str,
    # Trim
    trim_start: float = 0.0,
    trim_end: float = 0.0,     # 0 = no trim
    # Adjustments
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    gamma: float = 1.0,        # exposure
    sharpness: float = 0.0,
    # Transforms
    flip_h: bool = False,
    flip_v: bool = False,
    rotate: int = 0,
    # Speed
    speed: float = 1.0,
    # Filters
    colour_filter: str = "None",
    # Audio
    volume: float = 1.0,
    normalize_audio: bool = False,
    noise_reduce: bool = False,
    bass_boost: bool = False,
    mute: bool = False,
    # Text overlay
    text_overlay: str = "",
    text_position: str = "bottom",
    text_size: int = 48,
    text_colour: str = "white",
    # Quality
    output_quality: str = "High",
    output_resolution: str = "Original",
    # Effects
    vignette: bool = False,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
) -> tuple:
    """
    Master pipeline: applies all selected operations in optimal order.
    Uses temp files between each stage.
    Returns (success, output_path, error_message)
    """
    import shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        current = input_path
        step    = 0

        def _next(suffix: str) -> str:
            nonlocal step
            step += 1
            return os.path.join(tmpdir, f"step{step:02d}_{suffix}.mp4")

        info = get_video_info(input_path)
        duration = info.get("duration", 10.0)

        # 1. Trim
        if trim_end > trim_start and trim_end > 0:
            nxt = _next("trim")
            ok, err = trim_video(current, nxt, trim_start, trim_end)
            if ok: current = nxt; duration = trim_end - trim_start

        # 2. Flip / rotate
        if flip_h or flip_v or rotate:
            nxt = _next("flip")
            ok, err = flip_rotate(current, nxt, flip_h, flip_v, rotate)
            if ok: current = nxt

        # 3. Colour adjustments
        needs_adjust = (brightness != 0 or contrast != 1.0 or
                        saturation != 1.0 or gamma != 1.0 or sharpness != 0)
        if needs_adjust:
            nxt = _next("adj")
            ok, err = adjust_video(current, nxt, brightness, contrast,
                                    saturation, gamma, sharpness)
            if ok: current = nxt

        # 4. Colour filter
        if colour_filter and colour_filter != "None":
            nxt = _next("filter")
            ok, err = apply_filter(current, nxt, colour_filter)
            if ok: current = nxt

        # 5. Speed
        if speed != 1.0:
            nxt = _next("speed")
            ok, err = change_speed(current, nxt, speed)
            if ok: current = nxt

        # 6. Vignette
        if vignette:
            nxt = _next("vignette")
            ok, err = add_vignette(current, nxt)
            if ok: current = nxt

        # 7. Fade
        if fade_in > 0 or fade_out > 0:
            nxt = _next("fade")
            ok, err = add_fade(current, nxt, fade_in, fade_out, duration)
            if ok: current = nxt

        # 8. Text overlay
        if text_overlay and text_overlay.strip():
            nxt = _next("text")
            ok, err = add_text_overlay(current, nxt, text_overlay,
                                        text_position, text_size, text_colour)
            if ok: current = nxt

        # 9. Audio
        if mute:
            nxt = _next("audio")
            ok, err = mute_audio(current, nxt)
            if ok: current = nxt
        elif volume != 1.0 or normalize_audio or noise_reduce or bass_boost:
            nxt = _next("audio")
            ok, err = audio_enhance(current, nxt, volume, noise_reduce,
                                     normalize_audio, bass_boost)
            if ok: current = nxt

        # 10. Resolution
        if output_resolution and output_resolution != "Original":
            nxt = _next("res")
            ok, err = change_resolution(current, nxt, output_resolution)
            if ok: current = nxt

        # 11. Final quality encode
        nxt = _next("final")
        ok, err = change_quality(current, nxt, output_quality)
        if ok: current = nxt

        # Copy to output
        try:
            shutil.copy2(current, output_path)
            return True, output_path, ""
        except Exception as e:
            return False, "", str(e)


# ─────────────────────────────────────────────────────────────────────────────
#  GENERATE VIDEO FROM IMAGES (slideshow)
# ─────────────────────────────────────────────────────────────────────────────
def images_to_video(images: list, output_path: str,
                    fps: int = 24, duration_per_image: float = 3.0) -> tuple:
    """Create a video slideshow from a list of PIL Images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_paths = []
        for i, img in enumerate(images):
            # Write each image multiple times = duration * fps
            frame = img.convert("RGB")
            n_frames = max(1, int(duration_per_image * fps))
            for j in range(n_frames):
                path = os.path.join(tmpdir, f"frame_{i:04d}_{j:04d}.jpg")
                frame.save(path, "JPEG", quality=95)
                frame_paths.append(path)

        # Write list file
        list_path = os.path.join(tmpdir, "frames.txt")
        with open(list_path, "w") as f:
            for fp in frame_paths:
                f.write(f"file '{fp}'\nduration {1.0/fps:.4f}\n")

        ok, err = _run_ffmpeg([
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-vf", f"fps={fps},scale=1080:1080:force_original_aspect_ratio=decrease,pad=1080:1080:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264", "-crf", "20", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            output_path
        ])
        return ok, err
