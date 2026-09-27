"""Generate CCTV pool surveillance scenario footage with Veo 3.1."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Add FeverCoach scripts path for gemini_client credentials
sys.path.insert(0, "/Users/sunghwan_cho/Documents/likelion/FeverCoach/video-gen/scripts")
from lib.gemini_client import create_client
from google.genai import types

PROMPTS = {
    1: (
        "Authentic commercial outdoor CCTV surveillance camera footage overlooking a commercial swimming pool at midday. "
        "Locked-off fixed wide shot mounted high on an adjacent wall or light pole, angled approximately 35 degrees downward and to the side of the pool basin. "
        "Commercial security camera optics: fixed focal length, subtle wide-angle barrel distortion, natural harsh direct sunlight, realistic water caustics, "
        "faint digital sensor noise, mild H.264 compression artifacts, flat neutral surveillance color profile, no cinematic color grade, no CGI gloss. "
        "Clean raw camera feed only: no timestamp, no on-screen display (OSD), no HUD, no bounding boxes, no text, no visual overlays, no camera motion. "
        "Scene action: The pool water is standing waist-to-belly depth (approximately 3.5 feet / 1.1 meters) throughout the frame. "
        "A group of seven distinct people (diverse adults and teenagers in various colored swimwear) are dispersed in the pool. "
        "Four individuals remain upright in the water with their heads, shoulders, and upper chests clearly visible and stationary/wading above the surface. "
        "Concurrently, two other individuals intentionally duck down and submerge completely beneath the water surface, disappearing beneath the water ripples "
        "with visible submerged silhouettes, while the seventh person holds their breath underwater, then surfaces smoothly to waist height, creating natural water splashes and ripples. "
        "Natural ambient pool splash audio and distant outdoor acoustic reverberation."
    ),
    2: (
        "Real facility CCTV camera footage mounted on an exterior roof bracket, looking down from a high side angle across a shallow recreational swimming pool "
        "under bright, diffuse overcast daylight. Static wide camera perspective with zero tilt or pan. "
        "Industrial surveillance sensor quality: sharp digital depth of field, natural dynamic range, unpolished real-world optical characteristics, "
        "realistic surface reflections, no motion blur, no film grain, no 3D animation look. "
        "Clean raw video feed with no overlays, no graphic UI, no tracking markers, no watermark, no timestamps. "
        "Scene action: The water level consistently hits the bellies of the subjects standing in the pool. "
        "Seven people are scattered across the visible quadrant of the pool. "
        "Three individuals stand waist-deep talking in a loose cluster, upper torsos and heads continuously exposed above the waterline. "
        "Two individuals in the center simultaneously squat down and plunge underwater, fully breaking the surface tension and remaining submerged underwater "
        "for several seconds with optical refraction distorting their forms. Two other individuals move through the water: one floating with only head and shoulders exposed, "
        "the other pushing underwater forward before surfacing upright. Natural outdoor aquatic environmental audio."
    ),
}

NEGATIVE_PROMPT = (
    "text, timestamp, timecode, watermark, captions, UI, bounding boxes, detection boxes, "
    "tracking markers, OSD, split screen, low quality, distortion, CGI, 3D animation, cartoon"
)


def generate_cctv_clip(
    prompt: str,
    out_path: Path,
    duration: int = 8,
    model: str = "veo-3.1-generate-preview",
) -> Path:
    client = create_client()
    config = types.GenerateVideosConfig(
        number_of_videos=1,
        aspect_ratio="16:9",
        duration_seconds=duration,
        negative_prompt=NEGATIVE_PROMPT,
    )

    op = None
    for attempt in range(5):
        try:
            print(f"Submitting Veo request ({duration}s, attempt {attempt + 1})...")
            op = client.models.generate_videos(
                model=model,
                prompt=prompt,
                config=config,
            )
            print(f"Operation started: {op.name}")
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_t = 30 * (attempt + 1)
                print(f"Rate limited (429). Backing off {wait_t}s...")
                time.sleep(wait_t)
            else:
                print(f"API error: {e}. Retrying in 15s...")
                time.sleep(15)

    if not op:
        raise RuntimeError("Failed to launch Veo operation after retries")

    start_t = time.time()
    while True:
        res = client.operations.get(op)
        elapsed = int(time.time() - start_t)
        print(f"  [{elapsed}s elapsed] done={res.done}")
        if res.done:
            break
        time.sleep(12)

    if hasattr(res, "error") and res.error:
        raise RuntimeError(f"Veo generation error: {res.error}")

    response = getattr(res, "response", None) or getattr(res, "result", None) or res
    filtered = getattr(response, "rai_media_filtered_reasons", None)
    if filtered:
        raise RuntimeError(f"Veo safety filter triggered: {filtered}")

    videos = getattr(response, "generated_videos", None)
    if not videos:
        raise RuntimeError("No videos returned from Veo")

    v0 = videos[0]
    video_bytes = None
    video_obj = getattr(v0, "video", None)
    if video_obj and getattr(video_obj, "video_bytes", None):
        video_bytes = video_obj.video_bytes
    elif getattr(v0, "video_bytes", None):
        video_bytes = v0.video_bytes
    elif video_obj and getattr(video_obj, "uri", None):
        print(f"Downloading video from {video_obj.uri}...")
        video_bytes = client.files.download(file=video_obj.uri)

    if not video_bytes:
        raise RuntimeError("Could not retrieve video bytes")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(video_bytes)
    print(f"SUCCESS: Saved video to {out_path} ({len(video_bytes) // 1024} KB)")

    # Extract preview frame
    preview_path = out_path.with_suffix(".jpg")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", "00:00:03", "-i", str(out_path), "-vframes", "1", str(preview_path)],
            check=True,
            capture_output=True,
        )
        print(f"Preview frame extracted: {preview_path}")
    except Exception as e:
        print(f"Warning: could not extract preview frame: {e}")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CCTV pool video using Veo 3.1")
    parser.add_argument("--prompt-id", type=int, choices=[1, 2], help="Prompt ID 1 or 2")
    parser.add_argument("--prompt", type=str, help="Custom prompt string")
    parser.add_argument("--out", type=str, required=True, help="Output MP4 file path")
    parser.add_argument("--duration", type=int, default=8, choices=[4, 6, 8], help="Duration in seconds")
    parser.add_argument("--model", type=str, default="veo-3.1-generate-preview", help="Veo model ID")
    args = parser.parse_args()

    if args.prompt_id:
        prompt_text = PROMPTS[args.prompt_id]
    elif args.prompt:
        prompt_text = args.prompt
    else:
        raise ValueError("Must provide either --prompt-id or --prompt")

    out_path = Path(args.out).resolve()
    generate_cctv_clip(
        prompt=prompt_text,
        out_path=out_path,
        duration=args.duration,
        model=args.model,
    )


if __name__ == "__main__":
    main()
