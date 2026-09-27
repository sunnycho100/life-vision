"""Generate 15-second two-part Veo 3.1 pool security camera scenario and stitch with ffmpeg."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Add FeverCoach scripts path for gemini_client
sys.path.insert(0, "/Users/sunghwan_cho/Documents/likelion/FeverCoach/video-gen/scripts")
from lib.gemini_client import create_client
from google.genai import types


def generate_clip(client, prompt: str, duration: int, model: str = "veo-3.1-generate-preview") -> bytes:
    for clip_attempt in range(3):
        config = types.GenerateVideosConfig(
            number_of_videos=1,
            aspect_ratio="16:9",
            duration_seconds=duration,
        )
        op = None
        for attempt in range(5):
            try:
                print(f"Submitting request ({duration}s, attempt {attempt + 1})...")
                op = client.models.generate_videos(
                    model=model,
                    prompt=prompt,
                    config=config,
                )
                print(f"Operation started: {op.name}")
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait_t = 25 * (attempt + 1)
                    print(f"Rate limited (429). Retrying in {wait_t}s...")
                    time.sleep(wait_t)
                else:
                    print(f"Generation error: {e}. Retrying in 15s...")
                    time.sleep(15)

        if not op:
            raise RuntimeError("Failed to initiate Veo video generation after retries")

        start_t = time.time()
        while True:
            res = client.operations.get(op)
            elapsed = int(time.time() - start_t)
            print(f"  [{elapsed}s elapsed] done={res.done}")
            if res.done:
                break
            time.sleep(10)

        if hasattr(res, "error") and res.error:
            print(f"Veo operation error on attempt {clip_attempt + 1}: {res.error}")
            if clip_attempt < 2:
                print("Retrying clip generation in 10s...")
                time.sleep(10)
                continue
            raise RuntimeError(f"Veo operation error: {res.error}")

        response = getattr(res, "response", None) or getattr(res, "result", None) or res
        filtered = getattr(response, "rai_media_filtered_reasons", None)
        if filtered:
            print(f"Veo safety filter triggered on attempt {clip_attempt + 1}: {filtered}")
            if clip_attempt < 2:
                print("Retrying clip generation in 10s...")
                time.sleep(10)
                continue
            raise RuntimeError(f"Veo safety filter triggered: {filtered}")

        videos = getattr(response, "generated_videos", None)
        if not videos:
            raise RuntimeError("No videos returned from Veo")

        v0 = videos[0]
        video_obj = getattr(v0, "video", None)
        if video_obj and getattr(video_obj, "video_bytes", None):
            return video_obj.video_bytes
        if getattr(v0, "video_bytes", None):
            return v0.video_bytes
        if video_obj and getattr(video_obj, "uri", None):
            print(f"Downloading video from {video_obj.uri}...")
            return client.files.download(file=video_obj.uri)

    raise RuntimeError("Could not extract video bytes from Veo response")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variation", type=int, required=True, help="Variation number 1, 2, or 3")
    parser.add_argument("--out", type=str, required=True, help="Output MP4 file path")
    parser.add_argument("--model", type=str, default="veo-3.1-generate-preview", help="Veo model ID")
    args = parser.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = out_path.parent / f"tmp_var_{args.variation}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    client = create_client()

    # Variation styling
    var_styles = {
        1: {
            "cam": "Locked-off wide shot from a slightly high porch angle, looking down. Cheap consumer security-camera look: mild wide-angle distortion, natural auto-exposure, light compression, faint digital noise, real sun on real water, no cinematic grade.",
            "scene_1": "Photoreal Ring doorbell / patio camera footage of a real suburban backyard pool at midday. {cam} Clean camera video feed only, no bounding boxes, no computer vision overlays, no text, no UI. Scene action: An adult in a white shirt and khaki shorts walks along the poolside deck from left to right and exits the frame completely. Three young toddlers are playing on the shallow pool steps in colorful swimsuits with small float toys. Natural pool audio only.",
            "scene_2": "Photoreal Ring doorbell / patio camera footage of the same suburban backyard pool at midday. {cam} Clean camera video feed only, no bounding boxes, no computer vision overlays, no text, no UI. Scene action: Continuous locked-off take. The adult is completely gone from the frame. Three young toddlers play on the shallow entry steps. One toddler in a bright swimsuit accidentally slips and falls forward into the shallow standing-depth water, creating vigorous splashes while keeping head safely above water. Natural pool audio only."
        },
        2: {
            "cam": "Locked-off overhead wide patio camera angle, corner fence mount overlooking rectangular pool. Ring consumer security camera aesthetic: slight barrel distortion, crisp midday sun glare on water ripples, natural color balance, subtle digital noise, no color grading, no animation.",
            "scene_1": "Photoreal consumer backyard security camera footage of a pool at midday. {cam} Clean camera video feed only, no bounding boxes, no text overlays, no UI. Scene action: An adult in a light gray t-shirt walks along the perimeter stone deck towards the house doorway and exits the view. Three young toddlers are safely sitting and paddling on the shallow sun-shelf ledge of the pool with floating rings. Natural pool water audio.",
            "scene_2": "Photoreal consumer backyard security camera footage of the same pool at midday. {cam} Clean camera video feed only, no bounding boxes, no text overlays, no UI. Scene action: Continuous stationary camera angle. The adult is entirely out of view. Three toddlers are on the sun shelf ledge. One toddler loses balance and tumbles off the step into shallow standing-depth water with active splashing, head remains above water. Natural splashing audio only."
        },
        3: {
            "cam": "Locked-off high pergola mounted security camera feed overlooking backyard pool and lawn. Ring consumer security camera aesthetic: 16:9 wide perspective, natural exposure adjustments, authentic pool water reflections, light compression artifacts, no CGI gloss.",
            "scene_1": "Photoreal consumer backyard security camera footage of a pool at midday. {cam} Clean camera video feed only, no bounding boxes, no computer vision overlays, no captions, no UI. Scene action: An adult in a light polo shirt walks along the perimeter pool deck and exits the view. Three young toddlers are safely sitting and paddling on the shallow pool steps with floating rings. Natural pool audio only.",
            "scene_2": "Photoreal consumer backyard security camera footage of the same pool at midday. {cam} Clean camera video feed only, no bounding boxes, no computer vision overlays, no captions, no UI. Scene action: Continuous stationary locked-off camera view. The adult is entirely out of view. Three young toddlers are on the shallow steps. One toddler in a bright swimsuit accidentally slips into shallow standing-depth water, creating vigorous splashes while keeping head safely above water as companions watch. Natural pool audio only."
        }
    }

    style = var_styles.get(args.variation, var_styles[1])
    cam = style["cam"]
    p1 = style["scene_1"].replace("{cam}", cam)
    p2 = style["scene_2"].replace("{cam}", cam)

    clip1_file = temp_dir / "part1.mp4"
    if clip1_file.exists() and clip1_file.stat().st_size > 0:
        print(f"Part 1 already exists ({clip1_file.stat().st_size} bytes): {clip1_file}")
    else:
        print("Generating Part 1 (8 seconds)...")
        clip1_bytes = generate_clip(client, p1, duration=8, model=args.model)
        clip1_file.write_bytes(clip1_bytes)
        print(f"Saved Part 1 ({len(clip1_bytes)} bytes): {clip1_file}")

    clip2_file = temp_dir / "part2.mp4"
    if clip2_file.exists() and clip2_file.stat().st_size > 0:
        print(f"Part 2 already exists ({clip2_file.stat().st_size} bytes): {clip2_file}")
    else:
        print("Generating Part 2 (8 seconds)...")
        clip2_bytes = generate_clip(client, p2, duration=8, model=args.model)
        clip2_file.write_bytes(clip2_bytes)
        print(f"Saved Part 2 ({len(clip2_bytes)} bytes): {clip2_file}")

    # Concatenate using ffmpeg concat demuxer (trim to 15 seconds)
    concat_list = temp_dir / "concat.txt"
    concat_list.write_text(f"file '{clip1_file}'\nfile '{clip2_file}'\n")

    print(f"Stitching Part 1 and Part 2 into {out_path}...")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-r", "24",
        "-t", "15",
        str(out_path)
    ]
    subprocess.run(cmd, check=True)
    print(f"SUCCESS: Generated 15s video at {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
