# Render the synthetic YOLO training set (run from the repo root, takes about 6 minutes on an RTX 4070 Laptop).
#   81 random stills   (--random, new sky every frame)
#   180 scripted frames (3 clips x 60 frames, seeds 1-3: layout, timing, cast, sky and camera differ
#                        from the seed-0 test clip)
$env:OMNI_KIT_ACCEPT_EULA = "YES"
$py = "sim\isaac\.venv\Scripts\python.exe"
$common = @("--width", "1280", "--height", "720", "--pathtrace", "32", "--subframes", "1", "--yolo-every", "1")
& $py sim\isaac\pool_video.py --random --seed 11 --seconds 81 --fps 1 --out sim\isaac\_out_train\r11 @common
foreach ($s in 1, 2, 3) {
    & $py sim\isaac\pool_video.py --seed $s --seconds 20 --fps 3 --out sim\isaac\_out_train\s$s @common
}
