# Full synthetic pipeline, run from the repo root after sim\isaac\make_training_set.ps1:
#   1. render the held-out seed-0 test clip (never used for training) with labels
#   2. fine-tune YOLO11n on the training frames and score it on the test frames
#   3. run detection + tracking + drowning alarms on the test clip and score alarm timing
$ErrorActionPreference = "Continue"
$env:OMNI_KIT_ACCEPT_EULA = "YES"
& sim\isaac\.venv\Scripts\python.exe sim\isaac\pool_video.py --seed 0 --seconds 20 --fps 15 --pathtrace 32 `
    --subframes 1 --yolo-every 1 --out sim\isaac\_out_test0
& model\.venv\Scripts\python.exe model\train_yolo.py
& model\.venv\Scripts\python.exe model\detect_drowning.py --video sim\isaac\_out_test0\pool.mp4 `
    --gt sim\isaac\_out_test0\labels.jsonl
