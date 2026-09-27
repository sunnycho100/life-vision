# Milestones (12-hour build)

updated: 2026-09-26 · H0 = kickoff. Check-ins at H1, H3, H6, H9, H11.

Rule: build against **fake data first**. Nobody waits for someone else's part. Contracts are in [architecture.md](architecture.md#data-contracts).

## Everyone, H0 to H1
- [ ] Read [architecture.md](architecture.md) and [decisions.md](decisions.md). Object now or never.
- [ ] Freeze contracts A (`tracks.json`) and B (`results.json`).
- [ ] Pick 4 demo clips: public pool videos plus MuJoCo renders. Record which license each has.
- [ ] Write one hand-made `results.json` for one clip, so David can start.

## Joanne: detector
| By | Milestone |
|---|---|
| H1 | Pretrained YOLO11n or YOLO26n running on one pool clip, boxes saved |
| H3 | Person detection checked on all 4 demo clips (misses, false boxes, speed on a MacBook). Head evidence source chosen with Sam (head detector or pose keypoints). |
| H6 | `detect(frame) -> people with bbox, conf, head, head_conf` used by Sunny's pipeline |
| H9 | Optional: short fine-tune on the Mibugi dataset (splits by source video) only if swimmers are missed. Otherwise tune thresholds. |
| H11 | Numbers for the pitch: detection recall on our clips, FPS |

## Sam: pose, evaluation, data
| By | Milestone |
|---|---|
| H1 | Test case list and clip list in `data/` (with licenses) |
| H3 | Pretrained YOLO-Pose on the demo clips. Report whether head keypoints are reliable enough to be the head signal. |
| H6 | Optional pose features per track: arm repetition, forward progress. Written into `keypoints` in contract A. |
| H9 | `model/eval/` script: time to alarm, false alarms per hour, missed events, merge success. Baseline = plain "person missing" timer. |
| H11 | Results table with counts for the pitch, plus one failure case |

## Sunny: pipeline, tracker, sim
| By | Milestone |
|---|---|
| H1 | YOLO run on a MuJoCo render: does it detect capsule people? Decide sim's role. |
| H3 | Sim writes a ground-truth `tracks.json` (contract A) from `pool_scene.py`, projected into the camera. Handed to Rohan. |
| H6 | `backend/pipeline/`: FileSource → Joanne's detector → ByteTrack → `tracks.json` for any video |
| H9 | ByteTrack vs BoT-SORT on the demo clips (ID switches). One command runs the whole chain: video → tracks → results. |
| H11 | All 4 demo videos processed, results files in the frontend |

## Rohan: event engine
| By | Milestone |
|---|---|
| H1 | Engine skeleton reading contract A and the pool polygon, writing contract B |
| H3 | Person states and head timer working on a hand-written `tracks.json` |
| H6 | Missing registry, merge rule with the outside-entry safeguard, still-person rule. Tested on Sunny's sim tracks. |
| H9 | Entry alert (armed), visibility warning, optional distress yellow from Sam's features. Thresholds in a config file with a demo flag. |
| H11 | Engine run on all 4 demo videos, events checked by eye |

## David: frontend
| By | Milestone |
|---|---|
| H1 | One page: 4 video feeds playing |
| H3 | Canvas overlay drawing green, yellow, red, and dashed missing boxes from the hand-made `results.json`, aligned at any window size |
| H6 | Timers, reasons, alert panel, audio alert with an "enable audio" click, "DEMO THRESHOLDS" badge |
| H9 | Activation animation (time-boxed), pool polygon click tool saving `pool.json`, notification settings, confirm and false-alarm buttons |
| H11 | Incident replay, real results files loaded, screen recording of the demo as a backup |

## Integration
| By | Check |
|---|---|
| H6 | Sim tracks → engine → frontend shows a box turning red. First end-to-end run. |
| H9 | Real clip → detector → tracker → engine → frontend |
| H11 | Scope frozen. Only fixes after this. |
| H12 | Rehearsed demo, backup recording, limits slide |
