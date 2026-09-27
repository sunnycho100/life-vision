# Decisions

updated: 2026-09-26 · source: [research comparison](../archive/research-comparison.md) of all five members' docs

Rule: 3 of 5 agree → adopted. Items under "Decided by default" were split; the choice below stands unless someone objects at kickoff.

## Adopted

| # | Decision | Support |
|---|---|---|
| 1 | Detector has one `person` class. Same alarm rule for everyone. | 5 of 5 |
| 2 | Time "head not above water", with "lost inside the pool" as backup. "Person missing" alone is not "underwater". | Sunny, Sam, Rohan, David (UI) |
| 3 | The timer and person IDs live in our event engine, outside the tracker. | Joanne, Sam, Rohan, Sunny |
| 4 | Entry alert as a second feature. | Sunny, Joanne, Sam, Rohan |
| 5 | Pretrained YOLO is the first baseline. Other detectors are compared on the same clips. | Sunny, Sam, Rohan, Joanne |
| 6 | Foundation models (SAM 3, Grounding DINO, VLMs) only for offline labeling. | Sunny, Joanne, Sam |
| 7 | Split train and test by recording session, never by frame. | Sunny, Joanne, Sam, David |
| 8 | Prerecorded video first. Webcam, RTSP, Ring, Nest later. | Sam, Rohan, David |
| 9 | Thresholds are configurable. Demo values are labeled as demo. | 5 of 5 |
| 10 | No long breath holds when recording. Dummy or mannequin for "on the bottom". | Sunny, Joanne, Sam, Rohan, David |
| 11 | Pose and "active distress" are an optional second layer, not the core alarm. | Sunny, Sam, Rohan |
| 12 | Simulation supports the build (demo feeds, engine tests), it is not the critical path. | Sam, Rohan, Joanne, Sunny |
| 13 | Input resolution 960 px. | Sunny, Joanne |

## Decided by default (object at kickoff if you disagree)

| Topic | Decision | Why |
|---|---|---|
| Thresholds | Real: warn 5 s, alarm 12 s, 8 s if still. Demo: 2 / 5 / 4 s. | Rohan's numbers are grounded in MYLO, ASTM F3698-24, and CPSC. 5 s would fire on normal breath holds. |
| Baby vs adult | Not in the 12-hour build. Joanne's "classify on the deck, carry the label into the water" is the plan for after. | Needs consented child footage we don't have. |
| "Child with no adult nearby" alert | Not in this build. Entry alert uses an explicit "armed" switch instead. | Sam: an adult in frame is not proof of supervision. |
| Resurfacing elsewhere | Merge a new track that appears inside the pool near the last position. A track entering from outside never clears a missing person. Ambiguous matches stay open. | 3 votes for merging, plus Sam's safeguard against clearing a real missing person. |
| Tracker | Start with ByteTrack, benchmark BoT-SORT. | Low stakes because of decision 3. |
| YOLO vs RF-DETR | YOLO for the hackathon. RF-DETR (Apache 2.0) if we ever sell. | Licensing only matters for a product. |
| Isaac Sim | Out of scope. | No RTX 4080+ machine. We have MacBooks. |
| Simulation in the build | MuJoCo only: demo feeds and ground-truth tracks. | Fits 12 hours and needs no GPU. |
| Transport to frontend | JSON files synced by timestamp. WebSocket later. | Removes a live server from the demo's failure points. |
| Activation animation | Kept, time-boxed to about 1 hour. | Meeting was split. Keep it but don't let it eat the build. |

## Open (not blocking the build)
- Project name for the pitch (Sam suggests something friendlier).
- Head evidence source: head detector vs pose keypoints. Joanne and Sam pick by hour 3 based on our clips.
- Whether to fine-tune at all. Only if pretrained YOLO misses swimmers on our clips.
