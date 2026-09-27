# backend

- `pipeline/` (Sunny, sunnycho100): video source, detector wrapper, tracker. Writes `tracks.json` (contract A).
- `events/` (Rohan, rsusarla3): event engine. Reads `tracks.json` and `<video>.pool.json`, writes `results.json` (contract B).

Contracts: [docs/global/architecture.md](../docs/global/architecture.md#data-contracts).
