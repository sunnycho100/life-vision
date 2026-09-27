"""Browser UI contract checks with explicit route fixtures; NOT an ML validation.

Uses the running server's real reference video for browser decoding. All job,
track, event, and review API responses below are deterministic test fixtures.
"""
import argparse
import asyncio
import copy
import json
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]


async def main(base):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(channel="chrome", headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 1000})
        errors, builds, track_requests, job_posts = [], [], [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.add_init_script("""window.alertPlays=0;
          window.AudioContext=class {state='running';currentTime=0;destination={};
            async resume(){} createOscillator(){return {frequency:{},connect(){},start(){window.alertPlays++},stop(){}}}
            createGain(){return {connect(){},gain:{setValueAtTime(){},exponentialRampToValueAtTime(){}}}}};""")
        state = {"analysis": "not_started", "polls": 0, "fail_save": True}
        job = {"id": "fixture-review-job", "state": "completed", "start": 8, "end": 90,
               "progress": 1, "frames_analyzed": 410, "backend": "UI fixture"}
        incident = {"id": "fixture-event-1", "track_id": 1, "kind": "visibility_lost",
                    "reason": "visibility_lost_in_pool", "start_time": 8, "trigger_time": 20,
                    "last_seen": 8, "last_position": [.4, .5], "severity": "urgent", "status": "pending",
                    "evidence": "visibility-only", "clip": {"state": "ready", "start": 8, "end": 25},
                    "review": {"decision": "needs_review", "note": "", "audit": []}}
        events = [incident]

        async def route_api(route):
            request = route.request
            path = urlparse(request.url).path
            if path == "/api/jobs":
                job_posts.append(request.post_data_json)
                return await route.fulfill(status=202, json=job)
            if path.endswith("/review-analysis"):
                if request.method == "POST":
                    builds.append(request.post_data_json)
                    state["analysis"] = "running"
                elif state["analysis"] == "running":
                    state["polls"] += 1
                    if state["polls"] >= 2:
                        state["analysis"] = "completed"
                result = {"state": state["analysis"], "progress": .5 if state["analysis"] == "running" else 1}
                if builds:
                    result["config"] = builds[0]
                return await route.fulfill(json=result)
            if path.endswith("/observations"):
                query = parse_qs(urlparse(request.url).query)
                start, end = float(query["start"][0]), float(query["end"][0])
                return await route.fulfill(json={"observations": [{"media_time": t / 10, "status": "analyzed",
                    "detections": [{"bbox_xyxy_normalized": [.3,.3,.4,.5], "confidence": .9, "class_name": "person"}]}
                    for t in range(int(start * 10), int(end * 10))]})
            if path.endswith("/tracks"):
                query = parse_qs(urlparse(request.url).query)
                start, end = float(query["start"][0]), float(query["end"][0])
                track_requests.append((start, end))
                frames = []
                for tick in range(int(start * 10), int(end * 10)):
                    time = tick / 10
                    if time < 8 or time >= 90:
                        continue
                    tracks = [{"track_id": i + 1, "bbox_xyxy_normalized": [.3,.3,.4,.5], "confidence": .9,
                               "state": "visible", "last_seen": time, "missing_seconds": 0,
                               "review_severity": "none", "reason": "observed"} for i in range(150)]
                    quality = "ok"
                    if 9 <= time < 10:
                        tracks = [{**tracks[0], "state": "missing", "last_seen": 8, "missing_seconds": 12,
                                   "review_severity": "urgent", "reason": "visibility_lost_in_pool"}]
                    if 10 <= time < 11:
                        quality = "degraded"
                    if 11 <= time < 12:
                        quality = "unavailable"
                    if 12 <= time < 13:
                        tracks = []
                    frames.append({"media_time": time, "quality": quality, "tracks": tracks})
                return await route.fulfill(json={"state": "completed", "frames": frames})
            if path.endswith("/incidents"):
                return await route.fulfill(json={"state": state["analysis"], "incidents": events})
            if "/incidents/" in path and request.method == "PATCH":
                if state["fail_save"]:
                    state["fail_save"] = False
                    return await route.fulfill(status=500, json={"detail": "fixture save failure"})
                event = next(i for i in events if i["id"] == path.rsplit("/", 1)[1])
                event["review"].update(request.post_data_json)
                return await route.fulfill(json=event)
            if path.endswith("/clip"):
                return await route.fulfill(path=str(ROOT / "artifacts/browser-vfr.mp4"), content_type="video/mp4")
            return await route.continue_()

        await page.route("**/api/jobs**", route_api)
        await page.goto(base)
        await page.wait_for_function("!document.querySelector('#reference-button').disabled")
        await page.click("#reference-button")
        await page.wait_for_function("!document.querySelector('#calibrate-panel').hidden")
        await page.fill("#analysis-start", "8")
        await page.fill("#analysis-end", "90")
        await page.click("#scan-button")
        await page.wait_for_function("!document.querySelector('#twin-panel').hidden")
        assert not builds, "Review must require explicit user action"
        await page.wait_for_function("!document.querySelector('#build-review').disabled")
        await page.click("#build-review")
        await page.wait_for_function("document.querySelector('#review-analysis-status').textContent.includes('running')")
        await page.wait_for_function("document.querySelector('#review-analysis-status').textContent.includes('completed')", timeout=10000)
        assert len(builds) == 1 and builds[0]["start"] >= 8 and builds[0]["end"] == 90
        assert builds[0]["low_threshold"] == .1 and builds[0]["urgent_seconds"] == 12
        await page.evaluate("video.currentTime=8.1")
        await page.wait_for_function("overlay.dataset.boxCount==='150'")
        assert await page.locator(".track-chip").count() == 150
        await page.evaluate("video.currentTime=9.1")
        await page.wait_for_function("overlay.dataset.boxCount==='0' && document.querySelector('#track-states').textContent.includes('1 lost')")
        assert "12.0s missing" in await page.locator(".track-chip").text_content()
        assert await page.locator(".track-chip").get_attribute("data-severity") == "urgent"
        for time in [10.1, 11.1, 100]:
            await page.evaluate("time=>video.currentTime=time", time)
            await page.wait_for_function("document.querySelector('#person-count').textContent==='—'")
        await page.evaluate("video.currentTime=12.1")
        await page.wait_for_function("document.querySelector('#person-count').textContent==='0'")
        for time in [8.1, 9.1, 8.1]:
            await page.evaluate("time=>video.currentTime=time", time)
            await page.wait_for_timeout(600)
        assert len(builds) == 1 and len(job_posts) == 1, "Seeking must not rerun analysis"
        assert all(end - start <= 30 for start, end in track_requests)
        assert await page.evaluate("alertPlays") == 0
        await page.click(".incident-card")
        await page.wait_for_function("document.querySelector('#incident-video').readyState>=2")
        await page.locator("#incident-video").evaluate("v=>v.play()")
        assert await page.locator("#incident-video").evaluate("v=>!v.paused")
        async with page.expect_download():
            await page.click("#download-clip")
        await page.fill("#incident-note", "Fixture reviewer checked source context")
        await page.click('[data-decision="confirmed_concern"]')
        await page.wait_for_function("document.querySelector('#review-save-status').textContent.includes('fixture save failure')")
        await page.click('[data-decision="confirmed_concern"]')
        await page.wait_for_function("document.querySelector('#review-save-status').textContent==='Review saved.'")
        assert incident["review"]["decision"] == "confirmed_concern"
        assert incident["review"]["note"] == "Fixture reviewer checked source context"
        await page.select_option("#incident-filter", "confirmed_concern")
        assert await page.locator(".incident-card").count() == 1
        await page.click("#enable-alerts")
        assert await page.evaluate("alertPlays") == 0, "Enabling alerts must not synthesize an event"
        added = copy.deepcopy(incident)
        added["id"] = "fixture-event-2"
        added["review"]["decision"] = "needs_review"
        events.append(added)
        await page.wait_for_function("alertPlays===1", timeout=7000)
        await page.wait_for_timeout(2500)
        assert await page.evaluate("alertPlays") == 1, "Existing pending events must not repeat alerts"
        await page.click("#seek-incident")
        await page.wait_for_function("Math.abs(video.currentTime-8)<.01 && !video.seeking")
        await page.set_viewport_size({"width": 390, "height": 844})
        assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert not errors, errors
        report = {"status": "passed", "evidence": "Explicit API route fixtures, not real ML inference",
                  "checks": ["explicit build", "progress", "150 tracked boxes", "lost timer and severity", "unavailable/degraded gaps",
                             "empty frame", "seek without rerun", "saved clip playback/download", "review failure/retry/note",
                             "queue filtering", "sound consent and event deduplication", "mobile width"], "page_errors": errors}
        (ROOT / "artifacts/review-ui-fixture-results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5173")
    asyncio.run(main(parser.parse_args().url))
