"""Real Chrome acceptance tests. Uses real server inference; fixtures only for >100/empty/gap checks."""
import argparse
import asyncio
import json
from pathlib import Path
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"


async def main(base):
    OUT.mkdir(exist_ok=True)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(channel="chrome", headless=True, args=["--enable-unsafe-swiftshader"])
        page = await browser.new_page(viewport={"width": 1440, "height": 1080})
        errors, requests, jobs = [], [], []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("request", lambda r: requests.append(r.url))
        page.on("request", lambda r: jobs.append(r.url) if r.method == "POST" and r.url.endswith("/api/jobs") else None)
        await page.goto(base)
        await page.wait_for_function("!document.querySelector('#reference-button').disabled")
        unit = await page.evaluate("""async () => {
          const c = await import('./core.mjs');
          const assert = (v, m) => {if (!v) throw Error(m);};
          const q = [[.1,.1],[.9,.2],[.8,.9],[.2,.8]];
          const dst = [[0,0],[20,0],[20,30],[0,30]], h=c.homography(q,dst);
          assert(c.validQuad(q), 'valid outline');
          q.forEach((p,i)=>assert(Math.hypot(...c.project(h,...p).map((v,j)=>v-dst[i][j]))<1e-6,'homography'));
          assert(!c.validQuad([q[0],q[2],q[1],q[3]]),'crossed outline');
          const frames=[{media_time:1,status:'analyzed',detections:[]},{media_time:2,status:'decode_error',detections:[]}];
          assert(c.observationAt(frames,1.2)?.detections.length===0,'analyzed empty');
          assert(c.observationAt(frames,.999)===null && c.observationAt(frames,1.31)===null && c.observationAt(frames,2)===null,'future, stale, failed hidden');
          const cache = new c.ObservationCache(async start=>[{media_time:start,status:'analyzed',detections:[]}]);
          for(let t=0;t<100;t+=10) await cache.ensure(t,true);
          assert(cache.windows.size===4,'cache bound');
          let finish; const stale=new c.ObservationCache(()=>new Promise(r=>finish=r));
          const pending=stale.ensure(0); stale.clear(); finish([]); await pending;
          assert(stale.windows.size===0,'old source response discarded');
          return 'geometry, observation timing, bounded cache and source isolation passed';
        }""")
        print(unit, flush=True)
        await page.click("#reference-button")
        await page.wait_for_function("!document.querySelector('#calibrate-panel').hidden")
        await page.fill("#water-height", "9")
        await page.click("#scan-button")
        assert "water height" in await page.locator("#message").text_content()
        await page.fill("#water-height", "1.1")
        await page.locator('.more-settings summary').click()
        await page.fill("#analysis-start", "8")
        await page.fill("#analysis-end", "10")
        async with page.expect_response(lambda r: r.url.endswith('/api/jobs') and r.request.method == 'POST') as submitted:
            await page.click("#scan-button")
        submission = await submitted.value
        assert submission.status == 202, await submission.text()
        await page.wait_for_function("!document.querySelector('#twin-panel').hidden", timeout=90000)
        # Wait for openReview's initial seek before testing a user seek.
        await page.wait_for_function("!video.seeking && video.currentTime > 8.008 && document.querySelector('#person-count').textContent !== '\\u2014'")
        # The merged tracker confirms identities after consecutive observations.
        await page.evaluate("video.currentTime=8.25")
        await page.wait_for_function("Number(document.querySelector('#person-count').textContent)>0", timeout=15000)
        assert await page.locator("#stage").get_attribute("data-view") == "video"
        assert await page.evaluate("video.playbackRate") == 1
        metrics = await page.evaluate("""({
          detections:Number(document.querySelector('#person-count').textContent),
          inference_ms:Number(document.querySelector('#inference-time').textContent),
          runtime:document.querySelector('#delegate').textContent,
          video_time:video.currentTime,
          boxes:Number(document.querySelector('#overlay').dataset.boxCount)
        })""")
        assert metrics["boxes"] == metrics["detections"]
        await page.screenshot(path=str(OUT / "rfdetr-person-detection.png"), full_page=True)
        before = len(jobs)
        for time in [9.5, 8.5, 9.1]:
            await page.evaluate("(t)=>video.currentTime=t", time)
            await page.wait_for_function("!video.seeking && Number(document.querySelector('#person-count').textContent)>0")
        assert len(jobs) == before, "Seeking started another job"
        await page.set_viewport_size({"width": 1100, "height": 900})
        await page.wait_for_timeout(100)
        assert await page.evaluate("Number(overlay.dataset.boxCount)>0")
        # Old JSON accepts boxes even with unusable former identity/landmark data.
        source = await page.evaluate("async()=>await (await fetch('/api/reference')).json()")
        people = [{"track_id": "ignored", "keypoints": "ignored", "bbox_xyxy": [.35,.35,.4,.5], "confidence": .9} for _ in range(150)]
        fixture = {"schema_version": 1, "video": {"sha256": source["sha256"], "width": source["width"], "height": source["height"], "duration_sec": source["duration"]},
                   "frames": [{"t_sec": 22, "people": people}, {"t_sec": 23, "people": []}, {"t_sec": 96, "people": people}]}
        await page.locator("#tracking-file").set_input_files({"name": "box-fixture.json", "mimeType": "application/json", "buffer": json.dumps(fixture).encode()})
        await page.evaluate("video.currentTime=22")
        await page.wait_for_function("document.querySelector('#person-count').textContent==='150' && overlay.dataset.boxCount==='150'")
        await page.evaluate("video.currentTime=23")
        await page.wait_for_function("document.querySelector('#person-count').textContent==='0'")
        await page.evaluate("video.currentTime=23.5")
        await page.wait_for_function("document.querySelector('#person-count').textContent==='\u2014'")
        await page.evaluate("video.currentTime=96")
        await page.wait_for_function("document.querySelector('#person-count').textContent==='150'")
        assert await page.locator("#pool-count").text_content() == "\u2014", "Pool count must expire while detections continue"
        await page.click('button[data-view="split"]')
        await page.wait_for_function("document.querySelector('#stage').dataset.view==='split'")
        assert await page.locator("#zoom-warning").is_visible()
        await page.click('button[data-view="video"]')
        assert await page.locator("#zoom-warning").is_hidden()
        # Compare actual browser presentation timestamps against decoder PTS.
        timing = []
        for target in [8.008,22.0053166667,45.0116333333,70.0032666667,90.0065833333]:
            actual = await page.evaluate("""t=>new Promise((resolve,reject)=>{
              const timeout=setTimeout(()=>reject(Error('No presented frame')),5000);
              video.requestVideoFrameCallback((_,m)=>{clearTimeout(timeout);resolve(m.mediaTime);});
              video.currentTime=t+0.001;
            })""", target)
            assert abs(actual-target) <= .5/59.94 + 1e-5, (target, actual)
            timing.append({"decoder_time": target, "browser_time": actual})
        mobile = await browser.new_page(viewport={"width": 390, "height": 844})
        await mobile.goto(base)
        await mobile.wait_for_function("!document.querySelector('#reference-button').disabled")
        assert await mobile.evaluate("document.documentElement.scrollWidth <= innerWidth")
        await mobile.screenshot(path=str(OUT / "rfdetr-mobile.png"), full_page=True)
        annotation = await browser.new_page(viewport={"width": 1440, "height": 1000})
        annotation.on("pageerror", lambda e: errors.append(str(e)))
        await annotation.goto(base + "/annotate.html")
        await annotation.locator("#folder").set_input_files(str(OUT / "evaluation/reference"))
        await annotation.wait_for_function("document.querySelector('#canvas').width===1280")
        bounds = await annotation.locator("#canvas").bounding_box()
        await annotation.mouse.move(bounds["x"] + 200, bounds["y"] + 200)
        await annotation.mouse.down()
        await annotation.mouse.move(bounds["x"] + 240, bounds["y"] + 260)
        await annotation.mouse.up()
        assert await annotation.locator("#person option").count() == 1
        await annotation.locator("#reviewed").check()
        async with annotation.expect_download() as download_info:
            await annotation.click("#save")
        download = await download_info.value
        exported = json.loads(Path(await download.path()).read_text())
        assert exported["frames"][0]["reviewed"] and len(exported["frames"][0]["people"]) == 1
        # Do not save this synthetic browser-test label into the actual evaluation bundle.
        import av
        import numpy as np
        from fractions import Fraction
        vfr_path = OUT / "browser-vfr.mp4"
        with av.open(str(vfr_path), 'w') as container:
            stream = container.add_stream('libx264', rate=25)
            stream.width, stream.height, stream.pix_fmt = 160, 96, 'yuv420p'
            stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
            stream.options = {'bf': '0'}
            for pts in [0, 33, 100, 250, 460, 500, 710, 840]:
                frame = av.VideoFrame.from_ndarray(np.zeros((96,160,3), dtype=np.uint8), format='rgb24')
                frame.pts, frame.time_base = pts, Fraction(1,1000)
                for packet in stream.encode(frame): container.mux(packet)
            for packet in stream.encode(): container.mux(packet)
        vfr_page = await browser.new_page()
        await vfr_page.goto(base)
        await vfr_page.locator('#video-file').set_input_files(str(vfr_path))
        await vfr_page.wait_for_function("!document.querySelector('#calibrate-panel').hidden")
        await vfr_page.click('#scan-button')
        await vfr_page.wait_for_function("!document.querySelector('#twin-panel').hidden", timeout=30000)
        await vfr_page.wait_for_function("document.querySelector('#person-count').textContent==='0'")
        vfr_timing = []
        for target in [.25, .46, .71, .84]:
            actual = await vfr_page.evaluate("""t=>new Promise((resolve,reject)=>{
              const timeout=setTimeout(()=>reject(Error('No VFR presented frame')),5000);
              video.requestVideoFrameCallback((_,m)=>{clearTimeout(timeout);resolve(m.mediaTime);});
              video.currentTime=t+0.001;
            })""", target)
            assert abs(actual-target)<.001, (target, actual)
            vfr_timing.append({'decoder_time': target, 'browser_time': actual})
        forbidden = [url for url in requests if any(s in url.lower() for s in ["pose-worker", "yolov8n-pose", "mediapipe", "landmarker", "vision_bundle"])]
        assert not forbidden, forbidden
        assert not errors, errors
        report = {"status": "passed", "real_rf_detr": metrics, "timing_checks": timing, "vfr_timing_checks": vfr_timing,
                  "synthetic_checks": ["150 boxes", "empty vs unavailable", "legacy import", "pool boundary"],
                  "page_errors": errors, "pose_network_requests": forbidden}
        (OUT / "rfdetr-browser-results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5173")
    asyncio.run(main(parser.parse_args().url))

