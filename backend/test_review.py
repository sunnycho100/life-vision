"""Real persisted API/worker/MP4 integration, using synthetic person observations."""
from contextlib import contextmanager
import json
import time

import av
from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest

from backend.common import digest, probe, read_json, write_json
from backend.jobs import JobManager, observation_db
from backend.review import install_review_routes, run


CONFIG = {'corners': [[.1, .1], [.9, .1], [.9, .9], [.1, .9]], 'start': 0, 'end': 8}
JOB_ID = 'a' * 32


def tiny_video(path):
    with av.open(str(path), 'w') as out:
        stream = out.add_stream('libx264', rate=5)
        stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
        for i in range(40):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 5, dtype=np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


@contextmanager
def fixture_api(tmp_path, state='completed', stored_threshold=.1, changed=False):
    video = tmp_path / 'source.mp4'
    if not video.exists():
        tiny_video(video)
    source = {**probe(video), 'id': digest(video), 'sha256': digest(video), 'path': str(video)}
    manager = JobManager(tmp_path / 'jobs', tmp_path / 'absent-model.json')
    directory = manager.root / JOB_ID
    directory.mkdir(exist_ok=True)
    write_json(directory / 'status.json', {'id': JOB_ID, 'state': state, 'incomplete': state != 'completed'})
    write_json(directory / 'manifest.json', {'source_id': source['id'], 'source_sha256': '0' * 64 if changed else source['sha256'], 'start': 0, 'end': 8, 'stored_threshold': stored_threshold})
    db = observation_db(directory / 'observations.sqlite3')
    for i in range(40):
        observation = {'media_time': i / 5, 'status': 'analyzed', 'detections': ([{'confidence': .9, 'bbox_xyxy_normalized': [.4, .3, .5, .5]}] if i < 5 else [])}
        db.execute('INSERT OR REPLACE INTO observations VALUES (?, ?)', (i / 5, json.dumps(observation)))
    db.commit()
    db.close()
    app = FastAPI()
    review = install_review_routes(app, manager, lambda _: source, tmp_path)
    try:
        with TestClient(app) as client:
            yield client, directory
    finally:
        review.close()
        manager.close()


def wait_done(client):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = client.get(f'/api/jobs/{JOB_ID}/review-analysis').json()
        if status['state'] not in {'queued', 'running'}:
            return status
        time.sleep(.05)
    pytest.fail('Review worker did not complete in 30 seconds.')


def test_review_real_clip_and_persisted_decisions(tmp_path):
    prefix = f'/api/jobs/{JOB_ID}'
    with fixture_api(tmp_path) as (client, directory):
        assert client.get(prefix + '/review-analysis').json()['state'] == 'not_started'
        assert client.post(prefix + '/review-analysis', json=CONFIG).status_code == 202
        status = wait_done(client)
        assert status['state'] == 'completed', status
        assert status['summary']['frames'] == 40
        assert status['clip_failures'] == 0
        frames = client.get(prefix + '/tracks?start=0&end=8').json()['frames']
        assert len(frames) == 40
        assert frames[-1]['tracks'][0]['state'] == 'missing'
        assert client.get(prefix + '/tracks?start=0&end=31').status_code == 422
        incidents = client.get(prefix + '/incidents').json()['incidents']
        assert len(incidents) == 1
        incident = incidents[0]
        assert incident['kind'] == 'visibility_lost'
        assert incident['clip']['state'] == 'ready'
        assert incident['clip']['partial']
        response = client.get(incident['clip']['url'])
        assert response.status_code == 200 and response.headers['content-type'] == 'video/mp4'
        downloaded = tmp_path / 'downloaded.mp4'
        downloaded.write_bytes(response.content)
        with av.open(str(downloaded)) as clip:
            assert 7.8 <= clip.duration / av.time_base <= 8.2
            assert len(list(clip.decode(video=0))) == 40
        route = prefix + '/incidents/' + incident['id']
        for decision in ['confirmed_concern', 'dismissed']:
            result = client.patch(route, json={'decision': decision, 'note': 'Synthetic test review'})
            assert result.status_code == 200
        assert len(result.json()['review']['audit']) == 2
        assert client.patch(route, json={'decision': 'diagnosis'}).status_code == 422
        assert client.get(prefix + '/incidents/unknown/clip').status_code == 404
        assert client.post(prefix + '/review-analysis', json=CONFIG).json()['state'] == 'completed'
        assert client.post(prefix + '/review-analysis', json={**CONFIG, 'warn_seconds': 4}).status_code == 409
    with fixture_api(tmp_path) as (client, directory):
        review = client.get(prefix + '/incidents').json()['incidents'][0]['review']
        assert review['decision'] == 'dismissed' and len(review['audit']) == 2


@pytest.mark.parametrize('state,stored_threshold,config', [
    ('running', .1, CONFIG),
    ('completed', .2, CONFIG),
    ('completed', .1, {**CONFIG, 'corners': [[0, 0], [1, 1], [0, 1], [1, 0]]}),
    ('completed', .1, {**CONFIG, 'end': 9}),
    ('completed', .1, {**CONFIG, 'high_threshold': .05}),
])
def test_review_rejects_invalid_evidence_and_config(tmp_path, state, stored_threshold, config):
    with fixture_api(tmp_path, state, stored_threshold) as (client, _):
        assert client.post(f'/api/jobs/{JOB_ID}/review-analysis', json=config).status_code in {409, 422}


def test_review_changed_source_has_no_ready_clip(tmp_path):
    with fixture_api(tmp_path, changed=True) as (client, _):
        assert client.post(f'/api/jobs/{JOB_ID}/review-analysis', json=CONFIG).status_code == 202
        status = wait_done(client)
        assert status['state'] == 'failed'
        assert 'Source file changed' in status['error']
        assert client.get(f'/api/jobs/{JOB_ID}/incidents').json()['incidents'] == []


def test_failed_encoding_is_never_downloadable(tmp_path, monkeypatch):
    from backend.review import ReviewRequest
    def fail_encoding(*args, **kwargs):
        raise RuntimeError('Synthetic encoder failure')
    monkeypatch.setattr('backend.clip_worker.encode_clip', fail_encoding)
    with fixture_api(tmp_path) as (client, directory):
        review = directory / 'review'
        review.mkdir()
        source = {**probe(tmp_path / 'source.mp4'), 'path': str(tmp_path / 'source.mp4')}
        write_json(review / 'request.json', {'config': ReviewRequest(**CONFIG).model_dump(), 'source': source, 'source_sha256': digest(source['path'])})
        write_json(review / 'status.json', {'job_id': JOB_ID, 'state': 'queued'})
        run(review)
        status = client.get(f'/api/jobs/{JOB_ID}/review-analysis').json()
        assert status['state'] == 'completed' and status['clip_failures'] == 1
        incident = client.get(f'/api/jobs/{JOB_ID}/incidents').json()['incidents'][0]
        assert incident['clip']['state'] == 'failed'
        assert 'Synthetic encoder failure' in incident['clip']['error']
        assert client.get(f"/api/jobs/{JOB_ID}/incidents/{incident['id']}/clip").status_code == 409
