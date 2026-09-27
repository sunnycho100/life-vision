"""Persistent, bounded recorded-video review API and disposable analysis worker."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
import traceback

from fastapi import HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from backend.common import ROOT, code_digest, digest, read_json, write_json
from backend.monitoring import _polygon


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    corners: list[list[float]]
    start: float = Field(default=0, ge=0)
    end: float = Field(gt=0)
    low_threshold: float = Field(default=.1, ge=0, le=1)
    high_threshold: float = Field(default=.25, ge=0, le=1)
    new_track_threshold: float = Field(default=.25, ge=0, le=1)
    warn_seconds: float = Field(default=5, gt=0, le=600)
    urgent_seconds: float = Field(default=12, gt=0, le=600)


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    decision: str = Field(pattern=r'^(confirmed_concern|dismissed|needs_review)$')
    note: str = Field(default='', max_length=10000)


def validate_config(config, spec, source):
    _polygon(config['corners'])
    if not spec['start'] <= config['start'] < config['end'] <= min(spec['end'], source['duration']) + 1e-6:
        raise ValueError('Review interval must be within the completed job and recording.')
    if not config['low_threshold'] <= config['high_threshold'] <= config['new_track_threshold']:
        raise ValueError('Thresholds must satisfy low <= high <= new track.')
    if config['warn_seconds'] >= config['urgent_seconds']:
        raise ValueError('Urgent duration must exceed warning duration.')
    if spec.get('stored_threshold', 1) > config['low_threshold'] + 1e-8:
        raise ValueError('This detection job did not retain the requested low confidence observations. Start a new detection job with the lower storage threshold.')


class ReviewManager:
    def __init__(self, manager, get_source):
        self.manager, self.get_source = manager, get_source
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='review')
        self.process, self.active, self.closed = None, None, False
        for path in manager.root.glob('*/review/status.json'):
            status = read_json(path)
            if status['state'] in {'queued', 'running'}:
                status.update(state='interrupted', updated=time.time(), error='Server stopped before review and clips completed.')
                write_json(path, status)

    def directory(self, job_id):
        try:
            return self.manager.path(job_id) / 'review'
        except KeyError as error:
            raise HTTPException(404, 'Unknown job') from error

    def status(self, job_id):
        path = self.directory(job_id) / 'status.json'
        return read_json(path) if path.is_file() else {'state': 'not_started'}

    def submit(self, job_id, config):
        with self.lock:
            directory = self.directory(job_id)
            job = self.manager.status(job_id)
            if job['state'] != 'completed' or job.get('incomplete', True):
                raise HTTPException(409, 'Review requires a completed detection job.')
            spec = read_json(directory.parent / 'manifest.json')
            source = self.get_source(spec['source_id'])
            try:
                validate_config(config, spec, source)
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
            fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            status = self.status(job_id)
            if status['state'] != 'not_started':
                if status.get('config_hash') != fingerprint:
                    raise HTTPException(409, 'This job already has an immutable review run. Create a new detection job for different review settings.')
                return status
            if self.active or self.closed:
                raise HTTPException(409, 'Another review is running. Wait for completion.')
            directory.mkdir(exist_ok=True)
            status = {'state': 'queued', 'job_id': job_id, 'config': config, 'config_hash': fingerprint,
                      'created': time.time(), 'updated': time.time(), 'progress': 0, 'summary': {}}
            write_json(directory / 'request.json', {'config': config, 'source': source, 'source_sha256': spec['source_sha256']})
            write_json(directory / 'status.json', status)
            self.active = job_id
            self.executor.submit(self._run, job_id, directory)
            return status

    def _run(self, job_id, directory):
        try:
            with (directory / 'worker.log').open('wb') as log:
                with self.lock:
                    if self.closed:
                        return
                    self.process = subprocess.Popen([sys.executable, '-m', 'backend.review', str(directory)], cwd=ROOT,
                        stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                    process = self.process
                code = process.wait()
            status = read_json(directory / 'status.json')
            if status['state'] in {'queued', 'running'}:
                status.update(state='interrupted' if self.closed else 'failed', updated=time.time(), error=f'Review worker exited ({code}); see worker.log.')
                write_json(directory / 'status.json', status)
        except Exception as error:
            status = read_json(directory / 'status.json')
            status.update(state='failed', updated=time.time(), error=str(error))
            write_json(directory / 'status.json', status)
        finally:
            with self.lock:
                self.process, self.active = None, None

    def incidents(self, job_id):
        directory = self.directory(job_id)
        incidents = read_json(directory / 'incidents.json') if (directory / 'incidents.json').exists() else []
        with self.lock:
            decisions = read_json(directory / 'decisions.json') if (directory / 'decisions.json').exists() else {}
        for incident in incidents:
            incident['review'] = decisions.get(incident['id'], {'decision': 'needs_review', 'note': '', 'audit': []})
        return incidents

    def incident(self, job_id, incident_id):
        return next((item for item in self.incidents(job_id) if item['id'] == incident_id), None)

    def decide(self, job_id, incident_id, decision):
        with self.lock:
            if self.incident(job_id, incident_id) is None:
                raise HTTPException(404, 'Unknown incident')
            path = self.directory(job_id) / 'decisions.json'
            decisions = read_json(path) if path.exists() else {}
            previous = decisions.get(incident_id, {'audit': []})
            entry = {**decision, 'updated_at': time.time()}
            decisions[incident_id] = {**entry, 'audit': [*previous['audit'], entry]}
            write_json(path, decisions)
            return self.incident(job_id, incident_id)

    def close(self):
        with self.lock:
            self.closed = True
            process = self.process
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self.executor.shutdown(wait=True, cancel_futures=True)


def install_review_routes(app, manager, get_source, data_root):
    review = ReviewManager(manager, get_source)
    app.state.review = review

    @app.post('/api/jobs/{job_id}/review-analysis', status_code=202)
    def begin(job_id: str, body: ReviewRequest):
        return review.submit(job_id, body.model_dump())

    @app.get('/api/jobs/{job_id}/review-analysis')
    def status(job_id: str):
        return review.status(job_id)

    @app.get('/api/jobs/{job_id}/tracks')
    def tracks(job_id: str, start: float = Query(ge=0), end: float = Query(gt=0)):
        if not math.isfinite(start) or not math.isfinite(end) or not start < end or end - start > 30:
            raise HTTPException(422, 'Request a finite track window of at most 30 seconds.')
        state = review.status(job_id)['state']
        path = review.directory(job_id) / 'tracks.sqlite3'
        if not path.exists():
            return {'state': state, 'frames': []}
        with sqlite3.connect(path, timeout=10) as db:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='frames'").fetchone():
                return {'state': state, 'frames': []}
            rows = db.execute('SELECT payload FROM frames WHERE media_time >= ? AND media_time <= ? ORDER BY media_time', (start, end))
            return {'state': state, 'frames': [json.loads(row[0]) for row in rows]}

    @app.get('/api/jobs/{job_id}/incidents')
    def incidents(job_id: str):
        return {'state': review.status(job_id)['state'], 'incidents': review.incidents(job_id)}

    @app.patch('/api/jobs/{job_id}/incidents/{incident_id}')
    def decide(job_id: str, incident_id: str, body: DecisionRequest):
        return review.decide(job_id, incident_id, body.model_dump())

    @app.get('/api/jobs/{job_id}/incidents/{incident_id}/clip')
    def clip(job_id: str, incident_id: str):
        incident = review.incident(job_id, incident_id)
        if incident is None:
            raise HTTPException(404, 'Unknown incident')
        metadata = incident.get('clip', {})
        # The filename is produced by this worker, never from the URL identifier.
        filename = metadata.get('filename', '')
        path = review.directory(job_id) / 'clips' / filename
        if metadata.get('state') != 'ready' or not filename or Path(filename).name != filename or not path.is_file():
            raise HTTPException(409, 'Clip is not ready.')
        return FileResponse(path, media_type='video/mp4', filename=f'incident-{incident_id}.mp4')

    return review


def run(directory):
    from backend.monitoring import process_observations
    from backend.clip_worker import encode_clip
    directory = Path(directory)
    status = read_json(directory / 'status.json')
    request = read_json(directory / 'request.json')
    config, source = request['config'], request['source']
    try:
        status.update(state='running', updated=time.time(), engine_sha256=code_digest(ROOT / 'backend' / 'monitoring.py'))
        write_json(directory / 'status.json', status)
        if digest(source['path']) != request['source_sha256']:
            raise ValueError('Source file changed after detection analysis.')
        with sqlite3.connect(directory.parent / 'observations.sqlite3') as incoming, sqlite3.connect(directory / 'tracks.sqlite3') as outgoing:
            outgoing.execute('PRAGMA journal_mode=WAL')
            outgoing.execute('CREATE TABLE frames (media_time REAL PRIMARY KEY, payload TEXT NOT NULL)')
            outgoing.commit()
            count = 0
            def sink(frame):
                nonlocal count
                outgoing.execute('INSERT INTO frames VALUES (?, ?)', (frame['media_time'], json.dumps(frame, allow_nan=False)))
                count += 1
                if count % 100 == 0:
                    outgoing.commit()
                    status.update(frames_analyzed=count, progress=min(.9, .9 * (frame['media_time'] - config['start']) / (config['end'] - config['start'])), updated=time.time())
                    write_json(directory / 'status.json', status)
            rows = incoming.execute('SELECT payload FROM observations WHERE media_time >= ? AND media_time <= ? ORDER BY media_time', (config['start'], config['end']))
            result = process_observations((json.loads(row[0]) for row in rows), {**config, 'frame_sink': sink})
            outgoing.commit()
        incidents = result['incidents']
        for index, incident in enumerate(incidents):
            incident['clip'] = {'state': 'pending', 'filename': f'{index:06d}.mp4'}
        write_json(directory / 'incidents.json', incidents)
        status.update(summary=result['summary'], frames_analyzed=count, progress=.9, updated=time.time())
        write_json(directory / 'status.json', status)
        if incidents and digest(source['path']) != request['source_sha256']:
            raise ValueError('Source file changed before clip encoding.')
        (directory / 'clips').mkdir(exist_ok=True)
        for incident in incidents:
            metadata = incident['clip']
            requested_start = float(incident['start_time']) - 10
            requested_end = float(incident.get('end_time') or incident['trigger_time']) + 10
            start = max(0., requested_start)
            end = min(float(source['duration']), requested_end, start + 60)
            metadata.update(start=start, end=end, requested_start=requested_start, requested_end=requested_end,
                            partial=start > requested_start or end < requested_end,
                            capped=end < min(float(source['duration']), requested_end))
            try:
                metadata.update(encode_clip(source['path'], directory / 'clips' / metadata['filename'], start, end))
                metadata['url'] = f"/api/jobs/{status['job_id']}/incidents/{incident['id']}/clip"
            except Exception as error:
                metadata.update(state='failed', error=f'{type(error).__name__}: {error}')
            write_json(directory / 'incidents.json', incidents)
        status.update(state='completed', progress=1, updated=time.time(), clip_failures=sum(i['clip']['state'] == 'failed' for i in incidents))
    except Exception as error:
        traceback.print_exc()
        status.update(state='failed', updated=time.time(), error=f'{type(error).__name__}: {error}')
        if (directory / 'incidents.json').exists():
            incidents = read_json(directory / 'incidents.json')
            for incident in incidents:
                if incident.get('clip', {}).get('state') == 'pending':
                    incident['clip'].update(state='failed', error=status['error'])
            write_json(directory / 'incidents.json', incidents)
    write_json(directory / 'status.json', status)


if __name__ == '__main__':
    run(sys.argv[1])
