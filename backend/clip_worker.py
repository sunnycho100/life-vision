"""Encode short, source-timestamped review clips in the disposable review process."""
from fractions import Fraction
from pathlib import Path
import math


def encode_clip(source, destination, start, end):
    import av
    destination = Path(destination)
    temporary = destination.with_suffix('.partial.mp4')
    count, first, last = 0, None, None
    try:
        with av.open(str(source)) as incoming, av.open(str(temporary), 'w', format='mp4', options={'movflags': '+faststart'}) as outgoing:
            video = incoming.streams.video[0]
            origin = video.start_time
            if origin is None:
                initial = next(incoming.decode(video), None)
                if initial is None or initial.pts is None:
                    raise ValueError('Source has no timestamped frames.')
                origin = initial.pts
            rate = video.average_rate or Fraction(25, 1)
            output = outgoing.add_stream('libx264', rate=rate)
            output.width = video.width + video.width % 2
            output.height = video.height + video.height % 2
            output.pix_fmt = 'yuv420p'
            output.time_base = Fraction(1, 90000)
            output.codec_context.time_base = output.time_base
            output.options = {'crf': '23', 'preset': 'veryfast'}
            incoming.seek(max(origin, int(start / float(video.time_base)) + origin), stream=video, backward=True, any_frame=False)
            for frame in incoming.decode(video):
                if frame.pts is None:
                    raise ValueError('Source frame has no timestamp.')
                media_time = float((frame.pts - origin) * video.time_base)
                if media_time < start - 1e-7:
                    continue
                if media_time >= end - 1e-7:
                    break
                first = media_time if first is None else first
                last = media_time
                rgb = frame.to_ndarray(format='rgb24')
                encoded = av.VideoFrame.from_ndarray(rgb, format='rgb24').reformat(width=output.width, height=output.height, format='yuv420p')
                encoded.pts = round((media_time - first) * 90000)
                encoded.time_base = Fraction(1, 90000)
                for packet in output.encode(encoded):
                    outgoing.mux(packet)
                count += 1
            if not count:
                raise ValueError('No decodable frames in clip interval.')
            for packet in output.encode():
                outgoing.mux(packet)
        with av.open(str(temporary)) as check:
            duration = float(check.duration or 0) / av.time_base
            if not math.isfinite(duration) or duration <= 0 or next(check.decode(video=0), None) is None:
                raise ValueError('Encoded clip could not be verified.')
        temporary.replace(destination)
        return {'state': 'ready', 'start': start, 'end': end, 'first_frame_time': first,
                'last_frame_time': last, 'duration': duration, 'frames': count, 'audio': False}
    finally:
        temporary.unlink(missing_ok=True)
