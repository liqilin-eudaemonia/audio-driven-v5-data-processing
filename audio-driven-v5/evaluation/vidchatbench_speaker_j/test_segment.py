"""词时间裁剪：零时长丢掉，超出本条 wav 的词不进入 J。"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import segment


def test_drop_zero_and_clip_tail(tmp_path):
    path = tmp_path / 'human.json'
    path.write_text(json.dumps(dict(text='hello there later', words=[
        dict(text=' hello', timestamp=[0.0, 0.4]),
        dict(text=' there', timestamp=[0.4, 0.4]),
        dict(text=' later', timestamp=[1.2, 1.8]),
        dict(text=' out', timestamp=[2.2, 2.6]),
    ])))
    doc, dropped = segment.local_word_document(path, 1.5)
    words = doc['timestamps']['words']
    assert dropped == 2
    assert [(row['text'].strip(), row['start_time'], row['end_time']) for row in words] == [
        ('hello', 0.0, 0.4),
        ('later', 1.2, 1.5),
    ]


def test_official_segment_past_media(tmp_path):
    path = tmp_path / 'official_asr.json'
    path.write_text(json.dumps(dict(text='inside outside', segments=[
        dict(start=0.0, end=1.0, text=' inside'),
        dict(start=1.0, end=29.0, text=' outside'),
    ])))
    block = segment.official_blocks(path, 4.0)
    assert block['segments'][0]['past_media'] is False
    assert block['segments'][1]['past_media'] is True
    assert block['text'] == 'inside outside'


if __name__ == '__main__':
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        test_drop_zero_and_clip_tail(root)
        test_official_segment_past_media(root)
    print('ok')
