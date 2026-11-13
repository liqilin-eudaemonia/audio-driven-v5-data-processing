"""Publish the next independent audio-first review batch."""
import concurrent.futures,json,shutil
from pathlib import Path
import publish_listen_v8_recovery as exporter
from prepare_listen_v8_audio_first import ROOT,OUT,PREVIEW,FEEDBACK
from p3_listen.common import read,write,sha

def main():
    exporter.OUT=PREVIEW;exporter.exporter.OUT=PREVIEW
    selection=read(OUT/'review_selection.json');write(PREVIEW/'selection.json',selection)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        rows=list(pool.map(exporter.export,selection['samples']))
    write(PREVIEW/'samples.json',dict(samples=rows,selection_sha256=sha(PREVIEW/'selection.json'),
        summary=read(OUT/'summary.json'),training_admitted=False))
    shutil.copyfile(FEEDBACK,PREVIEW/'previous_review_feedback.json')
    shutil.copyfile(ROOT/'scripts/listen_v8_audio_first_review.html',PREVIEW/'index.html')
    from fontTools import subset
    from fontTools.ttLib import TTFont
    source=Path('/tmp/v8_browser_libs/extracted/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if source.exists():
        font=TTFont(source,fontNumber=2);sub=subset.Subsetter()
        sub.populate(text=(PREVIEW/'index.html').read_text()+(PREVIEW/'samples.json').read_text());sub.subset(font)
        font.flavor='woff';font.save(PREVIEW/'review.woff')
    link=ROOT/'data_previews/seamless_face_listening_v1/listen_v8_audio_first'
    if not link.exists():link.symlink_to(PREVIEW,target_is_directory=True)
    print(json.dumps(dict(samples=len(rows),url='http://127.0.0.1:8875/listen_v8_audio_first/')),flush=True)

if __name__=='__main__':main()
