"""Re-render selected intervals from originals; preserve pairing and boundaries."""
import concurrent.futures
import json
from build_seamless_listening_preview import OUT, dump, render_face_media


def process(pair):
    sources={s['id']:s for s in pair['sources']}
    for c in pair['clips']:
        d=OUT/c['path']
        existing=json.loads((d/'metadata.json').read_text())
        if existing.get('face_resolution')!=512:
            render_face_media(sources[c['speaker']],sources[c['listener']],
                              [c['crop_boxes'][role] for role in ['speaker','listener']],
                              c['start'],c['end'],d)
        c.update(face_resolution=512,media_revision=3)
        dump(d/'metadata.json',c)
    pair['version']=3
    dump(OUT/'media'/pair['pair']/'pair.json',pair)
    print('DONE',pair['pair'],len(pair['clips']),flush=True)
    return pair


if __name__=='__main__':
    manifest=json.loads((OUT/'manifest.json').read_text())
    before=[(p['pair'],c['speaker'],c['listener'],c['start'],c['end']) for p in manifest['pairs'] for c in p['clips']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        manifest['pairs']=list(pool.map(process,manifest['pairs']))
    after=[(p['pair'],c['speaker'],c['listener'],c['start'],c['end']) for p in manifest['pairs'] for c in p['clips']]
    assert before==after
    manifest['media_revision']=3
    manifest['segmentation_revision']=1
    dump(OUT/'manifest.json',manifest)
    print('FINISHED',len(manifest['pairs']),'pairs',len(after),'clips; original boundaries preserved',flush=True)
