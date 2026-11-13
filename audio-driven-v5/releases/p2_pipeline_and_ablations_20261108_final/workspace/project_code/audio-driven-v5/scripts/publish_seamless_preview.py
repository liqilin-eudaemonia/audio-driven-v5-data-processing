import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root/'data_previews/seamless_face_listening_v1'
manifest = json.loads((out/'manifest.json').read_text())
template = (root/'scripts/seamless_preview_template.html').read_text()
(out/'index.html').write_text(template.replace('__MANIFEST__', json.dumps(manifest, ensure_ascii=False).replace('<', '\\u003c')))
print('Published', len(manifest['pairs']), 'pairs', sum(len(p['clips']) for p in manifest['pairs']), 'clips')
