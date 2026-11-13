import json,tarfile,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SRC=Path('/public/0_DATA/2_Avatar/realtalk');catalog=json.loads((SRC/'preview/catalog.json').read_text());videos={v['id']:v for v in catalog['videos']};index=[]
for shard in sorted((SRC/'videos').glob('videos_*.tar')):
 with tarfile.open(shard,'r:') as archive:
  for member in archive:
   vid=Path(member.name).stem
   if member.isfile() and member.name.endswith('.avi') and vid in videos:
    index.append(dict(video_id=vid,archive=str(shard),member=member.name,offset=member.offset_data,size=member.size,source=f'subfile,,start,{member.offset_data},end,{member.offset_data+member.size},,:{shard}',annotation=str(ROOT/'annotations'/(vid+'.json')),asr=str(SRC/'asr'/(vid+'.json'))))
lookup={v['video_id']:v for v in index}
with tarfile.open(SRC/'annotations.tar.gz','r|gz') as archive:
 for member in archive:
  vid=Path(member.name).stem
  if member.isfile() and vid in lookup and member.name.endswith('.json'):
   data=archive.extractfile(member).read();dest=ROOT/'annotations'/(vid+'.json');dest.write_bytes(data);lookup[vid]['annotation_sha256']=hashlib.sha256(data).hexdigest()
index.sort(key=lambda x:x['video_id']);assert len(index)==len(videos)==692;assert all(Path(r['annotation']).exists() for r in index)
(ROOT/'index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n');print('indexed',len(index),flush=True)
