import json,subprocess,io
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];chosen=[];used=set()
for lo,hi in [(0,256),(256,320),(320,384),(384,448),(448,512),(512,640),(640,768),(768,1024),(1024,10000)]:
 found=None
 for file in sorted((ROOT/'manifests').glob('*.json')):
  m=json.loads(file.read_text())
  if m['video_id'] in used:continue
  options=[r for r in m['pairs'] if r['crop'] and lo<=r['crop']['native_edge']<hi and r['duration']>=2 and r['face']['valid_frames']/(r['end_frame']-r['start_frame'])>=.95]
  if options:found=(m,options[0]);break
 if found:chosen.append(found);used.add(found[0]['video_id'])
canvas=Image.new('RGB',(1440,960),'#10191b');draw=ImageDraw.Draw(canvas);evidence=[]
for j,(m,r) in enumerate(chosen):
 c=r['crop'];vf=f"drawbox=x={c['x']}:y={c['y']}:w={c['width']}:h={c['height']}:color=lime:t=3,scale=480:270";raw=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-filter_threads','1','-ss',str((r['start']+r['end'])/2),'-i',m['source']['source'],'-frames:v','1','-vf',vf,'-f','image2pipe','-vcodec','mjpeg','-']);im=Image.open(io.BytesIO(raw));x=j%3*480;y=j//3*320;canvas.paste(im,(x,y+35));draw.text((x+5,y+5),f"{m['video_id']} {r['listener']} crop={c['native_edge']}px",fill='white');draw.text((x+5,y+20),f"{r['start']:.2f}-{r['end']:.2f}s source={m['width']}x{m['height']}",fill='white');evidence.append(dict(video_id=m['video_id'],pair=r['pair_id'],crop=c,source_dimensions=[m['width'],m['height']]))
canvas.save(ROOT/'reports/native_crop_contact.jpg');(ROOT/'reports/native_crop_contact.json').write_text(json.dumps(evidence,indent=2));print('contact',len(chosen))
