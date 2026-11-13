"""Short, continuously played source excerpts with visible omission annotations."""
import json
from pathlib import Path
import subprocess

OUT=Path(__file__).resolve().parent/'review'
FONT='/tmp/v8_browser_libs/extracted/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
TEXT=OUT/'closeup_captions';TEXT.mkdir(exist_ok=True)

def render(number,start,duration,cutoff,title,captions,boxes):
    filters=['pad=iw:ih+160:0:0:color=0x101c2f']
    def caption(key,text,y,size,color='white',enable=None):
        path=TEXT/f'case{number}_{key}.txt';path.write_text(text)
        spec=f'drawtext=fontfile={FONT}:textfile={path}:x=14:y={y}:fontsize={size}:fontcolor={color}'
        if enable:spec+=f":enable='{enable}'"
        filters.append(spec)
    caption('title',title,489,23)
    caption('before','当前 pair 仍在保留范围内',526,23,'0x75e0ac',f'lt(t,{cutoff})')
    caption('after','↑ 当前 pair 已结束；后面仍是连续原视频',526,23,'0xffa493',f'gte(t,{cutoff})')
    for j,(a,b,text) in enumerate(captions):
        caption(f'words{j}',text,565,21,'white',f'between(t,{a},{b})')
    caption('footer','红框 = 被漏掉的发声；画面与声音连续，不跳过问题区间。',607,19,'0xc5d4e8')
    filters.append(f"drawbox=x=2:y=2:w=iw-4:h=476:color=0x75e0ac:t=3:enable='lt(t,{cutoff})'")
    for a,b,side in boxes:
        filters.append(f"drawbox=x={side*360+3}:y=35:w=354:h=406:color=0xff5555:t=5:enable='between(t,{a},{b})'")
    target=OUT/f'case{number}_closeup_annotated.mp4'
    cmd=['ffmpeg','-v','error','-nostdin','-y','-ss',str(start),'-i',str(OUT/f'case{number}_context.mp4'),
         '-t',str(duration),'-vf',','.join(filters),'-filter_threads','1','-c:v','libx264','-preset','fast',
         '-crf','20','-threads','2','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(target)]
    subprocess.run(cmd,check=True)
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(target),'-f','null','-'],check=True)
    poster=OUT/f'case{number}_closeup_poster.jpg'
    subprocess.run(['ffmpeg','-v','error','-y','-ss','0.3','-i',str(target),'-frames:v','1',str(poster)],check=True)
    return dict(file=target.name,poster=poster.name,duration=duration,cutoff_in_excerpt=cutoff,
                omitted_intervals=[dict(start=a,end=b,participant='AB'[side]) for a,b,side in boxes])

if __name__=='__main__':
    one=render(1,10,10,4.661,'例 1：右侧说话者的最后一句被漏掉',
               [(0,4.661,'先听前一句；视频第 4.66 秒处是当前 pair 的结束点。'),
                (4.661,6.9,"遗漏原转写：So I'm looking for a new job right now."),
                (6.9,10,'这整句话没进入任何生产 pair；原视频中仍然存在。')],
               [(4.921,6.642,1)])
    two=render(2,2,8,4.707,'例 2：右侧句尾和左侧短回应一起漏掉',
               [(0,4.707,'右侧说到 “…my car to heat”，当前 pair 就结束了。'),
                (4.707,5.05,'右侧遗漏句尾：up.（原始转写）'),
                (5.05,5.65,'左侧遗漏回应：Oh, yeah.（原始转写）'),
                (5.65,8,'完整内容应包括 heat up. 和随后左侧的短回应。')],
               [(4.787,4.928,1),(5.115,5.355,0)])
    (OUT/'closeups.json').write_text(json.dumps(dict(case1=dict(one,source_start=180,source_end=190),
                                                   case2=dict(two,source_start=331,source_end=339)),indent=2)+'\n')
    print('Two annotated closeups exported and decoded.')
