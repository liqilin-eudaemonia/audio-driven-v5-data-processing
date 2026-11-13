"""Package real waveform vectors with true transparent SVG/PNG backgrounds."""
from pathlib import Path
import json
import shutil
import xml.etree.ElementTree as ET
import zipfile
import cairosvg
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'writing/figures/data/selected_real_cases_20261112'
OUT=BASE/'transparent_waveforms'
GROUPS=['case1','case2','case2_context60s','case2_role_swap/role_swapped_176370_185297','case2_role_swap/later_reply_269053_273876']
NS='http://www.w3.org/2000/svg'
ET.register_namespace('',NS)
ET.register_namespace('xlink','http://www.w3.org/1999/xlink')


def transparent_svg(src,dest):
    tree=ET.parse(src);root=tree.getroot();removed=0
    for child in list(root):
        if (child.tag==f'{{{NS}}}rect' and child.get('x')=='0' and child.get('y')=='0'
                and child.get('width')==root.get('width') and child.get('height')==root.get('height')
                and child.get('fill') in ['#fff','#ffffff','white']):
            root.remove(child);removed+=1
    assert removed==1, str(src)
    dest.parent.mkdir(parents=True,exist_ok=True);tree.write(dest,encoding='unicode',xml_declaration=True)
    # Keep editable vectors and render a true alpha-channel PNG (not white pixels).
    cairosvg.svg2png(url=str(dest),write_to=str(dest.with_suffix('.png')),scale=2)


def main():
    OUT.mkdir(exist_ok=True)
    for group in GROUPS:
        d=BASE/group
        for f in sorted(d.glob('*transparent.svg')):
            dest=OUT/group/f.name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(f,dest);shutil.copyfile(f.with_suffix('.png'),dest.with_suffix('.png'))
    paired={'case2':['listener_before_after','listener_waveform_comparison'],
            'case2_context60s':['both_tracks_highlighted','both_tracks_plain'],
            'case2_role_swap/role_swapped_176370_185297':['both_tracks'],
            'case2_role_swap/later_reply_269053_273876':['both_tracks']}
    for group,names in paired.items():
        for name in names:transparent_svg(BASE/group/(name+'.svg'),OUT/group/(name+'_transparent.svg'))
    audit=[]
    for p in sorted(OUT.rglob('*.png')):
        with Image.open(p) as im:
            assert im.mode=='RGBA' and im.getpixel((0,0))[3]==0
            a=im.getchannel('A');lo,hi=a.getextrema();assert lo==0 and hi>0
            audit.append(dict(file=str(p.relative_to(OUT)),mode=im.mode,size=im.size,corner_alpha=0,alpha_range=[lo,hi]))
    (OUT/'ALPHA_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    (OUT/'README.md').write_text('''# 透明背景音轨图

所有PNG均为真正RGBA透明背景，左上角alpha=0；SVG保留可编辑矢量。不是把背景涂成白色。ALPHA_AUDIT.json记录每张图的透明通道检查。

- case1：S1371的speaker音轨。
- case2：S0557的speaker、listener净化前与净化后独立波形，以及带坐标的对照图。
- case2_context60s：附近一分钟双方独立波形和对齐图。
- case2_role_swap/role_swapped_176370_185297：P2254主讲、P2517倾听的波形。
- case2_role_swap/later_reply_269053_273876：后续双方讲话重叠的波形。

独立长条图不含坐标或文字，方便放进论文流程图；两行对照图保留文字坐标，但画布背景透明。时间、形状、颜色和幅度尺度均不改，只移除白色画布。
''')
    with zipfile.ZipFile(OUT/'all_transparent_waveforms.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file() and p.suffix!='.zip':z.write(p,p.relative_to(OUT))
    # A checkerboard makes the transparency visible in the download page.
    web=ROOT/'data_previews/listen_benchmark_review/data_pipeline_selected2_20261112'
    page=web/'index.html';html=page.read_text()
    if 'id="transparent-waveforms"' not in html:
        section='''<section id="transparent-waveforms"><h2>⑦ 所有音轨图：透明背景版本</h2><p>以下为真正透明背景PNG（RGBA）及可编辑SVG；棋盘格仅用于网页预览，不会写进图片。</p><p><a href="assets/transparent_waveforms/all_transparent_waveforms.zip" download>下载全部透明素材ZIP</a></p>'''
        for r in audit:
            png=r['file'];svg=str(Path(png).with_suffix('.svg'));b='assets/transparent_waveforms/'
            section+=f'<p>{png} · <a href="{b+png}" download>透明PNG</a> · <a href="{b+svg}" download>透明SVG</a></p><div style="background:repeating-conic-gradient(#ddd 0% 25%,#fff 0% 50%) 0/20px 20px;border:1px solid #ddd"><img src="{b+png}"></div>'
        page.write_text(html.replace('</html>',section+'</section></html>'))
    print(json.dumps(dict(output=str(OUT),pngs=len(audit),all_true_alpha=True),ensure_ascii=False))


if __name__=='__main__':main()
