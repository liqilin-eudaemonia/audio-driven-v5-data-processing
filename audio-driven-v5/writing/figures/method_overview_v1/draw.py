"""Editable vector draft of the current listening AV model; no real result media."""
from pathlib import Path
from html import escape
import math
import re
import cairosvg
from PIL import ImageFont

OUT = Path(__file__).resolve().parent
W, H = 1600, 1000
C = dict(ink='#263340', muted='#637181', border='#CBD3DC', blue='#EAF1F9',
         blue_edge='#5879A2', purple='#F0ECF8', purple_edge='#8170A3',
         orange='#FFF1DE', orange_edge='#C18438', teal='#E5F3EF',
         teal_edge='#408873', coral='#FBECE8', coral_edge='#B96A59',
         gray='#F5F6F8', white='#FFFFFF')
parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
 '<title id="title">Audio-conditioned listening backchannel generation — method draft</title>',
 '<desc id="desc">Current speaker audio and listener first frame condition joint audiovisual generation. Feature flow-matching supervises Connector conditions. Clip event BCE supervises final LTX hidden states. Auxiliary branches are training only.</desc>',
 '<defs>']
for key in ['ink','purple_edge','orange_edge','teal_edge','coral_edge','muted']:
    parts.append(f'<marker id="arrow-{key}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 1 1 L 9 5 L 1 9 z" fill="{C[key]}"/></marker>')
parts += ['</defs>',f'<rect width="{W}" height="{H}" fill="white"/>',
          '<g font-family="DejaVu Sans, Arial, sans-serif">']

def rect(x,y,w,h,fill='white',stroke='border',r=9,dash=False,sw=1.8):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{C.get(fill,fill)}" stroke="{C.get(stroke,stroke)}" stroke-width="{sw}"'+(' stroke-dasharray="7 5"' if dash else '')+'/>')

def text(x,y,s,size=24,color='ink',weight=400,anchor='start'):
    # Independent text runs keep SVG / PDF / PNG math placement consistent.
    pattern=re.compile(r'\b([UaIVFZL])_(feat|Event|AV|[vals])(?:\^([va]))?\b')
    runs=[];pos=0
    for m in pattern.finditer(s):
        if m.start()>pos: runs.append((s[pos:m.start()],size,0))
        runs.append((m[1],size,0))
        runs.append((m[2],size*.72,size*.23))
        if m[3]: runs.append((m[3],size*.72,-size*.36))
        pos=m.end()
    if pos<len(s): runs.append((s[pos:],size,0))
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if weight>=600 else '')+'.ttf'
    widths=[ImageFont.truetype(font,round(z)).getlength(t) for t,z,dy in runs]
    xx=x-(sum(widths)/2 if anchor=='middle' else sum(widths) if anchor=='end' else 0)
    for (t,z,dy),width in zip(runs,widths):
        parts.append(f'<text xml:space="preserve" x="{xx:.2f}" y="{y+dy:.2f}" font-size="{z:.2f}" fill="{C.get(color,color)}" font-weight="{weight}">{escape(t)}</text>')
        xx+=width

def lines(x,y,ss,size=24,gap=29,color='ink',weight=400,anchor='middle'):
    for i,s in enumerate(ss): text(x,y+i*gap,s,size,color,weight,anchor)

def path(d,color='ink',width=2.4,dash=False,arrow=True):
    parts.append(f'<path d="{d}" fill="none" stroke="{C.get(color,color)}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"'+(' stroke-dasharray="7 5"' if dash else '')+(f' marker-end="url(#arrow-{color})"' if arrow else '')+'/>')

def dot(x,y,color='ink',r=3.6):
    parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{C.get(color,color)}"/>')

def badge(x,y,n,color):
    parts.append(f'<circle cx="{x}" cy="{y}" r="16" fill="{C[color]}"/>')
    text(x,y+7,str(n),21,'white',700,'middle')

def lock(x,y):
    path(f'M{x+3} {y+9} v-4 a5 5 0 0 1 10 0 v4',color='muted',width=1.8,arrow=False)
    rect(x,y+8,16,12,fill='#FFFFFF',stroke='muted',r=2,sw=1.6)

def box(x,y,w,h,title,sub=None,fill='gray',edge='border',frozen=False,ts=24):
    rect(x,y,w,h,fill,edge)
    def fit(s,size,bold=False):
        font='/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf'
        while ImageFont.truetype(font,size).getlength(s)>w-22 and size>16: size-=1
        return size
    ts=fit(title,ts,True)
    text(x+w/2,y+(h/2+8 if sub is None else h/2-4),title,ts,weight=600,anchor='middle')
    if sub: text(x+w/2,y+h/2+25,sub,fit(sub,20),'muted',anchor='middle')
    if frozen: lock(x+w-19,y-10)
    if title in ['Audio adapter','Connector','Speaker cross-attention','Feature decoders','Event heads']:
        dot(x+w-10,y+10,r=4)

def wave(x,y,w,h,kind='speaker',color='orange_edge'):
    path(f'M{x} {y} H{x+w}',color='muted',width=1.1,arrow=False)
    for i in range(94):
        t=i/93
        if kind=='speaker': env=sum(a*math.exp(-((t-m)/s)**2) for m,s,a in [(0.13,.075,.8),(.35,.10,1),(.62,.08,.75),(.84,.08,.85)])
        else: env=.9*math.exp(-((t-.30)/.062)**2)+.68*math.exp(-((t-.72)/.04)**2)
        amp=h*(.12+.88*abs(math.sin(i*1.93)))*env
        path(f'M{x+t*w:.2f} {y-amp:.2f} V{y+amp:.2f}',color=color,width=2,arrow=False)

def portrait(x,y,w,h,tilt=0,smile=False):
    rect(x,y,w,h,fill='#F0F3F5',stroke='border',r=5,sw=1.3)
    cx,cy=x+w/2,y+h*.48
    parts.append(f'<g transform="rotate({tilt} {cx} {cy})">')
    parts.append(f'<path d="M{x+w*.15} {y+h} Q{cx} {y+h*.49} {x+w*.85} {y+h}" fill="#8FB3AD"/>')
    parts.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{w*.23}" ry="{h*.29}" fill="#E3C8AE"/>')
    parts.append(f'<path d="M{cx-w*.23} {cy} Q{cx-w*.30} {y+h*.12} {cx} {y+h*.14} Q{cx+w*.31} {y+h*.15} {cx+w*.24} {cy-h*.025} Q{cx+w*.10} {cy-h*.17} {cx-w*.07} {cy-h*.14} Q{cx-w*.17} {cy-h*.07} {cx-w*.23} {cy}" fill="#5E5753"/>')
    for ex in [-.09,.09]: dot(cx+w*ex,cy,'ink',r=1.6)
    path(f'M{cx} {cy+5} l-2 9 h4',width=1.1,color='muted',arrow=False)
    if smile: path(f'M{cx-w*.085} {cy+h*.16} Q{cx} {cy+h*.23} {cx+w*.085} {cy+h*.16}',width=1.6,color='ink',arrow=False)
    else: path(f'M{cx-w*.065} {cy+h*.17} h{w*.13}',width=1.4,color='ink',arrow=False)
    parts.append('</g>')

# Panel (a): solid paths used by the generation pipeline.
text(24,38,'(a)  Audio-conditioned joint generation',29,weight=700)
lock(1218,18);text(1244,37,'Frozen',20,'muted')
dot(1362,28,r=5);text(1382,37,'Trainable',20,'muted')

text(105,166,'Speaker audio',21,weight=600,anchor='middle')
wave(30,212,150,28)
text(105,267,'a_s',23,'muted',anchor='middle')
portrait(61,320,88,106)
text(105,459,'Listener',22,weight=600,anchor='middle')
text(105,490,'First frame I_l',21,'muted',anchor='middle')

# Direct speaker audio route, highlight 1.
box(250,81,200,80,'Whisper','speaker encoder',frozen=True)
box(500,81,205,80,'Audio adapter','LN + projection',fill='orange',edge='orange_edge')
badge(515,81,1,'orange_edge')
path('M185 212 H215 V121 H250','orange_edge',2.7)
path('M450 121 H500','orange_edge',2.7)
path('M705 121 H1040 V206','orange_edge',2.7)
text(862,104,'Direct acoustic features',20,'orange_edge',anchor='middle')

# High-level conditions.
box(250,209,200,91,'Qwen2.5-Omni','guided encoding',fill='purple',edge='purple_edge',frozen=True,ts=22)
box(500,209,205,91,'Connector','A/V conditions',fill='teal',edge='teal_edge')
box(747,223,99,63,'U_v, U_a',ts=21,fill='teal',edge='teal_edge')
badge(797,223,2,'teal_edge')
path('M185 212 H228 V238 H250','purple_edge')
path('M149 373 H229 V274 H250','purple_edge')
path('M450 254 H500','purple_edge')
text(475,237,'H',20,'purple_edge',anchor='middle')
path('M705 254 H747','teal_edge')
path('M846 254 H870 V324 H910','teal_edge')

# Description origin is explicit; Qwen prediction is a preceding inference step.
rect(250,357,455,106,'purple','purple_edge')
text(477,387,'Response description d',24,weight=600,anchor='middle')
text(477,416,'Train: annotated description',20,'muted',anchor='middle')
text(477,444,'Test: predicted by Qwen from (a_s, I_l)',19,'muted',anchor='middle')
path('M350 357 V300','purple_edge')
box(747,370,118,77,'Gemma',fill='gray',frozen=True,ts=21)
path('M705 410 H747','purple_edge')
path('M865 410 H888 V370 H910','purple_edge')

# LTX container, explicitly trainable adapters within a frozen base.
rect(894,179,342,302,'blue','blue_edge',r=12,sw=2.2)
text(1065,198,'LTX · repeated blocks',18,'blue_edge',weight=600,anchor='middle')
box(910,206,310,63,'Speaker cross-attention','Video / audio branches',fill='orange',edge='orange_edge',ts=22)
box(910,307,310,89,'Joint A/V denoising','Frozen base + adapters / LoRA',fill='white',edge='blue_edge',ts=24)
path('M1065 270 V307','orange_edge')
rect(932,421,266,43,'coral','coral_edge',r=6)
text(1065,449,'Final hidden  Z_v, Z_a',22,weight=600,anchor='middle')
badge(1198,421,3,'coral_edge')
path('M1065 396 V421',width=2)

# Native reference condition, separate from conditioning through Qwen.
dot(197,373,'purple_edge')
path('M197 373 V518 H987 V481','purple_edge')
text(587,509,'First-frame condition',21,'purple_edge',anchor='middle')
text(1182,115,'A/V noise',21,'muted',anchor='middle')
path('M1182 129 V179',width=2)

# Generated outputs (schematic media, not experimental evidence).
path('M1236 349 H1266 V280 H1292',width=2.5)
path('M1266 349 V416 H1292',width=2.5)
dot(1266,349)
text(1430,222,'Listener video  V_l',24,weight=600,anchor='middle')
for i,(tilt,smile) in enumerate([(0,False),(5,True),(-3,True)]): portrait(1295+i*94,240,82,100,tilt,smile)
text(1430,369,'Listener audio  a_l',24,weight=600,anchor='middle')
wave(1295,416,270,27,'listener','teal_edge')
text(1430,473,'Joint audiovisual output',22,'muted',anchor='middle')
text(1430,502,'Media shown schematically',18,'muted',anchor='middle')

# Two training-only detail panels. Repeated U/Z labels avoid long crossing arrows.
rect(24,559,758,339,'white','border',r=10,dash=True)
rect(808,559,768,339,'white','border',r=10,dash=True)
badge(51,592,2,'teal_edge');text(80,602,'(b)  Feature supervision',27,weight=700)
text(754,601,'Training only',19,'muted',anchor='end')
badge(835,592,3,'coral_edge');text(864,602,'(c)  Clip-level event supervision',26,weight=700)
text(1548,631,'Training only',19,'muted',anchor='end')

# Feature reconstruction: two modalities, FM objective, teacher is frozen.
box(48,649,126,64,'U_v, U_a',fill='teal',edge='teal_edge',ts=22)
box(220,637,250,88,'Feature decoders','Flow matching (A/V)',fill='teal',edge='teal_edge',ts=23)
path('M174 681 H220','teal_edge')
box(48,764,126,70,'GT A/V',sub=None,ts=23)
box(220,755,250,87,'Qwen A/V teachers','Frozen A/V towers',frozen=True,ts=23)
path('M174 799 H220',dash=True)
box(556,692,195,100,'Feature loss','L_feat^v + L_feat^a',fill='teal',edge='teal_edge',ts=24)
path('M470 681 H519 V721 H556','teal_edge')
path('M470 799 H519 V766 H556','teal_edge',dash=True)
text(517,826,'F_v, F_a',20,'teal_edge',anchor='middle')
text(49,877,'Supervises the condition mapping; Qwen remains frozen.',22,'muted')

# LTX final hidden -> valid clip pooling -> modality-specific event heads.
box(835,658,180,69,'Z_v, Z_a','Last LTX block',fill='coral',edge='coral_edge',ts=24)
box(1058,658,228,69,'Clip pooling','Per modality · valid tokens',fill='coral',edge='coral_edge',ts=24)
box(1328,658,220,69,'Event heads','Video / audio',fill='coral',edge='coral_edge',ts=24)
path('M1015 692 H1058','coral_edge')
path('M1286 692 H1328','coral_edge')
box(835,778,286,65,'Clip labels + masks',ts=23)
box(1234,778,314,65,'Event loss  L_Event','Masked, weighted BCE',fill='coral',edge='coral_edge',ts=24)
path('M1121 810 H1234','coral_edge',dash=True)
path('M1438 727 V778','coral_edge')
text(835,757,'Reference frame and padding excluded',20,'muted')
text(835,877,'Category presence, without timestamp targets.',22,'muted')

# Compact objective and path legend, outside the panels.
text(24,948,'L = L_AV + 0.05 L_Event + 0.025 L_feat^v + 0.025 L_feat^a',26,weight=500)
path('M1195 938 H1242',width=2.1);text(1254,945,'Model flow',19,'muted')
path('M1195 973 H1242',width=2.1,dash=True);text(1254,980,'Supervision targets',19,'muted')
parts += ['</g>','</svg>']
svg='\n'.join(parts)
(OUT/'method_overview_v1.svg').write_text(svg)
cairosvg.svg2png(bytestring=svg.encode(),write_to=str(OUT/'method_overview_v1.png'),output_width=2400,output_height=1500)
# Exactly 7 inches wide, with vector paths and embedded/subset text fonts.
cairosvg.svg2pdf(bytestring=svg.encode(),write_to=str(OUT/'method_overview_v1.pdf'),output_width=672,output_height=420)
print('Created SVG, PNG (2400 × 1500), and vector PDF (7 × 4.375 in).')
