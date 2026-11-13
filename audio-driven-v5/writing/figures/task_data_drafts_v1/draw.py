"""F1/F2 editable, schematic paper figures; no measured model results."""
from pathlib import Path
from html import escape
import math, json
import cairosvg
from PIL import ImageFont
import pymupdf
OUT=Path(__file__).resolve().parent
C={'ink':'#263340','muted':'#637181','border':'#CBD3DC','blue':'#EAF1F9','blue_edge':'#5879A2','purple':'#F0ECF8','purple_edge':'#8170A3','orange':'#FFF1DE','orange_edge':'#C18438','teal':'#E5F3EF','teal_edge':'#408873','coral':'#FBECE8','coral_edge':'#B96A59','gray':'#F5F6F8','white':'#FFFFFF'}
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans'
class Figure:
 def __init__(self,w,h,title):
  self.w,self.h=w,h;self.bounds=[];self.face_id=0
  self.p=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img"><title>{escape(title)}</title><defs>']
  for k in C:
   self.p.append(f'<marker id="a-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1L9 5L1 9Z" fill="{C[k]}"/></marker>')
  self.p+=['</defs>',f'<rect width="{w}" height="{h}" fill="white"/>','<g font-family="DejaVu Sans, Arial, sans-serif">']
 def rect(self,x,y,w,h,fill='white',edge='border',r=10,dash=False,sw=1.8):
  self.p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{C.get(fill,fill)}" stroke="{C.get(edge,edge)}" stroke-width="{sw}"'+(' stroke-dasharray="7 5"' if dash else '')+'/>')
 def text(self,x,y,s,size=23,color='ink',bold=False,anchor='start',maxw=None):
  def font(z):return ImageFont.truetype(FONT+('-Bold' if bold else '')+'.ttf',z)
  if maxw:
   while font(size).getlength(s)>maxw and size>16:size-=1
  wid=font(size).getlength(s);left=x-(wid/2 if anchor=='middle' else wid if anchor=='end' else 0)
  self.bounds.append((s,left,y-size,wid,size))
  self.p.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{C[color]}" font-weight="{700 if bold else 400}">{escape(s)}</text>')
 def lines(self,x,y,ss,size=22,gap=30,**kw):
  for i,s in enumerate(ss):self.text(x,y+i*gap,s,size,**kw)
 def path(self,d,color='ink',sw=2.5,arrow=False,dash=False):
  self.p.append(f'<path d="{d}" fill="none" stroke="{C[color]}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"'+(f' marker-end="url(#a-{color})"' if arrow else '')+(' stroke-dasharray="7 5"' if dash else '')+'/>')
 def badge(self,x,y,n,color='blue_edge'):
  self.p.append(f'<circle cx="{x}" cy="{y}" r="18" fill="{C[color]}"/>');self.text(x,y+7,str(n),22,'white',True,'middle')
 def wave(self,x,y,w,h,kind='speaker',color='orange_edge'):
  self.path(f'M{x} {y}H{x+w}','border',1.3)
  for i in range(160):
   t=i/159
   if kind=='speaker':env=sum(a*math.exp(-((t-m)/s)**2) for m,s,a in [(.13,.075,.8),(.35,.09,1),(.59,.08,.75),(.83,.10,.9)])
   else:env=.65*math.exp(-((t-.37)/.035)**2)+.9*math.exp(-((t-.83)/.065)**2)
   a=h*(.15+.85*abs(math.sin(i*1.93)))*env
   self.path(f'M{x+w*t:.2f} {y-a:.2f}V{y+a:.2f}',color,2.2)
 def face(self,x,y,w,h,pose='neutral',speaker=False):
  self.rect(x,y,w,h,'#F0F3F5',r=7,sw=1.2)
  cx,cy=x+w/2,y+h*.44;tilt=9 if pose=='nod' else -4 if pose=='smile' else 0
  self.face_id+=1
  cid=f'portrait-{self.face_id}'
  self.p.append(f'<defs><clipPath id="{cid}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7"/></clipPath></defs><g clip-path="url(#{cid})"><g transform="rotate({tilt} {cx} {cy})">')
  self.p.append(f'<path d="M{x+w*.12} {y+h}Q{cx} {y+h*.43} {x+w*.88} {y+h}" fill="{C["orange_edge"] if speaker else "#8FB3AD"}"/>')
  self.p.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{w*.24}" ry="{h*.29}" fill="#E3C8AE"/>')
  self.p.append(f'<path d="M{cx-w*.24} {cy}Q{cx-w*.3} {y+h*.10} {cx} {y+h*.12}Q{cx+w*.32} {y+h*.12} {cx+w*.24} {cy}Q{cx+w*.07} {cy-h*.19} {cx-w*.08} {cy-h*.12}Z" fill="#5E5753"/>')
  for dx in [-.10,.10]:
   ex=cx+w*dx
   if pose=='laugh':self.path(f'M{ex-4} {cy+1}q4 -4 8 0','ink',1.8)
   else:self.p.append(f'<circle cx="{ex}" cy="{cy}" r="2" fill="{C["ink"]}"/>')
   if pose=='brow':self.path(f'M{ex-5} {cy-10}h10','ink',1.6)
  self.path(f'M{cx} {cy+h*.04}l{-w*.02} {h*.065}h{w*.035}','muted',1.3)
  if pose=='laugh':self.p.append(f'<path d="M{cx-w*.07} {cy+h*.15}Q{cx} {cy+h*.19} {cx+w*.07} {cy+h*.15}Q{cx} {cy+h*.31} {cx-w*.07} {cy+h*.15}" fill="#795B57"/>')
  elif pose in ('smile','nod'):self.path(f'M{cx-w*.065} {cy+h*.17}q{w*.065} {h*.075} {w*.13} 0','ink',1.8)
  else:self.path(f'M{cx-w*.05} {cy+h*.17}h{w*.10}','ink',1.6)
  self.p.append('</g></g>')
  if pose=='nod':self.path(f'M{x+w-14} {y+15}q11 20 -1 32','teal_edge',1.7,True)
 def box(self,x,y,w,h,title,sub=None,fill='gray',edge='border'):
  self.rect(x,y,w,h,fill,edge)
  self.text(x+w/2,y+h/2-(4 if sub else -8),title,24,bold=True,anchor='middle',maxw=w-22)
  if sub:self.text(x+w/2,y+h/2+26,sub,19,'muted',anchor='middle',maxw=w-22)
 def save(self,name):
  svg='\n'.join(self.p+['</g></svg>']);(OUT/(name+'.svg')).write_text(svg)
  cairosvg.svg2png(bytestring=svg.encode(),write_to=str(OUT/(name+'.png')),output_width=2400,output_height=round(2400*self.h/self.w))
  cairosvg.svg2pdf(bytestring=svg.encode(),write_to=str(OUT/(name+'.pdf')),output_width=672,output_height=672*self.h/self.w)
  d=pymupdf.open(OUT/(name+'.pdf'));pg=d[0];outside=[]
  for block in pg.get_text('dict')['blocks']:
   for line in block.get('lines',[]):
    for span in line['spans']:
     if not pg.rect.contains(pymupdf.Rect(span['bbox'])):outside.append(span['text'])
  assert not outside,outside
  return {'name':name,'pdf_points':[pg.rect.width,pg.rect.height],'vector_drawings':len(pg.get_drawings()),'raster_images':len(pg.get_images()),'text_outside_page':outside}

def task():
 f=Figure(1600,850,'F1: Audiovisual backchannels while listening')
 f.text(28,46,'Audiovisual backchannels while listening',34,bold=True)
 f.text(28,82,'Current speaker audio + listener first frame → listener video and audio',23,'muted')
 # Inputs.
 f.rect(28,116,336,657,'gray')
 f.text(52,157,'(a) Inputs',26,bold=True)
 f.text(52,209,'Current speaker audio',22,bold=True)
 f.wave(52,259,286,31)
 f.lines(52,322,['“I finally passed my exam!', 'I’m so happy about it.”'],22,gap=31,color='orange_edge')
 f.face(116,394,158,180)
 f.text(196,612,'Listener first frame',22,bold=True,anchor='middle')
 f.lines(52,679,['No dialogue history','No target listener audio'],20,gap=30,color='muted')
 # Shared speaker turn.
 f.rect(480,116,1092,127,'orange','orange_edge')
 f.text(505,151,'(b) Feedback during the same speaker turn',25,bold=True)
 f.text(505,195,'Speaker',21,bold=True)
 f.wave(725,192,793,22)
 f.path('M725 230H1518','orange_edge',1.7,True)
 f.text(1115,223,'Speaker holds the conversational floor',18,'orange_edge',anchor='middle')
 # Visual-only output scope (not a claimed baseline experiment).
 f.rect(480,270,1092,218,'gray')
 f.text(505,309,'Visual listening',25,bold=True)
 f.lines(505,364,['Video output', 'Nods and facial', 'expressions'],21,gap=30,color='muted')
 xs=[755,950,1145,1340]
 for x,p,label in zip(xs,['neutral','nod','smile','smile'],['attend','nod','smile','smile']):
  f.face(x,328,148,122,p);f.text(x+74,476,label,18,'muted',anchor='middle')
 # Our output.
 f.rect(480,513,1092,275,'teal','teal_edge',sw=2.3)
 f.text(505,551,'Our task: audiovisual listening',25,bold=True)
 f.text(505,624,'Listener video',21,bold=True)
 for x,p,label in zip(xs,['neutral','nod','smile','laugh'],['attend','nod','smile','laugh']):
  f.face(x,568,148,113,p);f.text(x+74,705,label,18,'teal_edge',anchor='middle')
 f.text(505,747,'Listener audio',21,bold=True)
 f.wave(755,743,733,18,'listener','teal_edge')
 f.text(1026,769,'brief “mm-hm”',16,'teal_edge',anchor='middle')
 f.text(1370,769,'laughter',16,'teal_edge',anchor='middle')
 f.path('M364 641H480','teal_edge',3,True);f.text(420,625,'Generate',18,'teal_edge',anchor='middle')
 # Scope, rather than evidence of superiority.
 f.text(28,816,'Illustrative output scopes, not measured results. Quiet attentive listening is also valid.',21,'muted')
 return f.save('F1_task_motivation_v1')

def data():
 f=Figure(1600,1050,'F2: Data construction and cue-assisted annotation')
 f.text(28,46,'(a) From conversations to audiovisual listening pairs',32,bold=True,maxw=1544)
 f.text(28,82,'Aligned media → timestamped cues → ordered descriptions',23,'muted')
 xs=[28,425,822,1219];ww=353
 for i,(x,title) in enumerate(zip(xs,['Source conversations','Segment & pair','Prepare listener media','Annotate & screen'])):
  f.rect(x,116,ww,252,['gray','orange','blue','purple'][i]);f.badge(x+27,144,i+1,['blue_edge','orange_edge','blue_edge','purple_edge'][i]);f.text(x+57,153,title,23,bold=True,maxw=ww-72)
  if i<3:f.path(f'M{x+ww+4} 245H{xs[i+1]-6}','muted',2.3,True)
 # Stage 1.
 f.face(65,183,106,111,speaker=True);f.face(222,183,106,111,'nod')
 f.text(204,325,'Seamless Interaction',22,bold=True,anchor='middle')
 f.text(204,351,'Paired recordings and audio',19,'muted',anchor='middle')
 # Stage 2.
 f.text(445,198,'Speaker turn',19,'orange_edge');f.wave(449,227,300,18)
 f.rect(480,253,207,36,'orange','orange_edge',r=3)
 f.text(583,278,'Paired listening interval',17,'orange_edge',anchor='middle')
 f.lines(445,322,['Pauses + ASR punctuation', 'Sentence-boundary rules'],20,gap=28,color='muted')
 # Stage 3.
 f.face(844,181,84,106)
 f.text(951,200,'Aligned listener A/V',18,bold=True)
 f.wave(953,239,195,20,'listener','teal_edge')
 f.lines(844,321,['Face crop + quality screening','Audio masking + boundary fades'],19,gap=28,color='muted')
 # Stage 4.
 f.box(1240,187,311,62,'Qwen3-Omni','Cue-assisted description','purple','purple_edge')
 f.lines(1240,290,['Check cue coverage', 'and audio consistency', 'Annotation detail below'],20,gap=29,color='muted')
 f.path('M1390 368V393','purple_edge',2.4,True)
 # Expanded annotation mechanism.
 f.rect(28,398,1544,411,'white','border',dash=True)
 f.text(51,438,'(b) Cue-assisted annotation',27,bold=True)
 f.text(1548,438,'Illustrative example',20,'muted',anchor='end')
 # Direct media path explicitly reaches annotator.
 f.path('M201 529V477H920V529','blue_edge',2.3,True)
 f.text(550,468,'Listener video + audio',19,'blue_edge',anchor='middle')
 f.rect(53,529,297,226,'blue','blue_edge')
 f.text(74,567,'Paired listener media',23,bold=True,maxw=254)
 for x,pose in zip([77,160,243],['nod','brow','neutral']):f.face(x,588,65,77,pose)
 f.wave(77,707,231,20,'listener','teal_edge')
 f.rect(402,529,329,226,'orange','orange_edge')
 f.text(424,567,'Timestamped cue prompts',22,bold=True,maxw=282)
 f.lines(424,612,['Nod             0.8–1.5 s','Brow raise    1.9–2.2 s','“mm-hm”      2.7–3.1 s'],22,gap=35)
 f.text(424,738,'Detected visual + vocal cues',18,'muted')
 f.path('M350 646H402','blue_edge',2.4,True)
 f.rect(783,529,273,226,'purple','purple_edge')
 f.text(919,599,'Qwen3-Omni',26,bold=True,anchor='middle')
 f.lines(919,643,['Observe A/V', 'with cue prompts'],22,gap=31,color='purple_edge',anchor='middle')
 f.path('M731 646H783','orange_edge',2.4,True,True)
 f.rect(1108,529,440,226,'teal','teal_edge')
 f.text(1130,567,'Ordered reaction description',23,bold=True,maxw=397)
 f.lines(1130,611,['“The listener nods, then raises', 'their eyebrows, and gives', 'a brief ‘mm-hm’ response.”'],22,gap=32)
 f.text(1130,738,'Action order; no numeric timestamps',19,'teal_edge')
 f.path('M1056 646H1108','purple_edge',2.4,True)
 f.text(51,787,'Times guide annotation; the final description retains the order of reactions.',21,'muted')
 # Retained dataset and derived event labels.
 f.text(28,853,'(c) Retained samples and derived training labels',27,bold=True)
 f.rect(28,880,690,132,'gray')
 f.text(49,913,'Speaker audio · reference frame · listener A/V · caption',20,bold=True,maxw=646)
 f.text(49,965,'TRAIN  704,368',30,'blue_edge',True)
 f.text(393,965,'TEST  7,087',30,'blue_edge',True)
 f.text(49,996,'Final retained split counts',18,'muted')
 f.box(755,880,192,70,'Description',fill='teal',edge='teal_edge')
 f.box(982,880,278,70,'Behavior statistics','Qwen3-30B','purple','purple_edge')
 f.box(1295,880,277,70,'Clip labels + masks',fill='coral',edge='coral_edge')
 f.path('M947 915H982','muted',2.2,True);f.path('M1260 915H1295','muted',2.2,True)
 f.lines(755,978,['Resolve conflicting evidence; mask unknown categories.', 'Clip-level presence, without timestamp supervision.'],19,gap=27,color='muted')
 f.text(28,1037,'All portraits, waveforms, cue intervals and example text are schematic placeholders.',19,'muted')
 return f.save('F2_data_annotation_v1')

if __name__=='__main__':
 report=[task(),data()];(OUT/'CHECK.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
