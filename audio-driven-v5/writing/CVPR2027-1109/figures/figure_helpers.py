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
