"""Editable F4 protocol schematic and F5 empty result layout. No measured data."""
from figure_helpers import Figure, OUT
import json

def evaluation():
 f=Figure(1600,660,'F4: Evaluation design — schematic, not results')
 f.text(28,44,'Three questions about generated listening feedback',31,bold=True)
 f.text(28,81,'Reference agreement, context, and internal coordination answer different questions.',23,'muted')
 specs=[(28,'(a) Clip-level category agreement','teal','teal_edge'),(554,'(b) Context and engagement','blue','blue_edge'),(1080,'(c) Audiovisual coordination','purple','purple_edge')]
 for x,title,fill,edge in specs:
  f.rect(x,110,492,478,fill,edge)
  f.text(x+20,149,title,24,bold=True,maxw=452)
 f.text(49,198,'Reference',20,bold=True);f.path('M185 195H492','muted',1.5,True)
 f.text(49,251,'Generated',20,bold=True);f.path('M185 248H492','muted',1.5,True)
 f.rect(213,179,65,25,'#9FC7BA','teal_edge',3);f.rect(315,179,45,25,'#9FC7BA','teal_edge',3)
 f.rect(387,232,65,25,'#9FC7BA','teal_edge',3)
 f.text(245,225,'nod',17,'teal_edge',anchor='middle');f.text(420,284,'nod',17,'teal_edge',anchor='middle')
 f.text(49,327,'Collapse repetition and timing',21,bold=True)
 f.path('M275 345V375','teal_edge',2,True)
 f.box(65,389,417,62,'{nod}  vs.  {nod}',fill='white',edge='teal_edge')
 f.lines(49,491,['Separate visual / audio P, R, macro-F1','Same category; different event times.','Agreement is not full appropriateness.'],19,gap=30,color='muted')
 f.box(579,181,442,72,'Current speaker audio',fill='white',edge='blue_edge')
 f.lines(579,297,['Does the feedback fit this utterance?', 'Does it convey natural engagement?', 'Is it excessive or premature?'],22,gap=38)
 f.box(579,414,442,66,'Generated listener response',fill='white',edge='blue_edge')
 f.lines(579,520,['Different reasonable responses are allowed.','Quiet listening can be appropriate.'],19,gap=30,color='muted')
 f.text(1103,202,'Generated video',20,bold=True);f.text(1103,310,'Generated audio',20,bold=True)
 f.rect(1103,224,440,56,'white','border',5);f.text(1323,260,'facial behavior / mouth movement',20,anchor='middle')
 f.rect(1103,331,440,56,'white','border',5);f.text(1323,367,'brief speech / laughter / silence',20,anchor='middle')
 f.path('M1323 284V304','purple_edge',2,True)
 f.lines(1103,438,['Do sound and movement agree?', 'Matching category sets alone cannot', 'establish coordination within the output.'],21,gap=31)
 f.text(1103,554,'Human protocol: details pending',19,'muted')
 f.text(28,629,'Schematic intervals only. Event detectors, category lists, and human-rating details remain to be finalized.',20,'muted')
 return f.save('F4_evaluation')

def results():
 f=Figure(1600,1000,'F5: Qualitative results layout — all result slots empty')
 f.text(28,45,'Qualitative comparison: layout only',32,bold=True)
 f.text(28,83,'Replace empty slots with actual synchronized media. No model outputs or performance claims are shown.',22,'muted')
 cols=[(240,'Quiet listening'),(685,'Audiovisual feedback'),(1130,'Failure analysis')]
 for x,title in cols:
  f.rect(x,110,419,110,'blue','blue_edge')
  f.text(x+209,145,title,24,bold=True,anchor='middle')
  f.lines(x+18,181,['Speaker audio + first frame: to insert','Same display times across methods'],18,gap=25,color='muted')
 rows=['Reference','A0: Base','A1: Full','A2: No direct audio','A3: No Feature','A4: No Event']
 for j,label in enumerate(rows):
  y=242+j*112
  f.text(28,y+40,label,20,bold=True,maxw=195)
  for x,_ in cols:
   f.rect(x,y,419,98,'gray',r=5)
   for k in range(3):
    f.rect(x+15+k*136,y+9,117,47,'white','border',r=3,dash=True)
    f.text(x+73+k*136,y+38,'frame',18,'muted',anchor='middle')
   f.path(f'M{x+16} {y+76}H{x+401}','border',1.3,dash=True)
   f.text(x+209,y+90,'own audio / verified reaction label: to insert',16,'muted',anchor='middle')
 f.text(28,952,'Use one observed reference, five controlled variants, and complete playable clips; retain failures.',20,'muted')
 f.text(28,981,'Blank boxes and dashed lines are placeholders, not silent outputs or measured waveforms.',20,'muted')
 return f.save('F5_results_layout')

if __name__=='__main__':
 report=[evaluation(),results()]
 (OUT/'NEW_FIGURES_CHECK.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))
