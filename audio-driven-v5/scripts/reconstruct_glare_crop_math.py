def boxes_for_face(box):
 x1,y1,x2,y2=map(float,box[:4]);cx=(x1+x2)/2;cy=(y1+y2)/2;side=int(round(max(x2-x1,y2-y1)*1.5));side+=side%2
 x=int(round(cx-side/2));y=int(round(cy-side/2))
 return {'square_1p5':[x,y,x+side,y+side],'rectangle_1p5':[int(round(cx-(x2-x1)*.75)),int(round(cy-(y2-y1)*.75)),int(round(cx+(x2-x1)*.75)),int(round(cy+(y2-y1)*.75))]}
