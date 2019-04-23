#!/usr/bin/python
# -*- coding: latin-1 -*-
#- ai2svg converter 060901c (for Inkscape .inx) - Paulo Silva (GPL licence)
#- file version reading: 1,2,3,4,5,6,7 - saved from Adobe Illustrator 7
#- file version reading: 1,2,3,5,7 - exported from Macromedia Freehand 8
#---------------------------------------
#- missing:                            |
#-   - grouped paths (????_            |
#-   - placed and embedded pics (????_ |
#-   - gradients (????)                |
#-   - patterns (????)                 |
#-   - cmyk to rgb accuracy (????)     |
#---------------------------------------
#- layers, joined paths, dashes: ok
#- textboxes (with some bugs: anchors, characters, etc...)

import os,sys
finp_st=sys.argv[1];fout_st=finp_st+".svg"
fstcmd=0;xo=0;yo=0;xn=0;yn=0
#--------functions-------------------------------------------------
def reverse(a1_st): #- gets a reversed string
  a1_st=a1_st.replace("\n","");a1leng=len(a1_st);tmpr_st=""
  for lp1 in range(0,a1leng,1):
    tmpr_st+=a1_st[a1leng-1-lp1]
  return tmpr_st
def locstrinfo(a_st,b_st,c_st): #- seeks a substring between defined chars
  adra=0;adrb=0;d_st=""
  while adrb<=len(a_st) and adrb<len(b_st):
    if a_st[adra]==b_st[adrb]:adrb+=1
    adra+=1
  while a_st[adra]!=c_st[0]:
    d_st+=a_st[adra];adra+=1
  return d_st
def hexvl(a_st): #- does the reverse way from 'hex()'
  a_st=a_st.lower();tmpr=0;hx_st="0123456789abcdef";hx_st=hx_st.lower()
  for i in range(0,len(a_st),1):tmpr=(tmpr*16)+hx_st.find(a_st[i])
  return tmpr
def fixhxc_st(va): #- fixes the lenght of a hex string to 6 characters
  tm_st="000000"+hex(va).lstrip("0x");tm_st="#"+tm_st[len(tm_st)-6:]
  return tm_st.upper()
def fixhex(va,sz): #- fixes the lenght of a hex string
  tml_st=""
  for i in range (0,sz,1):tml_st+="0"
  tml_st+=hex(va).lstrip("0x");tml_st=tml_st[len(tml_st)-sz:]
  return tml_st.upper()
def hexfromrgb(r1,g1,b1): #- gets hexcolour string from rgb
  if r1<0:r1=0
  if g1<0:g1=0
  if b1<0:b1=0
  if r1>255:r1=255
  if g1>255:g1=255
  if b1>255:b1=255
  rgbv=(65536*r1)+(256*g1)+(b1);tm_st="000000"+hex(rgbv).lstrip("0x")
  tm_st=tm_st[len(tm_st)-6:]
  return tm_st.upper()
def hexcmykrgbinv(c1,m1,y1,k1): #- gets fake rgb from inverting cmyk
  r1=int((1-c1-k1)*255)
  if r1<0: r1=0
  g1=int((1-m1-k1)*255)
  if g1<0: g1=0
  b1=int((1-y1-k1)*255)
  if b1<0: b1=0
  rgbv=(65536*r1)+(256*g1)+(b1);hxrgbv_st=hex(rgbv)
  #print rgbv
  tm_st="000000"+hxrgbv_st.lstrip("0x");tm_st=tm_st[len(tm_st)-6:]
  return tm_st.upper()
def ystupdn(yc1_st,ye1_st): #- flips y position (string to string)
  return str(float(ye1_st)-float(yc1_st))
def aisvgtxrepl(txtm_st): #- brand new function i don't know if works (Tx)(latin-1)
  #- some cleanup
  txtm_st=txtm_st.replace("<","&lt;")
  txtm_st=txtm_st.replace(">","&gt;")
  txtm_st=txtm_st.replace("&","&amp;")
  txtm_st=txtm_st.replace("\"","&quot;")
  txtm_st=txtm_st.replace("\'","&#39;")
  txtm_st=txtm_st.replace("\\)",")") #- not working?
  txtm_st=txtm_st.replace("\\(","(")
  txtm_st=txtm_st.replace("\\015","") #- .ai linebreak
  txtm_st=txtm_st.replace("\\136","?")
  txtm_st=txtm_st.replace("\\177","?")
  #- from 128 to 159 (unicode stuff...)
  txtm_st=txtm_st.replace("\\200","&#8364;");txtm_st=txtm_st.replace("\\201","&#32;")
  txtm_st=txtm_st.replace("\\202","&#8218;");txtm_st=txtm_st.replace("\\203","&#402;")
  txtm_st=txtm_st.replace("\\204","&#8222;");txtm_st=txtm_st.replace("\\205","&#8230;")
  txtm_st=txtm_st.replace("\\206","&#8224;");txtm_st=txtm_st.replace("\\207","&#8225;")
  txtm_st=txtm_st.replace("\\210","&#710;"); txtm_st=txtm_st.replace("\\211","&#8240;")
  txtm_st=txtm_st.replace("\\212","&#352;"); txtm_st=txtm_st.replace("\\213","&#8249;")
  txtm_st=txtm_st.replace("\\214","&#338;"); txtm_st=txtm_st.replace("\\215","&#32;")
  txtm_st=txtm_st.replace("\\216","&#381;"); txtm_st=txtm_st.replace("\\217","&#32;")
  txtm_st=txtm_st.replace("\\220","&#32;");  txtm_st=txtm_st.replace("\\221","&#8216;")
  txtm_st=txtm_st.replace("\\222","&#8217;");txtm_st=txtm_st.replace("\\223","&#8220;")
  txtm_st=txtm_st.replace("\\224","&#8221;");txtm_st=txtm_st.replace("\\225","&#8226;")
  txtm_st=txtm_st.replace("\\226","&#8211;");txtm_st=txtm_st.replace("\\227","&#8212;")
  txtm_st=txtm_st.replace("\\230","&#732;"); txtm_st=txtm_st.replace("\\231","&#8482;")
  txtm_st=txtm_st.replace("\\232","&#353;"); txtm_st=txtm_st.replace("\\233","&#8250;")
  txtm_st=txtm_st.replace("\\234","&#339;"); txtm_st=txtm_st.replace("\\235","&#32;")
  txtm_st=txtm_st.replace("\\236","&#382;"); txtm_st=txtm_st.replace("\\237","&#376;")
  #- from 160 to 255
  for i in range(160,256,1):
    j=((i&7)+(((i&56)/8)*10)+(((i&192)/64)*100))
    txtm2_st="&#"+str(i)+";";txtm1_st="\\"+str(j)
    txtm_st=txtm_st.replace(txtm1_st,txtm2_st)
  txtm_st=txtm_st.replace("\\","&#92;")
  return txtm_st

def clzempty(f_hxstroke_st,f_strokewidth_st,f_miterlimit_st,f_dasharray_st,f_dashoffset_st):
  print"z\" style=\"fill-rule:evenodd;fill:none;fill-opacity:1;"
  print"stroke:#"+f_hxstroke_st+";stroke-opacity:1;stroke-width:"+f_strokewidth_st+";"
  print"stroke-linecap:butt;stroke-linejoin:miter;"
  print"stroke-miterlimit:"+f_miterlimit_st+";stroke-dasharray:"+f_dasharray_st+";stroke-dashoffset:"+f_dashoffset_st+";"
  print"visibility:visible;display:inline;overflow:visible\"/>\n"

def clzpolyline(f_hxstroke_st,f_strokewidth_st,f_miterlimit_st,f_dasharray_st,f_dashoffset_st):
  print"\" style=\"fill-rule:evenodd;fill:none;fill-opacity:1;"
  print"stroke:#"+f_hxstroke_st+";stroke-opacity:1;stroke-width:"+f_strokewidth_st+";"
  print"stroke-linecap:butt;stroke-linejoin:miter;"
  print"stroke-miterlimit:"+f_miterlimit_st+";stroke-dasharray:"+f_dasharray_st+";stroke-dashoffset:"+f_dashoffset_st+"\"/>\n"

def clzfilled(f_hxfill_st,f_hxstroke_st,f_strokewidth_st,f_miterlimit_st,f_dasharray_st,f_dashoffset_st):
  print"z\" style=\"fill-rule:evenodd;fill:#"+f_hxfill_st+";fill-opacity:1;"
  print"stroke:#"+f_hxstroke_st+";stroke-opacity:1;stroke-width:"+f_strokewidth_st+";"
  print"stroke-linecap:butt;stroke-linejoin:miter;"
  print"stroke-miterlimit:"+f_miterlimit_st+";stroke-dasharray:"+f_dasharray_st+";stroke-dashoffset:"+f_dashoffset_st+";"
  print"visibility:visible;display:inline;overflow:visible\"/>\n"

def clzsolid(f_hxfill_st,f_strokewidth_st,f_miterlimit_st,f_dashoffset_st):
  print"z\" style=\"fill-rule:evenodd;fill:#"+f_hxfill_st+";fill-opacity:1;"
  print"stroke:none;stroke-opacity:1;stroke-width:"+f_strokewidth_st+";"
  print"stroke-linecap:butt;stroke-linejoin:miter;"
  print"stroke-miterlimit:"+f_miterlimit_st+";stroke-dashoffset:"+f_dashoffset_st+";"
  print"visibility:visible;display:inline;overflow:visible\"/>\n"

#------------- code -----------------------------------------------
#if finp_st.lower()=="--help".lower():
#  print"ai2svg.py - Paulo Silva (GPL licence)"
#  print"usage: python ai2svg.py yourdrawing.ai"
#  print"the result will appear as neighbour named yourdrawing.ai.svg"
#  print"(please keep in mind all results may need some repairs)"
#else:
  #------------- starting --------------------------------------------
finp_fl=open(finp_st,"r")
#fout_fl=open(fout_st,"w")
rdflg=0;shapeflg=0;fntszflg=0
txboxflg=0;pathjoin=0;lastshape_st="f"
strokewidth=1;miterlimit=4;dasharray_st="";dashoffset=0;txboxline_st=""
hxfill_st="000000";hxstroke_st="000000"
xtxboxpos=0;ytxboxpos=0;ycurlead=0;ysteplead=10
idpath=0;idtxbox=0;idtxspan=0;lastjpath=0
id=0 #- try to make this variable obsolete, replacing with 'idpath'
#- svg header
print"<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>"
#------------- converter looping-------------------------------------

rdflg=1

while True:
  #- track state of being in a path
  is_open = False

  #- reading next .ai line
  wltext_st=finp_fl.readline()
  if len(wltext_st)==0:break
  wltext_st=wltext_st+"\n"
  wltext_st=wltext_st.replace("\r","\n")
  #- substituir por um loop do tamanho da string
  for tm1 in range(0,len(wltext_st)+2,1):
    wltext_st=wltext_st.replace("\n\n","\n")
  wltext_st=wltext_st.replace("%%TemplateBox:","%%TemplateBox: ")
  wltext_st=wltext_st.replace("%%TemplateBox:  ","%%TemplateBox: ")
  while len(wltext_st)>0:
    bkln=wltext_st.find("\n")
    text_st=wltext_st[:bkln+1]
    wltext_st=wltext_st[bkln+1:]

    #- cleaning breaklines and tabs from string
    text_st=text_st.replace("\n","");text_st=text_st.replace("\t"," ")
    text_st=text_st.strip();
    textrev_st=reverse(text_st);textrev_st=">"+textrev_st+" 0 0 0 0 0 0 0 0"
    text_st=">"+text_st+" 0 0 0 0 0 0 0 0"
    #- getting substrings
    v0_st=locstrinfo(text_st,">"," ")
    v1_st=locstrinfo(text_st,"> "," ")
    v2_st=locstrinfo(text_st,">  "," ")
    v3_st=locstrinfo(text_st,">   "," ")
    v4_st=locstrinfo(text_st,">    "," ")
    v5_st=locstrinfo(text_st,">     "," ")
    v6_st=locstrinfo(text_st,">      "," ")
    v7_st=locstrinfo(text_st,">       "," ")
    v0rev_st=reverse(locstrinfo(textrev_st,">"," "))
    v1rev_st=reverse(locstrinfo(textrev_st,"> "," "))
    #- gets paper size (ai1)
    if v0_st=="%%TemplateBox:":
      hv1v=float(v1_st)+float(v3_st);vv1v=float(v2_st)+float(v4_st)
      print"<svg width=\""+str(hv1v)+"pt\" height=\""+str(vv1v)+"pt\">\n\n"
      yed_st=str(vv1v)
    #- gets paper size (ai3)
    if v0_st=="%AI3_TemplateBox:":
      hv1v=float(v1_st)+float(v3_st);vv1v=float(v2_st)+float(v4_st)
      print"<svg width=\""+str(hv1v)+"pt\" height=\""+str(vv1v)+"pt\">\n\n"
      yed_st=str(vv1v)
    #- w - stroke width
    if v1_st=="w":strokewidth=float(v0_st)

    #- k - fill colour
    if v4_st=="k":hxfill_st=hexcmykrgbinv(float(v0_st),float(v1_st),float(v2_st),float(v3_st))
    #- K - stroke colour
    if v4_st=="K":hxstroke_st=hexcmykrgbinv(float(v0_st),float(v1_st),float(v2_st),float(v3_st))

    #- x - fill colour (?)
    if v0rev_st=="x":hxfill_st=hexcmykrgbinv(float(v0_st),float(v1_st),float(v2_st),float(v3_st))
    #- X - stroke colour (?)
    if v0rev_st=="X":hxstroke_st=hexcmykrgbinv(float(v0_st),float(v1_st),float(v2_st),float(v3_st))

    #- M - miter limit
    if v1_st=="M":miterlimit=float(v0_st)
    #- d - dashes (array and offset) and path start
    if v0rev_st=="d" and pathjoin==0: # and shapeflg==0 and rdflg==1
      #print"<path id=\"path_"+str(idpath)+"\" d=\""
      #idpath+=1;shapeflg=1
      dasharray_st=locstrinfo(text_st,"[","]")
      dasharray_st=dasharray_st.strip()
      dasharray_st=dasharray_st.replace(" ",",")
      if dasharray_st=="":dasharray_st="none"
      text1_st=text_st.replace("] ","]")
      dashoffset=float(locstrinfo(text1_st,"]"," ").strip())
      if dashoffset==0:dashoffset=0
    #- situation: 0 J 0 j 1 w 3.8636 M []0 d
    #- M: find 'M' (miter) position in string, get left$ until 'M' and get 2nd word as float from last
    if v7_st=="M":miterlimit=float(v6_st)

    #- w: find 'w' (line thickness width) in the same way of 'M' above
    if v5_st=="w":strokewidth=float(v4_st)

    #- XR - fill rule (what for?)

    #- Tp - textbox position (starts text box if not started???)
    if v0rev_st=="Tp":
      if shapeflg==1:
        print"\"/>"+"\n"
      txboxflg=1;xtxboxpos=float(v4_st);ytxboxpos=float(v5_st);ycurlead=ytxboxpos
      print"<text x=\""+str(xtxboxpos)+"\" y=\""+ystupdn(str(ytxboxpos),yed_st)+"\" id=\"tb_"+str(idtxbox)+"\" style=\"\n"
      idtxbox+=1
    #- Ta - text alignment (0 left, 1 mid, 2 right, 3 justified)
    if v0rev_st=="Ta":
      if int(float(v0_st))==0:
        print"text-align:start;text-anchor:start;\n"
      if int(float(v0_st))==1:
        print"text-align:center;text-anchor:middle;\n"
      if int(float(v0_st))==2:
        print"text-align:end;text-anchor:end;\n"
      if int(float(v0_st))==3: #- i'm not sure about this one
        print"text-align:start;text-anchor:start;\n"
    #- Tl - text leading   (check if "  " were replaced to " ")
    if v0rev_st=="Tl":
      ysteplead=float(v0_st)
    #- Tf - font size
    if v0rev_st=="Tf" and fntszflg==0:
      print"font-size:"+v1_st+"px;\""+"\n"
      fntszflg=1
    #- Tx - Tj - text strings
    if v0rev_st=="Tx" or v0rev_st=="Tj":
      txboxline_st=locstrinfo(text_st,"(",")")
      txboxline_st=aisvgtxrepl(txboxline_st)
      print"><tspan id=\"ts_"+str(idtxspan)+"\" x=\""+str(xtxboxpos)+"\" y=\""+ystupdn(str(ycurlead),yed_st)+"\"\n"
      print">"+txboxline_st+"</tspan\n"
      idtxspan+=1;ycurlead-=ysteplead
    #- TO - ends textbox
    if v0_st=="TO":
      print"></text>\n\n"
      txboxflg=0;shapeflg=0;fntszflg=0
    #- gets 'begin layer' ? (rdflg=1 ?)
    #if v0_st=="%AI5_BeginLayer":rdflg=1

    #- LB - ensures if path ends (?)
    if v0_st=="LB" and shapeflg==1:
      shapeflg=0;print"\"/>\n\n"
    #- %%EOF - end of file
    if v0_st=="%%EOF":print"</svg>\n"

    #- layer conversion - i have to clean it...
    #- Ln - begin layer? (beginning layer here, because layer name)
    #if v1_st=="Ln":print"<g id=\""+v0_st+"\">\n\n"
    #- %AI5_EndLayer-- - ends layer
    #if v0_st=="%AI5_EndLayer--":print"</g>\n\n"

    close_me = False
    #- *u - starts joined pathes
    if v0_st=="*u" or v0_st=="u":
      if is_open:
        close_me = True
      is_open = True
      pathjoin=1;lastjpath=0
    #- *U - ends joined pathes
    if v0_st=="*U" or v0_st=="U":
      if is_open:
        close_me = True
      is_open = False
      pathjoin=0;lastjpath=0

    if close_me:
      #- s - empty shapes (from joined pathes)
      if lastshape_st=="s":
        shapeflg=0
        clzempty(hxstroke_st,str(strokewidth),str(miterlimit),dasharray_st,str(dashoffset))
      #- S - polylines (from joined pathes)
      if lastshape_st=="S":
        shapeflg=0
        clzpolyline(hxstroke_st,str(strokewidth),str(miterlimit),dasharray_st,str(dashoffset))
      #- b - filled shapes (from joined pathes)
      if lastshape_st=="b":
        shapeflg=0
        clzfilled(hxfill_st,hxstroke_st,str(strokewidth),str(miterlimit),dasharray_st,str(dashoffset))
      #- B - filled shapes (from joined pathes (?))
      if lastshape_st=="B":
        shapeflg=0
        clzfilled(hxfill_st,hxstroke_st,str(strokewidth),str(miterlimit),dasharray_st,str(dashoffset))
      #- f - solid shapes (from joined pathes)
      if lastshape_st=="f":
        shapeflg=0
        clzsolid(hxfill_st,str(strokewidth),str(miterlimit),str(dashoffset))
      #- F - solid shapes (from joined pathes (?))
      if lastshape_st=="F":
        shapeflg=0
        clzsolid(hxfill_st,str(strokewidth),str(miterlimit),str(dashoffset))

    #- lastshape values from pathjoin=1
    if rdflg==1 and shapeflg==1 and pathjoin==1:
      if v0_st=="s":
        lastshape_st="s";print"z\n"
        if pathjoin==1:lastjpath+=1
      if v0_st=="S":
        lastshape_st="S";print"z\n"
        if pathjoin==1:lastjpath+=1
      if v0_st=="b":
        lastshape_st="b";print"z\n"
        if pathjoin==1:lastjpath+=1
      if v0_st=="f":
        lastshape_st="f";print"z\n"
        if pathjoin==1:lastjpath+=1
    #- checks if shapeflg and rdflg are true and pathjoin false? (... what for?)
    if rdflg==1 and pathjoin==0: # and shapeflg==1
      #- s - empty shapes
      if v0_st=="s":
        shapeflg=0
        clzempty(hxstroke_st,str(strokewidth),str(miterlimit),dasharray_st,str(dashoffset))
      #- S - polylines
      if v0_st=="S":
        shapeflg=0
        clzpolyline(hxstroke_st,str(strokewidth),str(miterlimit),dasharray_st,str(dashoffset))

      #- '(N) *' - polylines (guides exported from Freehand?)
      if v0_st=="(N)" and v0rev_st=="*":
        shapeflg=0
        clzpolyline(hxstroke_st,str(strokewidth),str(miterlimit),dasharray_st,str(dashoffset))

      #- N - leaves path opened (rare?)
      if v0_st=="N":
        shapeflg=0
        clzpolyline(hxstroke_st,str(strokewidth),str(miterlimit),dasharray_st,str(dashoffset))

      #- b - filled shapes
      if v0_st=="b":
        shapeflg=0
        clzfilled(hxfill_st,hxstroke_st,str(strokewidth),str(miterlimit),dasharray_st,str(dashoffset))
      #- B - filled shapes (?)
      if v0_st=="B":
        shapeflg=0
        clzfilled(hxfill_st,hxstroke_st,str(strokewidth),str(miterlimit),dasharray_st,str(dashoffset))
      #- f - solid shapes
      if v0_st=="f":
        shapeflg=0
        clzsolid(hxfill_st,str(strokewidth),str(miterlimit),str(dashoffset))
      #- F - solid shapes (?)
      if v0_st=="F":
        shapeflg=0
        clzsolid(hxfill_st,str(strokewidth),str(miterlimit),str(dashoffset))

    #- checks if shapeflg and rdflg are true
    if rdflg==1 and txboxflg==0: # and shapeflg==1

      #- m - first coordinate node from a path (shapeflg=0?)
      if v2_st=="m": # and shapeflg=0:
        # supposed to 'm' starting every path?
        #if pathjoin==0:
        if pathjoin==0 or (pathjoin==1 and lastjpath==0):
          print "<path id=\"path_"+str(idpath)+"\" d=\""
          idpath+=1;shapeflg=1
        print "M "+v0_st+" "+ystupdn(v1_st,yed_st)

      #- m - first coordinate node from a path (shapeflg=0)
      #if v2_st=="m" and shapeflg=0:
      #  # supposed to 'm' starting every path?
      #  print"<path id=\"path_"+str(idpath)+"\" d=\""
      #  idpath+=1;shapeflg=1
      #  print"M "+v0_st+" "+ystupdn(v1_st,yed_st)

      #- L - straight line coordinate from the last coordinate
      if v2_st=="l":
        print "L "+v0_st+" "+ystupdn(v1_st,yed_st)
      #- C - bezier line coordinates from the last coordinate
      if v6_st=="c":
        print "C "+v0_st+" "+ystupdn(v1_st,yed_st)+" "+v2_st+" "+ystupdn(v3_st,yed_st)+" "+v4_st+" "+ystupdn(v5_st,yed_st)

finp_fl.close()
#fout_fl.close()


 	  	 
