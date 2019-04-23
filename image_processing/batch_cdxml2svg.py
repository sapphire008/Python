# -*- coding: utf-8 -*-
"""
Created on Sat Aug 08 05:08:22 2015

Batch convert cdxml to svg file

@author: Edward
"""
import subprocess, glob, os
source_dir = 'C:/ProgramData/CambridgeSoft/ChemOffice2015/ChemDraw/ChemDraw Items/BioDrawResources'
target_dir = 'C:/Users/Edward/Desktop/save'
#'"C:/Program Files (x86)/CambridgeSoft/ChemOffice2015/ChemDraw/ChemDrawSVG/SVGConverter.exe"  -i"C:/Users/Edward/Desktop/save/GS_Amoeba.cdxml" -o"C:/Users/Edward/Desktop/save/GS_Amoeba.svg" -c"C:/Program Files (x86)/CambridgeSoft/ChemOffice2015/ChemDraw/ChemDrawSVG/configuration.xml"'
cmd_str = '"C:/Program Files (x86)/CambridgeSoft/ChemOffice2015/ChemDraw/ChemDrawSVG/SVGConverter.exe" -i"%s" -o"%s" -c"C:/Program Files (x86)/CambridgeSoft/ChemOffice2015/ChemDraw/ChemDrawSVG/configuration.xml"'

source_img_list = glob.glob(os.path.join(source_dir,'*.cdxml'))
fid = open(os.path.join(target_dir, 'cdxml2svg.bat'),'w')
for source_img in source_img_list:
    source_img = source_img.replace('\\','/')
    target_img = os.path.basename(source_img).replace('.cdxml','.svg')
    target_img = os.path.join(target_dir, target_img).replace('\\','/')
    #exe_cmd = [cmd_str[0], cmd_str[1]%(source_img), cmd_str[2]%(target_img), cmd_str[3]]
    #subprocess.call(exe_cmd)
    exe_cmd = cmd_str%(source_img, target_img)
    fid.write(exe_cmd)
    fid.write('\r\n')
fid.close()

