# -*- coding: utf-8 -*-
"""
Created on Fri Jul 03 18:02:13 2015

@author: Edward
"""
DEBUG=True
import re
txtdir = 'C:/Users/Edward/Documents/Assignments/Scripts/Python/Plots/resource/colorbrewer2.txt'

def readColorBrewers(txtdir, lineexp=r'(\w+)(?:,|$)\[([^\]]+)\](?:,|$)\[([^\]]+)\](?:,|$)',
                     rgbexp = r'(\d+),(\d+),(\d+)'):
    """read color brewers text file"""
    #colors={}
    with open(txtdir, 'r') as fid:
        for line in fid:
            if not line.strip() or line[0] == "#":
                continue  # skip comments or empty line
            name, colorcomb, rgb  = re.findall(lineexp, line)[0]
            # parse color list
            rgb_list = re.findall(rgbexp, rgb)
            # change into integer
            rgb_list = [tuple([int(r) for r in s]) for s in rgb_list]
            
            
            # print
            rgbstr = ""
            for n, rgb in enumerate(rgb_list):
                rgbstr += "(%d, (%d,%d,%d))," %(n, rgb[0],rgb[1],rgb[2])
            # print("'%s' : OrderedDict([%s])" % (name, rgbstr[:-1]))
            print("'%s',"% (name)),
    fid.close()