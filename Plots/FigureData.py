# -*- coding: utf-8 -*-
"""
Created on Sat Jul 04 21:50:41 2015

@author: Edward
"""
import os, re
import numpy as np


dataFile= 'C:/Users/Edward/Documents/Assignments/Scripts/Python/Plots/example/scatter3d.txt'

class PlotProperty(object):
    pass

class FigureData2(object):
    def __init__(self, dataFile=None):
        """Initialize class"""
        self.series = {'x':[],'y':[],'z':[]} 
        self.properties = PlotProperty() # composition
        self.annotations = {} # composition
        self.num = {'x':[],'y':[],'z':[]} # count number of data sets
        
        if dataFile is not None and isinstance(dataFile, str):
            # load directly if all the conditions are met
            self.LoadFigureData(dataFile=dataFile) 
        else:
            raise IOError('Unrecognized data file input')
    
    def LoadFigureData(self, dataFile, delimiter=','):
        """Load data file"""
        # check file exists
        if not os.path.isfile(dataFile):
            raise IOError('%s does not exist' %dataFile)
        fid = open(dataFile, 'rb')
        # initialize some temp variables
        self.has_label = False 
        self.current_annot_name = None
        for line in fid: # iterate through each line
            line = line.strip().replace('\t','')
            if not line or line[0] == "#" or line=='[Default]':
                continue # skip comments and empty lines
            if line[0:5] == '[Data': # new section
                self.NewDataSeries(line)
            elif line[0:11] == '[Annotation':
                self.NewAnnotation(line)
            # parse data
            elif line[1:4] == 'PRM': # additional parameters
                self.ParsePRM(line)
            elif line[1:4] == 'VAR':# dictionary to order data
                self.ParseVAR(line)
            elif line[1:4] == 'LAB': # label
                self.ParseLAB(line)
                self.has_label = True
            elif line[1:4] == 'NAM':# header
                if not self.has_label: 
                    self.ParseLAB(line)
            else: # data
                if self.current_read == 'data':
                    self.ParseData(line)
                elif self.current_read == 'annotation':
                    self.ParseAnnotation(line)

        fid.close()
        # clean up the attributes
        del(self.has_label)
        del(self.current_annot_name)
                 
    def NewDataSeries(self, line):
        """Add new group of data"""
        self.current_read = 'data'
        for k in self.series.keys(): # append an array
            self.series[k].append(np.array([]))
        # Parse Data block
        line = line[1:-1] # get rid of brackets
        # Group 
        if not hasattr(self.properties, 'groups'):
            self.properties.groups = []
        self.properties.groups += re.findall(r"Group\('(.*)'\)", line)
        
    def NewAnnotation(self,line):
        self.current_read = 'annotation'
        self.current_annot_name = re.findall(r"Type\('(.*)'\)", line)[0]
        # add new annotation object
        if self.current_annot_name not in self.annotations.keys():
            self.annotations[self.current_annot_name] = []
            
    def ParseVAR(self, line):
        lineregexp = r'\(([^\)]+)\)(?:,|$)'
        lst = re.findall(lineregexp, line)
        v = [tuple(s.split(',')) for s in lst]
        # a dictionary that helps identify data; reset when called
        self.vardict = {} 
        for _, (a,b,c) in enumerate(v):
            self.vardict[a] = (int(b), c) # (column number, data type)
    
    def ParseLAB(self, line):
        lst = line.split("'")[1::2]
        if not hasattr(self.properties, 'labels'): 
            self.properties.labels = {} # initialize as dictionary
        for n, l in enumerate(lst):
            s = chr(120+n) #x, y, z
            if s in self.properties.labels.keys():
                self.properties.labels[s].append(l)
            else:
                self.properties.labels[s] = [l]
               
    def ParsePRM(self, line):
        lst = line.split(',')
        if not hasattr(self.properties, 'parameters'):
            self.properties.parameters = {}
        lst[1] = self.ReadData(lst[1], 'str') # clean string
        self.properties.parameters[lst[1]] = self.ReadData(lst[3], lst[2])
    

    def ParseData(self, line):
        lst = line.split(',')
        for k, v in iter(self.vardict.items()):
            f = self.ReadData(lst[int(v[0])-1], v[1])
            self.series[k][-1] = np.append(self.series[k][-1], f)
            
    def ParseAnnotation(self, line):
        self.annotations[self.current_annot_name].append(line)
                                           
    def CountData(self):
        """Count number of data set"""
        for k in self.series.keys():
            self.num[k] = len(self.series[k])
            
    @staticmethod
    def ReadData(d, dt):
        f = {'float': float, 'int': int, 'str':str}.get(dt, float)
        s = f(d)
        if isinstance(s, str): # strip any quotes
            s = s.strip()
            if s[0] == "'":
                s = s[1:]
            if s[-1] == "'":
                s = s[:-1]
        return(s)
        
            
        
        
if __name__ == '__main__':
    K = FigureData2(dataFile)