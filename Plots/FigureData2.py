# -*- coding: utf-8 -*-
"""
Frontend for Plots

Created on Wed Jul 15 10:53:25 2015

@author: Edward
"""

import pandas as pd
import numpy as np

dataFile = 'C:/Users/Edward/Documents/Assignments/Scripts/Python/Plots/example/scatter3d.csv'

class FigureData(object):
    def __init__(self, dataFile=None):
        """Initialize class"""
        self.series = {'x':[],'y':[],'z':[]}
        self.meta = {} # a list of meta parameters in the file

        if dataFile is not None and isinstance(dataFile, str):
            # load directly if all the conditions are met
            self.LoadFigureData(dataFile=dataFile)
        else:
            raise IOError('Unrecognized data file input')

    def LoadFigureData(self, dataFile, sep=',', metachar="|"):
        """Load data file"""
        # check file exists
        fid = open(dataFile, 'r')
        self.table = []
        for rownum, line in enumerate(fid): # iterate through each line
            line = line.strip().strip(sep).replace('\t','').replace('"','')
            if not line or line[0] == "#" or line==line[0]*len(line):
                continue # skip comments and empty lines
            if line[0] == metachar: # metadata starts with "|"
                self.parse_meta(line, metachar)
            else: # assuming the rest of the file is data table
                fid.close()
                break
        # read in the rest of the data
        self.table = pd.read_csv(dataFile, sep=sep, comment="#",
                                 skipinitialspace=True, skiprows=rownum,
                                 skip_blank_lines=True)
        # parse the data to variable
        self.parse_data()
        # set more parameters
        self.set_default_labels()
        # Reorganize errorbar
        self.parse_error_bar()
        # Count number of data sets for each series
        self.count_series()
        self.get_shape()
        self.get_size()

    def set_default_labels(self,cat=None):
        def copyvalue(meta, f, g, cat=None):
            if f not in meta.keys() and g in meta.keys():
                if isinstance(meta[g],list):
                    meta[f] = cat.join(meta[g]) if cat is not None else ""                    
                else:
                    meta[f] = meta[g]
            return(meta)
        self.meta = copyvalue(self.meta, 'xlabel','x', cat=cat)
        self.meta = copyvalue(self.meta, 'ylabel','y', cat=cat)
        if self.series['z']:
            self.meta = copyvalue(self.meta, 'zlabel','z', cat=cat)
    
    def parse_error_bar(self):
        """Reorganize errorbar"""
        vec = [0,0,0,0,0,0]
        vec[0] = 1 if 'errorbar_pos' in self.series.keys() else 0
        if vec[0] == 1:
            vec[1] = 1 if self.series['errorbar_pos'] else 0
        vec[2] = 1 if 'errorbar_neg' in self.series.keys() else 0
        if vec[2] == 1:
            vec[3] = 1 if self.series['errorbar_neg'] else 0
        vec[4] = 1 if 'errorbar' in self.series.keys() else 0
        if vec[4] == 1:
            vec[5] = 1 if self.series['errorbar'] else 0
        # both pos and neg exist, but not errorbar
        if [vec[v] for v in [0,1,2,3,5]] == [1,1,1,1,0]:
            self.series['errorbar'] = [np.array([x,y]) for x,y in zip(\
                   self.series['errorbar_pos'], self.series['errorbar_neg'])]
        # pos exist but not neg
        elif [vec[v] for v in [0,1,3,5]] == [1,1,0,0]:
            self.series['errorbar'] = self.series['errorbar_pos']
        # neg exist but not pos
        elif [vec[v] for v in [1,2,3,5]] == [0,1,1,0]:
            self.series['errorbar'] == self.series['errorbar_neg']
            
        # remove errorbar_pos and errorbar_neg           
        self.series.pop('errorbar_pos',None)
        self.series.pop('errorbar_neg',None)
        
    def parse_meta(self, line,metachar="|"):
        """Parse parameter"""
        line = line.replace(metachar,"")
        m, v = line.split("=") # metavaraible, value
        m, v = m.strip(), v.strip()
        # parse value if it is a list
        if v.lower() == "none":
            self.meta[m] = None
            return    
        if "[" in v and "," in v:
            v = v.replace("[","").replace("]","").split(",")
            v = [x.strip() for x in v]
        self.meta[m] = v
        
    def parse_data(self):
        """Put table data into series"""
        # series
        cheader = list(self.table.columns.values)
        for key,val in iter(self.meta.items()):
            #if key in ['x','y','z']: continue
            if val is None:
                continue
            elif isinstance(val, str):
                if val not in cheader:
                    continue
                else:
                    self.series[key].append(np.array(\
                                        list(self.table[val].values)))
            elif all([v in cheader for v in val]): # assume it is a list
                self.series[key] = []             
                for v in val:
                    self.series[key].append(np.array(list(
                                                    self.table[v].values)))
            elif any([v in cheader for v in val]):
               raise SyntaxError(\
                     "Some header reference to '%s' is spelled wrong"% (key))
            #else:
                #raise RuntimeError(\
                #        "Fall through. Check (key, value)=(%s, %s)"%(key,val))
    def get_size(self):
        """Get the numel of each data series except the nan or empty string"""
        def checknumel(v):
            if isinstance(v[0], np.string_): # check for empty
                return(len([x for x in v if x]))
            else:
                try:
                    return(np.count_nonzero(~np.isnan(v)))
                except:
                    raise TypeError("unrecognized type %s" %(type(v))) 
            
        self.size = {}
        for k in self.series.keys():
            self.size[k] = [checknumel(v) for v in self.series[k]]
            
    def get_shape(self):
        """Get shape of each series"""
        self.shape = {}
        for k in self.series.keys():
            self.shape[k] = [v.shape for v in self.series[k]]
    
    def count_series(self):
        """Get number of series"""
        self.num = {}
        for k in self.series.keys():
            self.num[k] = len(self.series[k])
            
if __name__ == '__main__':
    K = FigureData(dataFile)
