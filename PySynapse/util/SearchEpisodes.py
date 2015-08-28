# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:34:30 2015

Create a database for all the experiments
Searching for episodes that meet certain criterion
Export a .csv report for list of cells that meet the search criteria

@author: Edward
"""
import os, sys, sqlite3, re, fnmatch, datetime
sys.path.append('C:/Users/Edward/Documents/Assignments/Scripts/Python/Plots/')
import numpy as np
import pandas as pd
from ImportData import NeuroData



# basedirectory of the data structure
basedir ='D:/Data/2015/'
db_dir = 'D:/Data/2015/Query/PersistentDatabase.db'
monthlist = ['01.January','02.February','03.March','04.April','05.May','06.June','07.July','08.August','09.September','10.October','11.November','12.December']
monthabbr = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
episodeformat = '%s.%s.S%d.E%d.dat' #Neocortex A.09Aug15.S1.E46.dat
episoderegex = '([\s\.\w]+)\.(\d+)([a-zA-Z]+)(\d+)\.S(\d+)\.E(\d+)\.dat'  #Neocortex A.09Aug15.S1.E46.dat
episodefolder = 'Data %d %s %s' # Data 9 Aug 2015
yearprefix='20' # '20' + '15' = '2015'
# name = 'Neocortex A.09Aug15.S1.E46.dat'

class EpisodeDB(object):
    def __init__(self, basedir=basedir, db=None):
        """Initialize object"""
        self.basedir = basedir
        self.db = db
        self.monthlist = monthlist
        self.monthabbr = monthabbr
        self.episodeformat = episodeformat
        self.episoderegex = episoderegex
        self.episodefolder = episodefolder
        self.yearprefix = yearprefix
    
    def get_full_path(self, name, regexfmt=None):
        """Get the full path of the .dat file based on single file name"""
        if regexfmt is None: regexfmt = self.episoderegex
        cell, date, month, year, series, epi = re.findall(regexfmt, name)[0]
        if name[-4:] != '.dat': name += '.dat'
        year = self.yearprefix+year
        ep_folder = episodefolder % (int(date), month, year)
        dir_levels = [self.basedir, year, 
                      self.monthlist[self.monthabbr.index(month)], 
                      ep_folder, name]
        cdir = os.path.basename(os.path.abspath(self.basedir))
        if cdir in dir_levels:
            dir_levels.pop(dir_levels.index(cdir))
        outpath = os.path.join(*dir_levels).replace('\\','/')
        # check existence
        if not os.path.exists(outpath):
            print('Returned path: %s, but the path does not exist'%outpath)
        return(outpath)
    
    def parse_name(self, name, regexfmt=None, contracted=True):
        """Extract information based on name
        contracted:
                If true, return (cell, date, epi) in contracted format
                if false, return (cell, date, month, year, series, epi)
        """
        if regexfmt is None: regexfmt = self.episoderegex
        cell, date, month, year, series, epi = re.findall(regexfmt, name)[0]
        if contracted:
            date = '%s%s%s'%(date, month, year)
            epi = 'S%sE%s'%(series, epi)
            return(cell, date, epi)
        else:
            return(cell, date, month, year, series, epi)
            
        
    def transverse_basedir(self, filt=None, db_dir=None):
        """Transverse through the data structure and locate data"""
        dir_tree = []
        for root, dirnames, filenames in os.walk(basedir):
            for filename in fnmatch.filter(filenames,'*.dat'):
                dir_tree.append(os.path.join(root, filename))
        # Initialize the database
        if db_dir is None:
            db_dir = os.path.join(os.getcwd(), 'temp-database-%s.db'%(
                datetime.datetime.strftime(datetime.datetime.now(), 
                '%Y-%m-%d %H:%M:%S')))
        self.make_db(db_dir)
        # Create the table
        table = []
        for f in dir_tree:
            # load everything
            zData = NeuroData(dataFile=f, old=True, infoOnly=False)
            # get metdata
            name = os.path.splitext(os.path.basename(os.path.abspath(f)))[0]
            cell, date, epi = self.parse_name(name, contracted=True)
            wctime = zData.Protocol.WCtime
            duration = zData.Protocol.sweepWindow
            drug_level = int(zData.Protocol.drug)
            drug_name = zData.Protocol.drugName
            drug_time = zData.Protocol.drugTime
            acq_rate_khz = 1.0/zData.Protocol.msPerPoint
            channels = ",".join(zData.Protocol.channelNames)
            holding_current =",".join([k+":"+str(v[0]) for k, v in iter(zData.Current.items())])
            RMP = ",".join([k+":"+str(v[0]) for k, v in iter(zData.Voltage.items())])
            comment=""
            # write the table
            table.append((name, date, cell, epi, wctime, duration, drug_level,\
                          drug_name, drug_time, acq_rate_khz, channels, \
                          holding_current, RMP, comment))
        # write the database
        self.update_db(db_dir, table, ncols=14)
        
        
    def make_db(self, db_dir):
        """Make a new SQL database"""
        # check existence
        if os.path.exists(db_dir): 
            print('Warning! %s already existed'%(db_dir))
            return
        # initialize a databse
        if 'db' in self.__dict__.keys():
            self.db.append(db_dir)
        else:
            self.db = [db_dir]
        # open the database
        conn = sqlite3.connect(db_dir)
        c = conn.cursor()
        # create the table
        c.execute('''CREATE TABLE Experiments
                        Name  TEXT    PRIMARY KEY     NOT NULL, 
                        Date TEXT, Cell TEXT, Episode TEXT, WCTime TEXT, 
                        Duration REAL, Drug_Level INT, Drug_Name INT, 
                        Drug_Time TEXT,Acq_Rate_kHz REAL, Channels TEXT, 
                        Holding_Current REAL, RMP REAL, Comments TEXT''')
        conn.commit()
        conn.close()
        
            
    def update_db(self, db_dir, table, ncols=None):
        """Update SQL database
        Write contents in 'table' into database at 'db_dir'
        """
        if ncols is None: ncols = len(table[0])
        conn = sqlite3.connect(db_dir)
        c = conn.cursor()
        c.executemany('INSERT INTO Experiments VALUES (' + \
                        ','.join(['?']*ncols)  + ')' , table)
        conn.commit()
        conn.close()

        
    def query_db(self, db_dir, criterion=None):
        """Query the databse by criterion (a dictionary)"""
        if criterion is None: return
        
    def db2pd(self, db):
        """Convert database table to panda dataframe"""
        return
        
dataFile='D:/Data/2015/07.July/Data 6 Jul 2015/Neocortex D.06Jul15.S1.E4.dat'
data = NeuroData(dataFile=dataFile, old=True, infoOnly=False)
"""
if __name__ == '__main__':
    K = EpisodeDB()
    A = K.get_full_path('Neocortex D.06Jul15.S1.E4.dat')
"""