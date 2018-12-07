#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 19:31:29 2018

@author: edward
"""

import os
import numpy as np


def readString(fid, strlength):
    tmp = np.fromfile(fid, '|S1', strlength)
    return np.ndarray.tostring(tmp).decode('UTF-8')


class Header(object):
    pass

class EDF(object):
    """Load a EDF file"""
    def __init__(self, filepath=None, header_only=False):
        self.filepath = filepath
        self.header = Header()
        
        if filepath is not None and isinstance(filepath, str) and os.path.isfile(filepath):
            self.loadData(filepath, header_only)
        else:
            raise(Exception('Invalid EDF file'))
    
    def loadData(self, filepath, header_only=False):
        with open(filepath, 'rb') as fid:
            self.header.version = readString(fid, 8)
            self.header.patient_id = readString(fid, 8)
            fid.seek(17)
            self.header.gender = readString(fid, 1)
            fid.seek(19)
            self.header.DOB = readString(fid, 11)
            fid.seek(44)
            self.header.age = int(readString(fid, 2))
            fid.seek(98)
            self.header.start_date_entered = readString(fid, 11)
            fid.seek(110)
            self.header.eeg_id = readString(fid, 13).strip()
            fid.seek(124)
            self.header.tech = readString(fid, 2)
            fid.seek(127)
            self.header.machine = readString(fid, 20).strip()
            fid.seek(167)
            self.header.start_date = readString(fid, 8)
            self.header.start_time = readString(fid, 8)
            self.header.header_size = int(readString(fid, 8).strip())
            self.header.file_type = readString(fid, 5).strip()
            fid.seek(235)
            self.header.num_records = int(readString(fid, 8).strip())
            self.header.duration = float(readString(fid, 8).strip())
            self.header.num_signals = int(readString(fid, 4).strip())
            self.header.labels = [[]]*self.header.num_signals
            for n in range(self.header.num_signals):
                self.header.labels[n] = readString(fid, 16).strip()
                
            self.header.trans_type = [[]]*self.header.num_signals
            for n in range(self.header.num_signals):
                self.header.trans_type[n] = readString(fid, 80).strip()
                
            self.header.phys_dim = [[]]*self.header.num_signals
            for n in range(self.header.num_signals):
                self.header.phys_dim[n] = readString(fid, 8).strip()
            
            fid.seek(fid.tell()+1)
            self.header.phys_min = [[]]*self.header.num_signals
            for n in range(self.header.num_signals):
                self.header.phys_min[n] = float(readString(fid, 8).strip())
        
            self.header.phys_max = [[]]*self.header.num_signals
            for n in range(self.header.num_signals):
                self.header.phys_max[n] = float(readString(fid, 8).strip())
                
            self.header.dig_min = [[]]*self.header.num_signals
            for n in range(self.header.num_signals):
                self.header.dig_min[n] = int(readString(fid, 8).strip())
                
            self.header.dig_max = [[]]*self.header.num_signals
            for n in range(self.header.num_signals):
                self.header.dig_max[n] = int(readString(fid, 8).strip())
                
            self.header.prefilt = [[]]*self.header.num_signals
            for n in range(self.header.num_signals):
                self.header.prefilt[n] = readString(fid, 80).strip()
                
            self.header.sample_rate = [[]]*self.header.num_signals
            for n in range(self.header.num_signals):
                self.header.sample_rate[n] = int(readString(fid, 8).strip())
                
            if header_only:
                self.data = []
            else:
                fid.seek(fid.tell()+32 * self.header.num_signals) 
                # Loading all the data for later organization
                Ch_data = np.fromfile(fid, np.int16).reshape(-1, self.header.num_records, order='F')
                Rs = np.cumsum(np.insert(int(self.header.duration)*np.array(self.header.sample_rate), 0, 0))
                self.data = [[]] * self.header.num_signals
                sf = (np.array(self.header.phys_max)- np.array(self.header.phys_min)) / (np.array(self.header.dig_max) - np.array(self.header.dig_min))
                dc = np.array(self.header.phys_max) - sf * np.array(self.header.dig_max)
                for n in range(self.header.num_signals):
                    self.data[n] = Ch_data[Rs[n]:(Rs[n+1])].flatten(order='F') * sf[n] + dc[n]
    


if __name__ == '__main__':
    filepath = '/Volumes/Storage/TUH_EEG/train/01_tcp_ar/009/00000906/s007_2003_04_28/00000906_s007_t000.edf'
    filepath = '/Volumes/Storage/TUH_EEG/train/01_tcp_ar/002/00000254/s005_2010_11_15/00000254_s005_t000.edf'
    edf = EDF(filepath)
    