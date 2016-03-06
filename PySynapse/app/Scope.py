# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 02:33:37 2016

Trace scope window

@author: Edward
"""

class Scope(object):
    pass
    def __init__(self, episode, index=[]):
        print('scope launching')
        return
        
        
if __name__ == '__main__':
    episode = {'Drug Level': 0, 'Comment': 'DAC0: PulseB -75', \
               'Name': 'Neocortex B.08Feb16.S1.E4', \
               'Dirs': 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex B.08Feb16.S1.E4.dat',\
               'Epi': 'S1.E4', 'Time': '29.1 sec', 'Duration': 4000}
    Scope(episode)