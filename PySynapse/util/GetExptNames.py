#!/usr/bin/python
# -*- coding: utf-8 -*-

# revised 26 July 2015 BWS

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
import os.path as path
import shutil as shutil
import csv as csv
import FileIO.ProcessEpisodes as PE
import Util.CommonFunctions as Common

def getExptNames(INIfileRoot):
    INIfileRoot = Common.testINIfile(INIfileRoot)
    if INIfileRoot:
        parserObj = PE.readEpisodeList(INIfileRoot)
        return  PE.getListOfNonexcludedExpts(parserObj)
    else:
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("You need to supply the INI file name to run getExptNames.py")
    else:
        getExptNames(sys.argv[1])
