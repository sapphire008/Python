# -*- coding: utf-8 -*-

# last revised 18 July 2015 BWS

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os.path as path
import sys
import FileIO.ProcessEpisodes as PE 

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("You need to specifiy the input csv file to be summarized")
    else:
        if not path.isfile(sys.argv[1]):
            print("The specified file does not exist: " + sys.argv[1])
        else:
            PE.createSummary(sys.argv[1])
