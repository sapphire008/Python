# -*- coding: utf-8 -*-

# last revised 28 July 2015 BWS

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os.path as path
import sys
import FileIO.ProcessEpisodes as PE 

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("You need to specifiy the keyValue and the two input csv files to be merged")
    else:
        if not path.isfile(sys.argv[2]):
            print("The specified file does not exist: " + sys.argv[2])
        else:
            if not path.isfile(sys.argv[3]):
                print("The specified file does not exist: " + sys.argv[3])
            else:
                # call is keyValue ("Expt"), first csv file, second csv file
                PE.mergeCSVfiles(sys.argv[1], sys.argv[2], sys.argv[3])
