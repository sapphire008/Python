# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

correct bibtex reference keys
"""


from optparse import OptionParser
import os

# input_file = 'D:/Edward/Documents/Assignments/Case Western Reserve/StrowbridgeLab/Projects/TeA Persistence Cui and Strowbridge 2015/docs/ReferencesBibTex.bib'

usage = "usage: %prog [options] input_file"
parser = OptionParser(usage)
parser.add_option("-o","--output", dest="output_file", help="output file. Default overwrite input", default=None)
options, args = parser.parse_args()

# parse input
input_file = args[0]

# parse output
if options.output_file is None:
    output_file = os.path.join(os.path.dirname(input_file), 'tmp.bib').replace('\\','/')
else:
    output_file = options.output_file

# open the files
fidi = open(input_file, 'r')
fido = open(output_file, 'w')

# correct the file
for row in fidi:
    if '@' == row[0]:
        row = row.lower()
    # if '-' in row[0]:
        # row.replace('-','')
    fido.write(row)

# close the file
fidi.close()
fido.close()

# replace old file if necessary
if options.output_file is None:
    os.remove(input_file)
    os.rename(output_file, input_file)
