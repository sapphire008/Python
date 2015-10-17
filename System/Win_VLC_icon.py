# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 17:56:32 2015
Change VLC multimedia icons
@author: Edward
"""

import subprocess
import re

# Command to list registry key queries
cmd = "REG QUERY HKCR /f VLC."
# Start the process and call Command
VLC = subprocess.check_output(cmd)
# filter out invalid results
VLC = filter(None,VLC.split('\r\n'))
VLC = [x for x in VLC if 'VLC' in x]
# Get a list of supported extension
VLCext = [re.findall('VLC.(\w+)', x) for x in VLC]
VLCext = [a[0] for a in VLCext]

# Command to change the icon
cmd = "REG ADD %s\\DefaultIcon /f /v (Default)  /t REG_SZ /d %s"

# Get a list of icon files: needs modification
iconfile=["VLC."+a+".ico" for a in VLCext]

for n, ext in enumerate(VLCext):
    # Get the corresponding icon file
    ico = [i for i in iconfile if ext in i]
    # Reassign the icon file name
    subprocess.call(cmd%(VLC[n], ico))