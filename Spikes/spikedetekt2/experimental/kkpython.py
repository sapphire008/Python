# Code example showing how to create temporary .fet and .fmask files from
# files in the new format.
#
# Arguments: 
#   * channel_group: the channel group index to process
#   * filename: the filename of the KWIK file
#   * params: a dictionary with all KK parameters

import os
import shutil
import tempfile
from spikedetekt2.dataio import Experiment

# get the basename (filename without the extension)
basename = os.path.splitext(filename)[0]

# Create a temporary working folder where we're going to run KK.
tmpdir = tempfile.mkdtmp()
curdir = os.getpwd()
os.chdir(tmpdir)

# Create the filenames of the .fet and .fmask files to create.
filename_fet = os.path.join(tmpdir, basename + '.fet')
filename_fmask = os.path.join(tmpdir, basename + '.fmask')
filename_clu = os.path.join(tmpdir, basename + '.clu')

with Experiment(filename) as exp:  # Open in read-only, close the file at the end of the block
    # Load all features and masks in memory.
    # WARNING: this might consume to much Ram ==> need to do it by chunks.
    fm = exp.channel_groups[channel_group].spikes.features_masks[:]
    # fm is a Nspikes x Nfeatures x 2 array (features AND masks)
    fet = fm[:,:,0]
    fmask = fm[:,:,1]
    # Convert to .fet and .fmask.
    # These functions are in (old) spikedetekt.files
    write_fet(fet, filename_fet)
    write_mask(fmask, filename_fmask, fmt="%f")

# Sort out the KK parameters.
opt = ' '.join(['-{k}={v}'.format(k=k, v=v) for k, v in params.iteritems()])

# Call KK
os.system("klustakwik {fn} {opt}".format(fn=basename, opt=opt))

# Read the .clu file.
clu = read_clu(filename_clu)

# Add the clusters to the KWIK file.
with Experiment(filename, mode='a') as exp:
    exp.channel_groups[channel_group].spikes.clusters.original[:] = clu

# Delete the temporary folder.
shutil.rmdir(tmpdir)

# Get back to the original folder.
os.chdir(curdir)
