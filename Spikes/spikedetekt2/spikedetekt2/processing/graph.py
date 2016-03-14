"""Graph routines."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from itertools import izip

import numpy as np


# -----------------------------------------------------------------------------
# Component class
# -----------------------------------------------------------------------------
class Component(object):
    def __init__(self, items=None, chunk=None, 
                 s_start=0, keep_start=0, keep_end=np.inf,
                 recording=0):
        self.items = items
        self.chunk = chunk
        if not chunk:
            self.s_start = s_start
            self.keep_start = keep_start
            self.keep_end = keep_end
            self.recording = recording
        else:
            self.s_start = chunk.s_start
            self.keep_start = chunk.keep_start
            self.keep_end = chunk.keep_end
            self.recording = chunk.recording
    
    def __repr__(self):
        return '<Component {0:s}>'.format(self.items)

        
# -----------------------------------------------------------------------------
# Graph
# -----------------------------------------------------------------------------
def _to_tuples(x):
    return ((i, j) for (i, j) in x)
    
def _to_list(x):
    return [(i, j) for (i, j) in x]

def connected_components(chunk_weak=None, chunk_strong=None, 
                         chunk=None, probe_adjacency_list=None, 
                         return_objects=True,
                         **prm):
    """
    Return a list of pairs (samp, chan) of the connected components in the 2D
    array chunk_weak, where a pair is adjacent if the samples are within join_size of
    each other, and the channels are adjacent in probe_adjacency_list, the channel graph.
    
    Arguments:
    
    * chunk_weak: Nsamples x Nchannels array with weak threshold crossings
    * chunk_strong: Nsamples x Nchannels array with strong threshold crossings
    * probe_adjacency_list: a dict {channel: [neighbors]}
    * join_size: the number of samples defining the tolerance in time for
      finding connected components
    
    """
    
    join_size = prm.get('join_size', prm.get('connected_component_join_size', 0))
    
    if probe_adjacency_list is None:
        probe_adjacency_list = {}
    
    if chunk_strong is None:
        chunk_strong = chunk_weak
    
    assert chunk_weak.shape == chunk_strong.shape
    
    # set of connected component labels which contain at least one strong 
    # node
    strong_nodes = set()
    
    n_s, n_ch = chunk_weak.shape
    join_size = int(join_size)
    
    # an array with the component label for each node in the chunk
    label_buffer = np.zeros((n_s, n_ch), dtype=np.int32)
    
    # component indices, a dictionary with keys the label of the component
    # and values a list of pairs (sample, channel) belonging to that component  
    comp_inds = {}
    # mgraph is the channel graph, but with edge node connected to itself
    # because we want to include ourself in the adjacency. Each key of the
    # channel graph (a dictionary) is a node, and the value is a set of nodes
    # which are connected to it by an edge
    mgraph = {}
    for source, targets in probe_adjacency_list.iteritems():
        # we add self connections
        mgraph[source] = targets.union([source])
    # label of the next component
    c_label = 1
    # for all pairs sample, channel which are nonzero (note that numpy .nonzero
    # returns (all_i_s, all_i_ch), a pair of lists whose values at the
    # corresponding place are the sample, channel pair which is nonzero. The
    # lists are also returned in sorted order, so that i_s is always increasing
    # and i_ch is always increasing for a given value of i_s. izip is an
    # iterator version of the Python zip function, i.e. does the same as zip
    # but quicker. zip(A,B) is a list of all pairs (a,b) with a in A and b in B
    # in order (i.e. (A[0], B[0]), (A[1], B[1]), .... In conclusion, the next
    # line loops through all the samples i_s, and for each sample it loops
    # through all the channels.
    for i_s, i_ch in izip(*chunk_weak.nonzero()):
        # the next two lines iterate through all the neighbours of i_s, i_ch
        # in the graph defined by graph in the case of edges, and
        # j_s from i_s-join_size to i_s.
        for j_s in xrange(i_s-join_size, i_s+1):
            # allow us to leave out a channel from the graph to exclude bad
            # channels
            if i_ch not in mgraph:
                continue
            for j_ch in mgraph[i_ch]:
                # label of the adjacent element
                adjlabel = label_buffer[j_s, j_ch]
                # if the adjacent element is nonzero we need to do something
                if adjlabel:
                    curlabel = label_buffer[i_s, i_ch]
                    if curlabel==0:
                        # if current element is still zero, we just assign
                        # the label of the adjacent element to the current one
                        label_buffer[i_s, i_ch] = adjlabel
                        # and add it to the list for the labelled component
                        comp_inds[adjlabel].append((i_s, i_ch))

                    elif curlabel!=adjlabel:
                        # if the current element is unequal to the adjacent
                        # one, we merge them by reassigning the elements of the
                        # adjacent component to the current one
                        # samps_chans is an array of pairs sample, channel
                        # currently assigned to component adjlabel
                        samps_chans = np.array(comp_inds[adjlabel], dtype=np.int32)
                        # samps_chans[:, 0] is the sample indices, so this
                        # gives only the samp,chan pairs that are within
                        # join_size of the current point
                        # TODO: is this the right behaviour? If a component can
                        # have a width bigger than join_size I think it isn't!
                        samps_chans = samps_chans[i_s-samps_chans[:, 0]<=join_size]
                        # relabel the adjacent samp,chan points with current
                        # label
                        samps, chans = samps_chans[:, 0], samps_chans[:, 1]
                        label_buffer[samps, chans] = curlabel
                        # add them to the current label list, and remove the
                        # adjacent component entirely
                        comp_inds[curlabel].extend(comp_inds.pop(adjlabel))
                        # did not deal with merge condition, now fixed it seems...
                        # WARNING: might this "in" incur a performance hit here...?
                        if adjlabel in strong_nodes:
                            strong_nodes.add(curlabel)
                            strong_nodes.remove(adjlabel)
                        
                    # NEW: add the current component label to the set of all
                    # strong nodes, if the current node is strong
                    if curlabel > 0 and chunk_strong[i_s, i_ch]:
                        strong_nodes.add(curlabel) 
                  
        if label_buffer[i_s, i_ch]==0:
            # if nothing is adjacent, we have the beginnings of a new component,
            # so we label it, create a new list for the new component which is
            # given label c_label, then increase c_label for the next new
            # component afterwards
            label_buffer[i_s, i_ch] = c_label
            comp_inds[c_label] = [(i_s, i_ch)]
            if chunk_strong[i_s, i_ch]:
                strong_nodes.add(c_label)
            c_label += 1
            
    # only return the values, because we don't actually need the labels
    comps = [comp_inds[key] for key in comp_inds.keys() if key in strong_nodes]
    if return_objects:
        return [Component(comp, chunk=chunk)
                    for comp in comps]
    else:
        return comps
    