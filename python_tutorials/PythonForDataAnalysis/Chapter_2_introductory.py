# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 01:34:26 2014

@author: Edward
"""

# Working example
import json
path = 'C:\Users\Edward\Documents\Assignments\Python\python_tutorials\data_analysis_tutorial\usagov_bitly_data2013-05-17-1368832207.txt'
records = [json.loads(line) for line in open(path)]

# Counting time zone in pure Python
time_zones = [rec['tz'] for rec in records if 'tz' in rec]
def get_counts(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] +=1
        else:
            counts[x]=1
    return counts

from collections import defaultdict
def get_counts2(sequence):
    counts = defaultdict(int)# values will initialize to 0
    for x in sequence:
        counts[x] +=1
    return counts
    
counts = get_counts(time_zones)

# count top 10 time zones
def top_counts(count_dict,n=10):
    value_key_pairs=[(count,tz) for tz, count in count_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]
    
top_counts(counts)

from collections import Counter
counts = Counter(time_zones)
counts.most_common(10)

# Count time zones with Panda
from pandas import DataFrame, Series
frame = DataFrame(records)
tz_counts = frame['tz'].value_counts()

clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz==''] = 'Unknown'
tz_counts = clean_tz.value_counts()
# plot
tz_counts[:10].plot(kind='barh',rot=0)

results = Series([x.split()[0] for x in frame.a.dropna()])
results.value_counts()[:8]

# split into Windows vs non-Windows users
import numpy as np
cframe = frame[frame.a.notnull()]
operating_system = np.where(cframe['a'].str.contains('Windows'),
                            'Windows','Not Windows')
by_tz_os = cframe.groupby(['tz',operating_system])
agg_counts = by_tz_os.size().unstack().fillna(0)
indexer = agg_counts.sum(1).argsort()
count_subset = agg_counts.take(indexer)[-10:]
count_subset.plot(kind='barh',stacked=True)
# normalized to 1 and plot again
normed_subset = count_subset.div(count_subset.sum(1),axis=0)
normed_subset.plot(kind='barh',stacked=True)


# This concludes today's study