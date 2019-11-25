# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 19:20:50 2016

Convenient R functions in Python

@author: Edward
"""

import numpy as np
import pandas as pd
from pdb import set_trace

def aggregate(df, by, fun, select=None, subset=None, **kwargs):
    """This should enhance the stupidly designed groupby functions.
    Will have to evoke this until it is fixed.
    
    df: pandas data frame
    by: columns to aggregate by
    fun: function(s) to apply. For examples:
        - fun = {'mean': np.mean, 'variance', np.var} will create 2 columns
          in the aggregated dataframe, 'mean' and 'variance', which stores
          the results of each aggregation
        - fun = ['sum', 'count', custom_function], apply the pandas built-in
          sum and count, as well as a custom_function defined by the user.
          The column aggregated by custom_function will be named 
          'custom_function'
    select: select columns to aggregate on, exclude other columns
    subset: select rows to aggregate on.
    """
    if 'DataFrameGroupBy' not in str(type(df)):
        for b in by:
            df[b] = df[b].astype('category')
        
        df = subset(df, select, subset)
        gp = df.groupby(by)
    
    gp = gp.agg(fun)#, **kwargs)
    # Remove any nuisance rows with all nans
    gp = subset(gp, subset=~np.all(np.isnan(gp.values), axis=1))

    return gp

def subset(df, select=None, subset=None):
    if select is not None:
        df = df[select]
    
    if select is not None:
        df = df.loc[subset, :]

    return df

def filterByCount(df, N=2, by=None, by2=None, keep_count=False):
    """
    Remove the rows if the total number of rows associated with the aggregate
    is less than a threshold N
    by: column to aggregate by
    by2: aggreagte on top of the aggregation (summary of the summary)
    keep_count: keep the count column
    """
    if not isinstance(by, list):
        raise(TypeError('parameter "by" must be a list'))
    df['filtered_count'] = 0
    gp = df.groupby(by=by, as_index=False, sort=False)
    ns0 = gp.count()
    ns0 = ns0[by+['filtered_count']]
    df.drop(columns=['filtered_count'], inplace=True)
    if by2 is not None:
        gp = ns0.groupby(by=by2, as_index=False, sort=False)
        ns1 = gp.count()
        ns1.drop(columns=np.setdiff1d(by, by2), inplace=True)        
        df = df.merge(ns1, on=by2)
    else:
        df = df.merge(ns0, on=by)
    df = df.loc[df['filtered_count']>=N,:]
    if not keep_count:
        df.drop(columns=['filtered_count'], inplace=True)
    return df

def filter_logicals(df, conditions, logicals='and'):
    """
    filter by logical and
    conditions: dictionary of col:value
    """
    cond_list = [[]] * len(conditions)
    for n, (key, val) in enumerate(conditions.items()):
        cond_list[n] = df[key] == val
    
    if logicals == 'and':
        df = df.loc[np.logical_and.reduce(cond_list)]
    elif logicals == 'or':
        df = df.loc[np.logical_or.reduce(cond_list)]
    return df





