# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 19:20:50 2016

Convenient R functions in Python

@author: Edward
"""

import numpy as np
import pandas as pd
from pdb import set_trace
import psutil
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm

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
    """
    select: columns to keep
    subset: rows to keep
    """
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

def filterAND(df, AND, select=None):
    """
    Filter rows of a dataframe based on the values in the specified columns

    AND: A dictionary, where the keys are the column names, and values are the
         the corresponding values of the column to apply AND filter on
    select: subset of columns to use
    """
    AND_filter = []
    for key, val in AND.items():
        if isinstance(val, (list, tuple, np.ndarray)):
            for v in val:
                AND_filter.append((df[key] == v).values)
        else:
            AND_filter.append((df[key] == val).values)
            
    if select is None:
        if len(AND_filter) > 1:
            return df.loc[np.logical_and.reduce(AND_filter), :]
        else:
            return df.loc[AND_filter[0], :]
    else:
        if len(AND_filter) > 1:
            return df.loc[np.logical_and.reduce(AND_filter), select]
        else:
            return df.loc[AND_filter[0], select]


def filterOR(df, OR, select=None):
    """
    Filter rows of a dataframe based on the values in the specified columns

    OR: A dictionary, where the keys are the column names, and values are the
         the corresponding values of the column to apply OR filter on
    select: subset of columns to use
    """
    OR_filter = []
    try:
        for key, val in OR.items():
            if isinstance(val, (list, tuple, np.ndarray)):
                for v in val:
                    OR_filter.append((df[key] == v).values)
            else:
                OR_filter.append((df[key] == val).values)
        if select is None:
            if len(OR_filter) > 1:
                return df.loc[np.logical_or.reduce(OR_filter), :]
            else:
                return df.loc[OR_filter[0], :]
        else:
            if len(OR_filter) > 1:
                return df.loc[np.logical_or.reduce(OR_filter), select]
            else:
                return df.loc[OR_filter[0], select]
    except:
        set_trace()


def filterRows(df, AND=None, OR=None, AndBeforeOr=True, select=None):
    """
    Filter rows of a dataframe based on the values in the specified columns

    AND: A dictionary, where the keys are the column names, and values are the
         the corresponding values of the column to apply AND filter on
    OR:  A dictionary like And, except to apply OR filter
    AndBeforeOr: If true (Default), apply AND filter before OR filter
    select: subset of columns to use
    """
    # Make a copy of the df first
    dff = df.copy()
    if AndBeforeOr:
        # Apply AND filter
        dff = filterAND(dff, AND, select=select)
        # Apply OR filter
        dff = filterOR(dff, OR, select=select)
    else:
        # Apply AND filter
        dff = filterOR(dff, OR, select=select)
        # Apply OR filter
        dff = filterAND(dff, AND, select=select)
        
    return dff


def merge_size(left_frame, right_frame, on, how='inner'):
    """
    Check memory usage of a merge
    """
    left_groups = left_frame.groupby(on).size()
    right_groups = right_frame.groupby(on).size()
    left_keys = set(left_groups.index)
    right_keys = set(right_groups.index)
    intersection = right_keys & left_keys
    left_diff = left_keys - intersection
    right_diff = right_keys - intersection

    left_nan = len(left_frame[left_frame[on] != left_frame[on]])
    right_nan = len(right_frame[right_frame[on] != right_frame[on]])
    left_nan = 1 if left_nan == 0 and right_nan != 0 else left_nan
    right_nan = 1 if right_nan == 0 and left_nan != 0 else right_nan

    sizes = [(left_groups[group_name] * right_groups[group_name]) for group_name in intersection]
    sizes += [left_nan * right_nan]

    left_size = [left_groups[group_name] for group_name in left_diff]
    right_size = [right_groups[group_name] for group_name in right_diff]
    if how == 'inner':
        return sum(sizes)
    elif how == 'left':
        return sum(sizes + left_size)
    elif how == 'right':
        return sum(sizes + right_size)
    return sum(sizes + left_size + right_size)

def mem_fit(df1, df2, on, how='inner'):
    """
    Check if a merge would fit in the memory
    """
    rows = merge_size(df1, df2, on, how)
    cols = len(df1.columns) + (len(df2.columns) - 1)
    required_memory = (rows * cols) * np.dtype(np.float64).itemsize

    return required_memory <= psutil.virtual_memory().available


def applyParallel(dfGrouped, func):
    """
    def apply_func(pandas_df):
        ...
        
    df = applyParallel(df.groupby(by=grouped_by_columns, as_index=False), apply_func)
    """
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(\
            delayed(func)(group) for name, group in tqdm(dfGrouped)) # enumerate(tqdm(dfGrouped))
    return pd.concat(retLst)


def reduce_mem_usage(df, int_cast=True, obj_to_category=False, subset=None):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    :param df: dataframe to reduce (pd.DataFrame)
    :param int_cast: indicate if columns should be tried to be casted to int (bool)
    :param obj_to_category: convert non-datetime related objects to category dtype (bool)
    :param subset: subset of columns to analyse (list)
    :return: dataset with the column dtypes adjusted (pd.DataFrame)
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2;
    gc.collect()
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    cols = subset if subset is not None else df.columns.tolist()

    for col in tqdm(cols):
        col_type = df[col].dtype

        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()

            # test if column can be converted to an integer
            treat_as_int = str(col_type)[:3] == 'int'
            if int_cast and not treat_as_int:
                treat_as_int = check_if_integer(df[col])

            if treat_as_int:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                    df[col] = df[col].astype(np.uint64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif 'datetime' not in col_type.name and obj_to_category:
            df[col] = df[col].astype('category')
    gc.collect()
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.3f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
