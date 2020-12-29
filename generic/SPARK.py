# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:25:03 2020

Snippets for Spark

@author: cudo9001
"""
import os
import pandas as pd
import dask as dd
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# Union all dataframes
def expr(mycols, allcols):
    """
    Fill missing columns with nan (map)
    For unionAll(columns, df1, df2, df3)
    """
    def processCols(colname):
        if colname in mycols:
            return colname
        else:
            return F.lit(None).alias(colname)
    cols = map(processCols, allcols)
    return list(cols)

def unionAll(columns, *dfs):
    """
    Union all dataframes (reduce)
    """
    return reduce(DataFrame.union, 
        [df.select(expr(df.columns, columns)) for df in dfs])

# Faster and more memory efficient toPandas
def _map_to_pandas(rdds):
    """ Needs to be here due to pickling issues """
    return [pd.DataFrame(list(rdds))]

def toPandas(df, n_partitions=None):
    """
    Returns the contents of `df` as a local `pandas.DataFrame` in a 
    speedy fashion. The DataFrame is repartitioned if `n_partitions` is passed.
    :param df:              pyspark.sql.DataFrame
    :param n_partitions:    int or None
    :return:                pandas.DataFrame
    """
    if n_partitions is not None: df = df.repartition(n_partitions)
    df_pand = df.rdd.mapPartitions(_map_to_pandas).collect()
    df_pand = pd.concat(df_pand)
    df_pand.columns = df.columns
    return df_pand

# Spark to Dask dataframe
def _map_to_dask(rdds):
    return [dd.DataFrame(list(rdds))]

def toDask(df, n_partitions=None):
    """
    Returns the content of `df` as a Dask Dataframe in a speedy fashion.
    The DataFrame is repartitioned if `n_partitions` is passed.
    :param df:              pyspark.sql.DataFrame
    :param n_partitions:    int or None
    :return:                Dask dataframe
    """
    raise(NotImplementedError('This method has not been fully implemented'))
    if n_partitions is not None: df = df.repartition(n_partitions)
    df_dd = df.rdd.mapPartitions(_map_to_dask).collect()
    return df_dd
    

def countNullCols(df, cols=[]):
    """
    Reduce for df.where(F.col(x).isNull()).count()

    """
    if len(cols) < 1: # All columns
        return df.where(reduce(lambda x, y: x | y, \
            (F.col(x).isNull() for x in df.columns))).count()
    elif len(cols) == 1:
        return df.where(F.col(cols[0]).isNull()).count()
    else: # len(cols)>1:
        return df.where(reduce(lambda x, y: x | y, \
            (F.col(x).isNull() for x in cols))).count()
    
from pyspark.sql.types import *

# Auxiliar functions
# Pandas Types -> Sparks Types
def equivalent_type(f):
    if f == 'datetime64[ns]':
        return DateType()
    elif f == 'int64':
        return LongType()
    elif f == 'int32':
        return IntegerType()
    elif f == 'float64':
        return FloatType()
    else:
        return StringType()

def define_structure(string, format_type):
    try: 
        typo = equivalent_type(format_type)
    except: 
        typo = StringType()
    return StructField(string, typo)

#Given pandas dataframe, it will return a spark's dataframe
def pandas_to_spark(df_pandas):
    columns = list(df_pandas.columns)
    types = list(df_pandas.dtypes)
    struct_list = []
    for column, typo in zip(columns, types): 
        struct_list.append(define_structure(column, typo))
    p_schema = StructType(struct_list)
    return sqlContext.createDataFrame(df_pandas, p_schema)
