from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
import pandas as pd



def apply_parallel(df_grouped, func):
    """
    Apply a function to grouped pandas dataframe using multiprocessing
    def apply_func(pandas_df):
        ...

    df = apply_parallel(df.groupby(by=grouped_by_columns, as_index=False), apply_func)
    """
    ret = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(func)(group) for name, group in tqdm(df_grouped)
    )  # enumerate(tqdm(dfGrouped))
    return pd.concat(ret)
