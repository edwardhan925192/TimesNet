import numpy as np 
import pandas as pd 

def split_train_validation_timeseries(df, validation_ranges, target_column, configs):
    '''
    Takes a DataFrame and a list of validation ranges (e.g., [(800, 900), (900, 1000)])
    and returns rows previous to the validation range as training data and the target column
    of the validation range as validation data, starting from range_start - configs.pred_len.

    Parameters:
    df (DataFrame): The DataFrame to split.
    validation_ranges (list of tuples): The ranges for validation data.
    target_column (str): The name of the target column in the DataFrame.
    configs: Configuration object that includes pred_len.

    Returns:
    tuple: A tuple containing two lists of DataFrames, one for training and one for validation.
    '''
    train_dfs = []
    validation_dfs = []

    for val_range in validation_ranges:
        # Ensure the range is valid
        start, end = val_range
        adjusted_start = start - configs.pred_len  # Adjust start for validation

        if adjusted_start < 0 or start >= end or end > len(df):
            raise ValueError("Invalid range with adjusted start: {}-{}".format(adjusted_start, end))

        # Split the DataFrame
        validation_df = df.iloc[adjusted_start:end][target_column]
        train_df = df.iloc[:adjusted_start]

        train_dfs.append(train_df)
        validation_dfs.append(validation_df)

    return train_dfs, validation_dfs
