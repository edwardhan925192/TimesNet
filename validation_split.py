import numpy as np 
import pandas as pd 

def split_train_validation_timeseries(df, validation_ranges, target_column, seq_len):
    '''
    Takes a DataFrame and a list of validation ranges (e.g., [(800, 900), (900, 1000)])
    Suppose sequence length is 100 then it returns 700 - 1000 as validation set so that the first prediction will be inside validation set 
    0 - 800 as training set for the first set 
    

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
        adjusted_start = start - seq_len  # Adjust start for validation

        if adjusted_start < 0 or start >= end or end > len(df):
            raise ValueError("Invalid range with adjusted start: {}-{}".format(adjusted_start, end))

        # Split the DataFrame
        validation_df = df.iloc[adjusted_start:end]  # Return the entire DataFrame for the range
        train_df = df.iloc[:start]

        train_dfs.append(train_df)
        validation_dfs.append(validation_df)

    return train_dfs, validation_dfs

