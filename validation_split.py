import numpy as np 
import pandas as pd 

def split_train_validation_timeseries(df, validation_ranges, seq_len):
    '''
    Takes a DataFrame and a list of validation ranges (e.g., [(800, 900), (900, 1000)])
    Validation is going to be the Range that is takes MINUS the sequence length e.g, (700,900) 
    if Seq_length is 100 and the range is from (800, 900)

    Parameters:
    df (DataFrame): The DataFrame to split.
    validation_ranges (list of tuples): The ranges for validation data.
    target_column (str): The name of the target column in the DataFrame.
    seq_len (int): The sequence length.
    single (bool): Whether to return only the target column or the entire DataFrame.

    Returns:
    tuple: A tuple containing two lists of DataFrames or Series, one for training and one for validation.
    
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
        validation_df = df.iloc[adjusted_start:end]
        train_df = df.iloc[:start]        

        train_dfs.append(train_df)
        validation_dfs.append(validation_df)

    return train_dfs, validation_dfs
