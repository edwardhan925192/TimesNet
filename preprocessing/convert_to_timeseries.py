# -*- coding: utf-8 -*-
"""convert_to_timeseries.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VzrOuEKuEZs16Oq4SP1lvX_cp4sVf-3d
"""

def convert_to_timeseries(df, date_columns, unique_identifier, target_column):
    """
    Converts a DataFrame into a time series-like format.

    :param df: pandas DataFrame
    :param date_columns: List of columns representing the datetime (Year, Month, Day, Hour)
    :param unique_identifier: Column name of the unique identifier
    :param target_column: Column name of the target values
    :return: Transformed DataFrame
    """
    # Combine the date columns into a single datetime column
    df['datetime'] = pd.to_datetime(df[date_columns])

    # Create a pivot table
    time_series_df = df.pivot_table(index='datetime',
                                    columns=unique_identifier,
                                    values=target_column,
                                    aggfunc='sum')

    # Rename the columns
    time_series_df.columns = [f"{col}_{target_column}" for col in time_series_df.columns]

    return time_series_df