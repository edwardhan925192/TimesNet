import pandas as pd

def convert_to_timeseries(df, date_columns, unique_identifier, target_column):
    """
    Converts a DataFrame into a time series-like format, keeping the original index.

    :param df: pandas DataFrame
    :param date_columns: List of columns representing the datetime (Year, Month, Day, Hour)
    :param unique_identifier: Column name of the unique identifier
    :param target_column: Column name of the target values
    :return: Transformed DataFrame
    """
    # Combine the date columns into a single datetime column
    df['datetime'] = pd.to_datetime(df[date_columns])

    # Reset the index (if needed) to ensure the original index is included as a column
    df = df.reset_index()

    # Create a pivot table
    time_series_df = df.pivot_table(index='index', 
                                    columns=unique_identifier, 
                                    values=target_column, 
                                    aggfunc='sum')

    # Rename the columns
    time_series_df.columns = [f"{col}_{target_column}" for col in time_series_df.columns]

    # Join the datetime column back to the pivoted DataFrame
    time_series_df = time_series_df.join(df.set_index('index')['datetime'])

    return time_series_df
