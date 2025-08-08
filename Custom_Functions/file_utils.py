import pandas as pd 
import csv

def read_csv_as_a_list(file_path):
    """
    Reads a CSV file and returns its content as a list of dictionaries.
    
    :param file_path: Path to the CSV file.
    :return: List of dictionaries representing the rows in the CSV file.
    """
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        return list(reader)

def read_csv_with_pandas(file_path):
    """Reads a CSV file using pandas and returns a DataFrame."""
    return pd.read_csv(file_path)

def read_csv_with_pandas_as_list(file_path):
    """
    Reads a CSV file using pandas and returns its content as a list of dictionaries.
    
    :param file_path: Path to the CSV file.
    :return: List of dictionaries representing the rows in the CSV file.
    """
    df = pd.read_csv(file_path)
    return df.to_dict(orient='records')

def write_list_to_csv(file_path, data_list):
    """
    Writes a list of dictionaries to a CSV file.
    
    :param file_path: Path to the output CSV file.
    :param data_list: List of dictionaries to write to the CSV file.
    """
    if not data_list:
        return
    
    keys = data_list[0].keys()
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data_list)

def append_to_csv(file_path, data_list):
    """
    Appends a list of dictionaries to an existing CSV file.
    
    :param file_path: Path to the output CSV file.
    :param data_list: List of dictionaries to append to the CSV file.
    """
    if not data_list:
        return
    
    keys = data_list[0].keys()
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writerows(data_list)

def write_pandas_to_csv(file_path, df):
    """
    Writes a pandas DataFrame to a CSV file.
    
    :param file_path: Path to the output CSV file.
    :param df: Pandas DataFrame to write to the CSV file.
    """
    df.to_csv(file_path, index=False)

def append_pandas_to_csv(file_path, df):
    """
    Appends a pandas DataFrame to an existing CSV file.
    
    :param file_path: Path to the output CSV file.
    :param df: Pandas DataFrame to append to the CSV file.
    """
    df.to_csv(file_path, mode='a', header=False, index=False)

def read_csv_with_pandas_and_filter(file_path, filter_func):
    """
    Reads a CSV file using pandas, applies a filter function, and returns the filtered DataFrame.
    
    :param file_path: Path to the CSV file.
    :param filter_func: Function to filter the DataFrame.
    :return: Filtered DataFrame.
    """
    df = pd.read_csv(file_path)
    return df[filter_func(df)]