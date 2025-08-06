import pandas as pd
import logging

class QueryEngine:
    def __init__(self, csv_path: str):
        """
        Initialize QueryEngine by loading a CSV file into a DataFrame.
        """
        self.csv_path = csv_path
        try:
            self.df = pd.read_csv(csv_path)
            logging.info(f"Data loaded successfully from {csv_path}")
        except FileNotFoundError:
            logging.error(f"Error: The file {csv_path} was not found.")
            self.df = pd.DataFrame()
        except Exception as e:
            logging.error(f"An error occurred while reading the file: {e}")
            self.df = pd.DataFrame()

    def preview(self, num_rows: int = 5) -> pd.DataFrame:
        """
        Return the first n rows of the DataFrame.
        """
        if self.df.empty:
            logging.warning("DataFrame is empty. Please check the CSV file path.")
            return pd.DataFrame()
        return self.df.head(num_rows)
    
    def search(self, keyword: str) -> pd.DataFrame:
        """
        Search for a keyword in all columns of the DataFrame.
        Returns rows that contain the keyword (case-insensitive).
        """
        if self.df.empty:
            logging.warning("DataFrame is empty. Please check the CSV file path.")
            return pd.DataFrame()
        try:
            return self.df[self.df.apply(
                lambda row: row.astype(str).str.contains(keyword, case=False).any(), axis=1
            )]
        except Exception as e:
            logging.error(f"An error occurred while searching: {e}")
            return pd.DataFrame()