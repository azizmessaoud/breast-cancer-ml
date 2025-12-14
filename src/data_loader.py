import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, filepath='data.csv'):
        self.filepath = filepath
        self.df = None
        logger.info(f"DataLoader initialized with path: {filepath}")

    def load_data(self):
        try:
            logger.info(f"Loading data from {self.filepath}...")
            self.df = pd.read_csv(self.filepath)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            logger.error(f"File not found: {self.filepath}")
            raise

    def clean_unnamed_columns(self):
        if self.df is None: raise ValueError("Data not loaded.")
        unnamed_cols = [col for col in self.df.columns if 'Unnamed' in col]
        if unnamed_cols:
            self.df = self.df.drop(columns=unnamed_cols)
            logger.info(f"Dropped {len(unnamed_cols)} unnamed columns")
        return self.df

    def explore_data(self):
        if self.df is None: raise ValueError("Data not loaded.")
        logger.info("DATA EXPLORATION SUMMARY")
        logger.info(f"Shape: {self.df.shape}")
        logger.info(f"Missing Values: {self.df.isnull().sum().sum()}")
        return self.df.shape

    def get_data(self):
        if self.df is None: raise ValueError("Data not loaded.")
        return self.df.copy()
