import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, df):
        self.df = df.copy()
        self.le = None
        self.scaler = None
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def remove_id_column(self):
        if 'id' in self.df.columns:
            self.df = self.df.drop('id', axis=1)
        return self.df

    def encode_diagnosis(self):
        if 'diagnosis' not in self.df.columns:
            raise KeyError("No diagnosis column")
        self.le = LabelEncoder()
        self.df['diagnosis'] = self.le.fit_transform(self.df['diagnosis'])
        return self.df

    def split_features_target(self):
        X = self.df.drop('diagnosis', axis=1)
        y = self.df['diagnosis']
        return X, y

    def train_test_split_data(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def standardize_features(self):
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        return X_train_scaled, X_test_scaled
