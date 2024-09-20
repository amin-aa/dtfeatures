import pandas as pd
from dtfeatures.base import BaseDatetimeFeatures

class BasicDateFeatures(BaseDatetimeFeatures):
    def __init__(self, datetime_column='datetime'):
        super().__init__(datetime_column=datetime_column)

    def extract(self, df):
        """
        Extract basic date features like year, month, day, etc.
        """
        
        features = pd.DataFrame()
        
        features['year'] = df[self.datetime_column].dt.year
        features['month'] = df[self.datetime_column].dt.month
        features['day'] = df[self.datetime_column].dt.day
        features['day_of_week'] = df[self.datetime_column].dt.dayofweek  # Monday = 0, Sunday = 6
        features['day_of_year'] = df[self.datetime_column].dt.dayofyear
        features['week_of_year'] = df[self.datetime_column].dt.isocalendar().week
        features['quarter'] = df[self.datetime_column].dt.quarter
        
        
        return features
