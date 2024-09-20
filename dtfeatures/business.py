import pandas as pd
from dtfeatures.base import BaseDatetimeFeatures

class BusinessFeatures(BaseDatetimeFeatures):
    def __init__(self, datetime_column='datetime'):
        super().__init__(datetime_column=datetime_column)

    def extract(self, df):
        """
        Extract business-related features like is_weekend, is_month_end, is_year_end.
        """

        features = pd.DataFrame()

        features['is_weekend'] = df[self.datetime_column].dt.dayofweek >= 5
        features['is_month_end'] = df[self.datetime_column].dt.is_month_end
        features['is_year_end'] = df[self.datetime_column].dt.is_year_end

        return features
