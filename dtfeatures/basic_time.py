import pandas as pd
from dtfeatures.base import BaseDatetimeFeatures

class BasicTimeFeatures(BaseDatetimeFeatures):
    def __init__(self, datetime_column='datetime'):
        super().__init__(datetime_column=datetime_column)

    def extract(self, df):
        """
        Extract basic time features like hour, minute, second.
        """
        
        features = pd.DataFrame()

        features['hour'] = df[self.datetime_column].dt.hour
        features['minute'] = df[self.datetime_column].dt.minute
        features['second'] = df[self.datetime_column].dt.second

        return features
