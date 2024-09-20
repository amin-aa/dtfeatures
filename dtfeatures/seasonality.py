import pandas as pd
from dtfeatures.base import BaseDatetimeFeatures

class SeasonalityFeatures(BaseDatetimeFeatures):
    def __init__(self, datetime_column='datetime'):
        super().__init__(datetime_column=datetime_column)

    def extract(self, df):
        """
        Extract season feature based on the month.
        """
        features = pd.DataFrame()

        features['season'] = df[self.datetime_column].dt.month.apply(self._get_season)
        
        return features

    def _get_season(self, month):
        """
        Helper function to classify months into seasons.
        """
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
