import datetime
import pandas as pd
from enum import Enum

class BaseDatetimeFeatures:
    def __init__(self, datetime_column='datetime'):
        """
        Initialize with the name of the datetime column.
        :param datetime_column: Name of the datetime column in the DataFrame.
        """
        self.datetime_column = datetime_column

    def ensure_dataframe(self, data):
        """
        Ensure the input data is a DataFrame with the datetime column.
        :param data: Input data (single datetime, Series, or DataFrame)
        :return: DataFrame with the datetime column
        """
        if isinstance(data, pd.Series):
            data = data.to_frame(name=self.datetime_column)
        elif isinstance(data, pd.Timestamp):
            data = pd.DataFrame([data], columns=[self.datetime_column])
        elif isinstance(data, datetime.datetime):
            return pd.DataFrame([data], columns=[self.datetime_column])
        elif isinstance(data, pd.DataFrame):
            if self.datetime_column not in data.columns:
                raise ValueError(f"DataFrame must contain a '{self.datetime_column}' column.")
        else:
            raise TypeError("Input data must be a pandas Series, DataFrame, or Timestamp.")
        return data

class FeaturType(Enum):
    """Enum to define cycle types for clarity."""

    BASIC_DATE = 'basic_date'
    BASIC_TIME = 'basic_time'
    CYCLICAL = 'cyclical'
    BUSINESS = 'business'
    SEASONALITY = 'seasonality'

class BasePeriod:
    """Class defining base time units."""
    MONTH_OF_YEAR = 12
    DAY_OF_MONTH = 30
    DAY_OF_WEEK = 7
    HOUR_OF_DAY = 24
    MINUTE_OF_HOUR = 60
    SECOND_OF_MINUTE = 60

class CyclePeriod(BasePeriod):
    """Class defining time cycles in terms of seconds and minutes."""
    MINUTE_OF_DAY = BasePeriod.MINUTE_OF_HOUR * BasePeriod.HOUR_OF_DAY
    SECOND_OF_HOUR = BasePeriod.SECOND_OF_MINUTE * BasePeriod.MINUTE_OF_HOUR
    SECOND_OF_DAY = BasePeriod.SECOND_OF_MINUTE * BasePeriod.MINUTE_OF_HOUR * BasePeriod.HOUR_OF_DAY
