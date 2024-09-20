import pytest
import pandas as pd
import datetime
from dtfeatures.base import BaseDatetimeFeatures

def test_ensure_dataframe_from_series():
    feature_extractor = BaseDatetimeFeatures(datetime_column='datetime')
    series = pd.Series(pd.date_range('2023-01-01', periods=3))
    result = feature_extractor.ensure_dataframe(series)
    assert isinstance(result, pd.DataFrame)
    assert 'datetime' in result.columns

def test_ensure_dataframe_from_dataframe():
    feature_extractor = BaseDatetimeFeatures(datetime_column='datetime')
    df = pd.DataFrame({'datetime': pd.date_range('2023-01-01', periods=3)})
    result = feature_extractor.ensure_dataframe(df)
    assert isinstance(result, pd.DataFrame)
    assert 'datetime' in result.columns

def test_ensure_dataframe_raises_error():
    feature_extractor = BaseDatetimeFeatures(datetime_column='custom_date')
    df = pd.DataFrame({'other_col': pd.date_range('2023-01-01', periods=3)})
    with pytest.raises(ValueError):
        feature_extractor.ensure_dataframe(df)

def test_ensure_dataframe_from_timestamp():
    feature_extractor = BaseDatetimeFeatures(datetime_column='datetime')
    timestamp = pd.Timestamp('2023-01-01')
    result = feature_extractor.ensure_dataframe(timestamp)
    assert isinstance(result, pd.DataFrame)
    assert 'datetime' in result.columns

def test_ensure_dataframe_from_datetime():
    feature_extractor = BaseDatetimeFeatures(datetime_column='datetime')
    timestamp = datetime.datetime.now()
    result = feature_extractor.ensure_dataframe(timestamp)
    assert isinstance(result, pd.DataFrame)
    assert 'datetime' in result.columns
