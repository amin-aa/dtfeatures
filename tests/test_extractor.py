import pytest
import pandas as pd
from dtfeatures.extractor import DatetimeFeatureExtractor, FeatureCategory

# Test input data
df = pd.DataFrame({
    'datetime': pd.date_range(start="2023-01-01", periods=5, freq='h')
})

def test_default_feature_extraction():
    """
    Test feature extraction with default feature categories (all features enabled).
    """
    extractor = DatetimeFeatureExtractor(datetime_column='datetime')
    features = extractor.extract(df)
    
    # Check if the expected features are in the DataFrame
    assert 'year' in features.columns
    assert 'hour' in features.columns
    assert 'sin_second_of_day' in features.columns
    assert 'month' in features.columns

def test_custom_feature_extraction():
    """
    Test feature extraction with custom selected feature categories.
    """
    selected_features = [FeatureCategory.BASIC_DATE, FeatureCategory.BUSINESS]
    extractor = DatetimeFeatureExtractor(selected_features=selected_features, datetime_column='datetime')
    features = extractor.extract(df)
    
    # Check if only the selected features are present
    assert 'year' in features.columns
    assert 'month' in features.columns
    assert 'quarter' in features.columns
    
    # Ensure no cyclic or time features are extracted
    assert 'hour' not in features.columns
    assert 'sin_second_of_day' not in features.columns

def test_cyclic_feature_extraction():
    """
    Test feature extraction with cyclical features.
    """
    cyclic_options = {'cyclic_type': ['hour_of_day'], 'cyclic_function': ['sin', 'cos']}
    extractor = DatetimeFeatureExtractor(selected_features=[FeatureCategory.CYCLICAL], 
                                         datetime_column='datetime', 
                                         cyclic_options=cyclic_options)
    features = extractor.extract(df)
    
    # Check if cyclical features are calculated correctly
    assert 'sin_hour_of_day' in features.columns
    assert 'cos_hour_of_day' in features.columns

def test_invalid_feature_category():
    """
    Test that invalid feature categories raise an error.
    """
    with pytest.raises(ValueError, match="Unknown feature category: invalid_feature"):
        DatetimeFeatureExtractor(selected_features=['invalid_feature'])

def test_invalid_feature_type():
    """
    Test that invalid feature type raises a TypeError.
    """
    with pytest.raises(TypeError, match="Invalid feature type: <class 'int'>. Must be str or FeatureCategory."):
        DatetimeFeatureExtractor(selected_features=[1])

def test_extract_single_datetime():
    """
    Test feature extraction from a single datetime input.
    """
    single_datetime = pd.Timestamp('2023-01-01 00:00:00')
    extractor = DatetimeFeatureExtractor(datetime_column='datetime')
    features = extractor.extract(single_datetime)
    
    # Check if features are extracted and returned in a DataFrame
    assert isinstance(features, pd.DataFrame)
    assert 'year' in features.columns
    assert 'hour' in features.columns

def test_extract_series():
    """
    Test feature extraction from a pandas Series of datetime values.
    """
    datetime_series = pd.Series(pd.date_range(start="2023-01-01", periods=5, freq='h'))
    extractor = DatetimeFeatureExtractor(datetime_column='datetime')
    features = extractor.extract(datetime_series)
    
    # Check if features are extracted and returned in a DataFrame
    assert isinstance(features, pd.DataFrame)
    assert 'year' in features.columns
    assert 'hour' in features.columns
