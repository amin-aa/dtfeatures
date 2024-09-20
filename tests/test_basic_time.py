import pytest
import pandas as pd
from dtfeatures.basic_time import BasicTimeFeatures

# Sample test data
df = pd.DataFrame({
    'datetime': pd.to_datetime(['2023-01-01 12:30:15', '2023-06-15 23:59:59', '2023-12-31 00:00:00'])
})

def test_basic_time_feature_extraction():
    """
    Test the extraction of basic time features from a DataFrame.
    """
    extractor = BasicTimeFeatures(datetime_column='datetime')
    features = extractor.extract(df)
    
    # Verify the columns
    expected_columns = ['hour', 'minute', 'second']
    assert list(features.columns) == expected_columns
    
    # Verify the extracted values
    assert features['hour'].tolist() == [12, 23, 0]
    assert features['minute'].tolist() == [30, 59, 0]
    assert features['second'].tolist() == [15, 59, 0]

def test_empty_dataframe():
    """
    Test feature extraction with an empty DataFrame.
    """
    empty_df = pd.DataFrame({'datetime': pd.to_datetime([])})
    extractor = BasicTimeFeatures(datetime_column='datetime')
    features = extractor.extract(empty_df)
    
    # Ensure the output is an empty DataFrame with the correct columns
    expected_columns = ['hour', 'minute', 'second']
    assert features.empty
    assert list(features.columns) == expected_columns

def test_non_standard_datetime_column_name():
    """
    Test feature extraction with a non-standard datetime column name.
    """
    df_with_custom_column = pd.DataFrame({
        'custom_time': pd.to_datetime(['2023-01-01 12:30:15', '2023-06-15 23:59:59', '2023-12-31 00:00:00'])
    })
    extractor = BasicTimeFeatures(datetime_column='custom_time')
    features = extractor.extract(df_with_custom_column)
    
    # Verify the extracted values for a custom datetime column
    assert features['hour'].tolist() == [12, 23, 0]
    assert features['minute'].tolist() == [30, 59, 0]
    assert features['second'].tolist() == [15, 59, 0]

def test_single_datetime_input():
    """
    Test feature extraction from a single datetime input.
    """
    single_datetime = pd.DataFrame({
        'datetime': [pd.Timestamp('2023-05-21 16:45:00')]
    })
    extractor = BasicTimeFeatures(datetime_column='datetime')
    features = extractor.extract(single_datetime)
    
    # Verify the extracted features for a single datetime
    assert features['hour'].tolist() == [16]
    assert features['minute'].tolist() == [45]
    assert features['second'].tolist() == [0]
