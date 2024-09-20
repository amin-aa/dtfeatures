import pytest
import pandas as pd
from dtfeatures.basic_date import BasicDateFeatures

# Sample test data
df = pd.DataFrame({
    'datetime': pd.to_datetime(['2023-01-01', '2023-06-15', '2023-12-31', '2024-03-01'])
})

def test_basic_date_feature_extraction():
    """
    Test the extraction of basic date features from a DataFrame.
    """
    extractor = BasicDateFeatures(datetime_column='datetime')
    features = extractor.extract(df)
    
    # Verify the columns
    expected_columns = ['year', 'month', 'day', 'day_of_week', 'day_of_year', 'week_of_year', 'quarter']
    assert list(features.columns) == expected_columns
    
    # Verify the extracted values
    assert features['year'].tolist() == [2023, 2023, 2023, 2024]
    assert features['month'].tolist() == [1, 6, 12, 3]
    assert features['day'].tolist() == [1, 15, 31, 1]
    assert features['day_of_week'].tolist() == [6, 3, 6, 4]  # Sunday=6, Wednesday=3, Sunday=6, Friday=4
    assert features['day_of_year'].tolist() == [1, 166, 365, 61]
    assert features['week_of_year'].tolist() == [52, 24, 52, 9]  # Week of the year (ISO format)
    assert features['quarter'].tolist() == [1, 2, 4, 1]

def test_empty_dataframe():
    """
    Test feature extraction with an empty DataFrame.
    """
    empty_df = pd.DataFrame({'datetime': pd.to_datetime([])})
    extractor = BasicDateFeatures(datetime_column='datetime')
    features = extractor.extract(empty_df)
    
    # Ensure the output is an empty DataFrame with the correct columns
    expected_columns = ['year', 'month', 'day', 'day_of_week', 'day_of_year', 'week_of_year', 'quarter']
    assert features.empty
    assert list(features.columns) == expected_columns

def test_non_standard_datetime_column_name():
    """
    Test feature extraction with a non-standard datetime column name.
    """
    df_with_custom_column = pd.DataFrame({
        'custom_date': pd.to_datetime(['2023-01-01', '2023-06-15', '2023-12-31'])
    })
    extractor = BasicDateFeatures(datetime_column='custom_date')
    features = extractor.extract(df_with_custom_column)
    
    # Verify the extracted values for a custom datetime column
    assert features['year'].tolist() == [2023, 2023, 2023]
    assert features['month'].tolist() == [1, 6, 12]
    assert features['day'].tolist() == [1, 15, 31]
    assert features['day_of_week'].tolist() == [6, 3, 6]  # Sunday, Wednesday, Sunday

def test_single_datetime_input():
    """
    Test feature extraction from a single datetime input.
    """
    single_datetime = pd.DataFrame({
        'datetime': [pd.Timestamp('2023-05-21')]
    })
    extractor = BasicDateFeatures(datetime_column='datetime')
    features = extractor.extract(single_datetime)
    
    # Verify the extracted features for a single datetime
    assert features['year'].tolist() == [2023]
    assert features['month'].tolist() == [5]
    assert features['day'].tolist() == [21]
    assert features['day_of_week'].tolist() == [6]  # Sunday
    assert features['day_of_year'].tolist() == [141]
    assert features['week_of_year'].tolist() == [20]
    assert features['quarter'].tolist() == [2]
