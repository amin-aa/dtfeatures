import pytest
import pandas as pd
from dtfeatures.seasonality import SeasonalityFeatures

# Sample test data
df = pd.DataFrame({
    'datetime': pd.to_datetime([
        '2023-01-15',  # Winter
        '2023-04-10',  # Spring
        '2023-07-20',  # Summer
        '2023-10-05',  # Fall
        '2023-12-25'   # Winter
    ])
})

def test_seasonality_feature_extraction():
    """
    Test the extraction of seasonality features from a DataFrame.
    """
    extractor = SeasonalityFeatures(datetime_column='datetime')
    features = extractor.extract(df)

    # Verify the columns
    expected_columns = ['season']
    assert list(features.columns) == expected_columns

    # Verify the extracted season values
    expected_seasons = ['Winter', 'Spring', 'Summer', 'Fall', 'Winter']
    assert features['season'].tolist() == expected_seasons

def test_empty_dataframe():
    """
    Test feature extraction with an empty DataFrame.
    """
    empty_df = pd.DataFrame({'datetime': pd.to_datetime([])})
    extractor = SeasonalityFeatures(datetime_column='datetime')
    features = extractor.extract(empty_df)

    # Ensure the output is an empty DataFrame with the correct column
    expected_columns = ['season']
    assert features.empty
    assert list(features.columns) == expected_columns

def test_non_standard_datetime_column_name():
    """
    Test feature extraction with a non-standard datetime column name.
    """
    df_with_custom_column = pd.DataFrame({
        'custom_time': pd.to_datetime(['2023-03-21',  # Spring
                                       '2023-08-15',  # Summer
                                       '2023-11-30']) # Fall
    })
    extractor = SeasonalityFeatures(datetime_column='custom_time')
    features = extractor.extract(df_with_custom_column)

    # Verify the extracted season values for a custom datetime column
    expected_seasons = ['Spring', 'Summer', 'Fall']
    assert features['season'].tolist() == expected_seasons

def test_single_datetime_input():
    """
    Test feature extraction from a single datetime input.
    """
    single_datetime = pd.DataFrame({
        'datetime': [pd.Timestamp('2023-09-15')]  # Fall
    })
    extractor = SeasonalityFeatures(datetime_column='datetime')
    features = extractor.extract(single_datetime)

    # Verify the extracted season for a single datetime
    assert features['season'].tolist() == ['Fall']
