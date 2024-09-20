import pytest
import pandas as pd
from dtfeatures.business import BusinessFeatures

# Sample test data
df = pd.DataFrame({
    'datetime': pd.to_datetime(['2023-01-01',  # Sunday (weekend)
                                '2023-06-30',  # Friday (month end)
                                '2023-12-31',  # Sunday (month end) (year end)
                                '2023-06-15'])  # Thursday (mid-month)
})

def test_business_feature_extraction():
    """
    Test the extraction of business-related features from a DataFrame.
    """
    extractor = BusinessFeatures(datetime_column='datetime')
    features = extractor.extract(df)

    # Verify the columns
    expected_columns = ['is_weekend', 'is_month_end', 'is_year_end']
    assert list(features.columns) == expected_columns

    # Verify the extracted values
    assert features['is_weekend'].tolist() == [True, False, True, False]
    assert features['is_month_end'].tolist() == [False, True, True, False]
    assert features['is_year_end'].tolist() == [False, False, True, False]

def test_empty_dataframe():
    """
    Test feature extraction with an empty DataFrame.
    """
    empty_df = pd.DataFrame({'datetime': pd.to_datetime([])})
    extractor = BusinessFeatures(datetime_column='datetime')
    features = extractor.extract(empty_df)

    # Ensure the output is an empty DataFrame with the correct columns
    expected_columns = ['is_weekend', 'is_month_end', 'is_year_end']
    assert features.empty
    assert list(features.columns) == expected_columns

def test_non_standard_datetime_column_name():
    """
    Test feature extraction with a non-standard datetime column name.
    """
    df_with_custom_column = pd.DataFrame({
        'custom_time': pd.to_datetime(['2023-01-01', '2023-06-30', '2023-12-31', '2023-06-15'])
    })
    extractor = BusinessFeatures(datetime_column='custom_time')
    features = extractor.extract(df_with_custom_column)

    # Verify the extracted values for a custom datetime column
    assert features['is_weekend'].tolist() == [True, False, True, False]
    assert features['is_month_end'].tolist() == [False, True, True, False]
    assert features['is_year_end'].tolist() == [False, False, True, False]

def test_single_datetime_input():
    """
    Test feature extraction from a single datetime input.
    """
    single_datetime = pd.DataFrame({
        'datetime': [pd.Timestamp('2023-05-21')]  # Sunday (weekend)
    })
    extractor = BusinessFeatures(datetime_column='datetime')
    features = extractor.extract(single_datetime)

    # Verify the extracted features for a single datetime
    assert features['is_weekend'].tolist() == [True]
    assert features['is_month_end'].tolist() == [False]
    assert features['is_year_end'].tolist() == [False]
