import pytest
import pandas as pd
import numpy as np
from dtfeatures.cyclical import CyclicalFeatures, CycleType

# Helper function for approximate comparison
def assert_series_almost_equal(series1, series2, tolerance=1e-6):
    assert np.allclose(series1, series2, atol=tolerance), f"Series are not almost equal:\n{series1}\n{series2}"

def test_extract_cyclic_second_of_day():
    # Create a DataFrame with datetime column
    df = pd.DataFrame({'datetime': pd.date_range('2023-01-01 00:00:00', periods=5, freq='h')})
    
    # Instantiate CyclicalFeatures with second_of_day cycle
    cyclical = CyclicalFeatures(datetime_column='datetime', cyclic_function=['sin', 'cos'], cyclic_type=['second_of_day'])
    
    result = cyclical.extract(df)
    
    # Check if cyclic features are generated correctly
    assert 'sin_second_of_day' in result.columns
    assert 'cos_second_of_day' in result.columns
    
    # Manually calculate expected values for sine and cosine
    expected_second_of_day = df['datetime'].apply(lambda dt: dt.hour * 3600 + dt.minute * 60 + dt.second)
    expected_sin = np.sin(2 * np.pi * expected_second_of_day / 86400)  # 86400 seconds in a day
    expected_cos = np.cos(2 * np.pi * expected_second_of_day / 86400)
    
    assert_series_almost_equal(result['sin_second_of_day'], expected_sin)
    assert_series_almost_equal(result['cos_second_of_day'], expected_cos)

def test_extract_cyclic_minute_of_day():
    # Create a DataFrame with datetime column
    df = pd.DataFrame({'datetime': pd.date_range('2023-01-01 00:00:00', periods=5, freq='h')})
    
    # Instantiate CyclicalFeatures with minute_of_day cycle
    cyclical = CyclicalFeatures(datetime_column='datetime', cyclic_function=['sin', 'cos'], cyclic_type=['minute_of_day'])
    
    result = cyclical.extract(df)
    
    # Check if cyclic features are generated correctly
    assert 'sin_minute_of_day' in result.columns
    assert 'cos_minute_of_day' in result.columns
    
    # Manually calculate expected values for sine and cosine
    expected_minute_of_day = df['datetime'].apply(lambda dt: dt.hour * 60 + dt.minute)
    expected_sin = np.sin(2 * np.pi * expected_minute_of_day / 1440)  # 1440 minutes in a day
    expected_cos = np.cos(2 * np.pi * expected_minute_of_day / 1440)
    
    assert_series_almost_equal(result['sin_minute_of_day'], expected_sin)
    assert_series_almost_equal(result['cos_minute_of_day'], expected_cos)

def test_extract_multiple_cycle_types():
    # Create a DataFrame with datetime column
    df = pd.DataFrame({'datetime': pd.date_range('2023-01-01 00:00:00', periods=5, freq='h')})
    
    # Instantiate CyclicalFeatures with multiple cycle types
    cyclical = CyclicalFeatures(datetime_column='datetime', cyclic_function=['sin', 'cos'], cyclic_type=['second_of_day', 'minute_of_day'])
    
    result = cyclical.extract(df)
    
    # Check if cyclic features are generated correctly for both types
    assert 'sin_second_of_day' in result.columns
    assert 'cos_second_of_day' in result.columns
    assert 'sin_minute_of_day' in result.columns
    assert 'cos_minute_of_day' in result.columns

def test_custom_datetime_column():
    # Create a DataFrame with custom datetime column
    df = pd.DataFrame({'custom_date': pd.date_range('2023-01-01 00:00:00', periods=5, freq='h')})
    
    # Instantiate CyclicalFeatures with custom datetime column
    cyclical = CyclicalFeatures(datetime_column='custom_date', cyclic_function=['sin', 'cos'], cyclic_type=['hour_of_day'])
    
    result = cyclical.extract(df)
    
    # Check if cyclic features are generated correctly for custom column
    assert 'sin_hour_of_day' in result.columns
    assert 'cos_hour_of_day' in result.columns

def test_empty_dataframe():
    # Test with an empty DataFrame
    df = pd.DataFrame({'datetime': pd.to_datetime([])})
    
    # Instantiate CyclicalFeatures
    cyclical = CyclicalFeatures(datetime_column='datetime', cyclic_function=['sin', 'cos'], cyclic_type=['second_of_day'])
    
    result = cyclical.extract(df)
    
    # Check that the result is also an empty DataFrame
    assert result.empty

def test_invalid_cycle_type():
    # Create a DataFrame with datetime column
    df = pd.DataFrame({'datetime': pd.date_range('2023-01-01 00:00:00', periods=5, freq='h')})
    
    # Instantiate CyclicalFeatures with an invalid cycle type
    with pytest.raises(ValueError):
        cyclical = CyclicalFeatures(datetime_column='datetime', cyclic_function=['sin'], cyclic_type=['invalid_cycle'])
        cyclical.extract(df)
