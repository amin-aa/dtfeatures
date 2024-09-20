import pandas as pd
import numpy as np
from enum import Enum
import calendar
from dtfeatures.base import BaseDatetimeFeatures, CyclePeriod


class CycleType(Enum):
    """Enum to define cycle types for clarity."""
    SECOND_OF_DAY = 'second_of_day'
    SECOND_OF_HOUR = 'second_of_hour'
    SECOND_OF_MINUTE = 'second_of_minute'
    MINUTE_OF_DAY = 'minute_of_day'
    HOUR_OF_DAY = 'hour_of_day'

class CyclicFunction:
    """Class encapsulating common cyclic functions."""
    sin = np.sin
    cos = np.cos


class CyclicalFeatures(BaseDatetimeFeatures):

    mapping_cyclic_type_2_value = {
        CycleType.SECOND_OF_DAY: lambda dt: dt.hour * CyclePeriod.SECOND_OF_HOUR +
                                            dt.minute * CyclePeriod.SECOND_OF_MINUTE + dt.second,
        CycleType.SECOND_OF_HOUR: lambda dt: dt.minute * CyclePeriod.SECOND_OF_MINUTE + dt.second,
        CycleType.SECOND_OF_MINUTE: lambda dt: dt.second,
        CycleType.MINUTE_OF_DAY: lambda dt: dt.hour * CyclePeriod.MINUTE_OF_HOUR + dt.minute,
        CycleType.HOUR_OF_DAY: lambda dt: dt.hour
    }
    
    def __init__(self, datetime_column='datetime', cyclic_function=['sin', 'cos'], cyclic_type=['second_of_day', 'minute_of_day']):
        super().__init__(datetime_column=datetime_column)
        self.cyclic_function = cyclic_function
        self.cyclic_type = cyclic_type
    
    def extract(self, df):

        cyclic_features = pd.DataFrame()
        # Validate that the cyclic type exists in CycleType
        for cycle_type in self.cyclic_type:
            if not hasattr(CyclePeriod, cycle_type.upper()):
                raise ValueError(f"Invalid cyclic type '{cycle_type}' provided. Must be one of {list(CycleType)}.")

        for function_name in self.cyclic_function:
            function = getattr(CyclicFunction, function_name)
            for cycle_type in self.cyclic_type:
                feature_name = f"{function_name}_{cycle_type}"
                period = getattr(CyclePeriod, cycle_type.upper())
                
                cyclic_features[feature_name] = df[self.datetime_column].apply(lambda dt: 
                                                                            function(2 * np.pi * \
                                                                                     CyclicalFeatures.mapping_cyclic_type_2_value\
                                                                                        [CycleType(cycle_type)](dt) / period)
                                                                            )

        return cyclic_features

