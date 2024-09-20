import pandas as pd
from enum import Enum
from dtfeatures.base import BaseDatetimeFeatures
from dtfeatures.basic_date import BasicDateFeatures
from dtfeatures.basic_time import BasicTimeFeatures
from dtfeatures.cyclical import CyclicalFeatures
from dtfeatures.business import BusinessFeatures
from dtfeatures.seasonality import SeasonalityFeatures

# Enum to define the feature categories
class FeatureCategory(Enum):
    BASIC_DATE = "basic_date"
    BASIC_TIME = "basic_time"
    CYCLICAL = "cyclical"
    BUSINESS = "business"
    SEASONALITY = "seasonality"

class DatetimeFeatureExtractor(BaseDatetimeFeatures):

    def __init__(self, selected_features=None, datetime_column='datetime', cyclic_options=None):
        """
        Initialize with options for selected features and cyclic features.
        :param selected_features: List of feature categories (either FeatureCategory enum members or strings).
        :param cyclic_options: Specify which features to apply cyclical transformations to, e.g., ['hour', 'month'].
        """
        super().__init__(datetime_column=datetime_column)
        self.selected_features = selected_features or [
            FeatureCategory.BASIC_DATE, 
            FeatureCategory.BASIC_TIME, 
            FeatureCategory.CYCLICAL, 
            FeatureCategory.BUSINESS, 
            FeatureCategory.SEASONALITY
        ]
        
        # Normalize the selected features to always be Enum members
        self.selected_features = self._normalize_selected_features(self.selected_features)
        
        self.cyclic_options = cyclic_options or {'cyclic_type':['second_of_day', 'minute_of_day'], 'cyclic_function':['sin', 'cos']}

        # Mapping from FeatureCategory Enum to corresponding classes
        self.feature_mapping = {
            FeatureCategory.BASIC_DATE: BasicDateFeatures,
            FeatureCategory.BASIC_TIME: BasicTimeFeatures,
            FeatureCategory.CYCLICAL: CyclicalFeatures,
            FeatureCategory.BUSINESS: BusinessFeatures,
            FeatureCategory.SEASONALITY: SeasonalityFeatures
        }

        # Dictionary to store initialized feature extractors
        self.feature_extractors = {}

        # Initialize extractors for the selected features
        self._initialize_extractors()

    def _normalize_selected_features(self, selected_features):
        """
        Convert string feature names into FeatureCategory Enum members, if necessary.
        :param selected_features: A list of FeatureCategory Enum members or strings.
        :return: A list of FeatureCategory Enum members.
        """
        normalized_features = []
        for feature in selected_features:
            if isinstance(feature, str):
                # Convert string to FeatureCategory Enum
                try:
                    normalized_features.append(FeatureCategory[feature.upper()])
                except KeyError:
                    raise ValueError(f"Unknown feature category: {feature}")
            elif isinstance(feature, FeatureCategory):
                # If it's already an Enum, use it directly
                normalized_features.append(feature)
            else:
                raise TypeError(f"Invalid feature type: {type(feature)}. Must be str or FeatureCategory.")
        return normalized_features

    def _initialize_extractors(self):
        """
        Initialize all feature extractors based on selected features.
        """
        for feature in self.selected_features:
            if feature in self.feature_mapping:
                if feature == FeatureCategory.CYCLICAL:
                    # Pass cyclic options to CyclicalFeatures extractor during initialization
                    self.feature_extractors[feature] = self.feature_mapping[feature](datetime_column=self.datetime_column, **self.cyclic_options)
                else:
                    self.feature_extractors[feature] = self.feature_mapping[feature](datetime_column=self.datetime_column)
            else:
                raise ValueError(f"Unknown feature category: {feature}")

    def extract(self, data):
        """
        Extract features from the input data based on the selected feature categories.
        :param data: Input data (single datetime, Series, or DataFrame)
        :return: DataFrame with extracted features
        """
        df = self.ensure_dataframe(data)
        df_features = pd.DataFrame()

        # Extract features from each initialized extractor
        for feature, extractor in self.feature_extractors.items():
            extracted = extractor.extract(df)
            df_features = pd.concat([df_features, extracted], axis=1)
        
        df_features.set_index(df[self.datetime_column], inplace=True)
        return df_features
