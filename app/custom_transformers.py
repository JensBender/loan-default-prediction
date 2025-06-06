from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


# Check missing values 
class MissingValueChecker(BaseEstimator, TransformerMixin):
    def __init__(self, critical_features=None, non_critical_features=None):
        if critical_features is None or non_critical_features is None:
            raise ValueError("'critical_features' and 'non_critical_features' cannot be None. Provide both lists.")

        self.critical_features = critical_features
        self.non_critical_features = non_critical_features
    
    def fit(self, X, y=None):
        return self  # No fitting needed

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        X_transformed = X.copy()
        
        # --- Critical features ---
        # Calculate total number of missing values  
        n_missing_total_critical = X_transformed[self.critical_features].isnull().sum().sum()
        # Calculate number of rows with missing values  
        n_missing_rows_critical = X_transformed[self.critical_features].isnull().any(axis=1).sum()
        # Create dictionary with number of missing values by column 
        n_missing_by_column_critical = X_transformed[self.critical_features].isnull().sum().to_dict()
        # Raise error  
        if n_missing_total_critical > 0:
            values = "value" if n_missing_total_critical == 1 else "values"
            rows = "row" if n_missing_rows_critical == 1 else "rows"
            raise ValueError(
                f"{n_missing_total_critical} missing {values} found in critical features "
                f"across {n_missing_rows_critical} {rows}. Please provide missing {values}.\n"
                f"Missing values by column: {n_missing_by_column_critical}"
            )

        # --- Non-critical features ---
        # Calculate total number of missing values 
        n_missing_total_noncritical = X_transformed[self.non_critical_features].isnull().sum().sum()        
        # Calculate number of rows with missing values 
        n_missing_rows_noncritical = X_transformed[self.non_critical_features].isnull().any(axis=1).sum()
        # Create dictionary with number of missing values by column 
        n_missing_by_column_noncritical = X_transformed[self.non_critical_features].isnull().sum().to_dict()
        # Display warning message
        if n_missing_total_noncritical > 0:
            values = "value" if n_missing_total_noncritical == 1 else "values"
            rows = "row" if n_missing_rows_noncritical == 1 else "rows"
            print(
                f"Warning: {n_missing_total_noncritical} missing {values} found in non-critical features "
                f"across {n_missing_rows_noncritical} {rows}. Missing {values} will be imputed.\n"
                f"Missing values by column: {n_missing_by_column_noncritical}"
            )
        
        return X_transformed


# Standardize categorical labels to snake_case
class CategoricalLabelStandardizer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self  # No fitting needed

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
            
        X_transformed = X.copy()
        columns = self.columns if self.columns else X_transformed.columns  # Use provided columns, otherwise all columns
        for column in columns:
            X_transformed[column] = X_transformed[column].apply(
                lambda categorical_label: (
                    categorical_label
                    .strip()  # Remove leading/trailing spaces
                    .lower()  # Convert to lowercase
                    .replace("-", "_")  # Replace hyphens with "_"
                    .replace("/", "_")  # Replace slashes with "_"
                    .replace(" ", "_")  # Replace spaces with "_"
                    if isinstance(categorical_label, str) else categorical_label
                )
            )

        return X_transformed


# Convert binary categorical columns to boolean columns 
class BooleanColumnTransformer(BaseEstimator, TransformerMixin):  
    def __init__(self, boolean_column_mappings=None):
        if boolean_column_mappings is None:
            raise ValueError("'boolean_column_mappings' cannot be None. It must be a dictionary specifying the mappings.")
            
        self.boolean_column_mappings = boolean_column_mappings 
            
    def fit(self, X, y=None):  
        return self  # No fitting needed

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
            
        X_transformed = X.copy()
        for column, mapping in self.boolean_column_mappings.items():
            if column in X_transformed.columns:
                X_transformed[column] = X_transformed[column].map(mapping)

        return X_transformed


# Derive job stability from profession 
class JobStabilityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, job_stability_map=None):
        if job_stability_map is None:
            raise ValueError("'job_stability_map' cannot be None. It must be a dictionary specifying the mappings from 'profession' to 'job_stability'.")

        self.job_stability_map = job_stability_map 
        
    def fit(self, X, y=None):
        return self  # No fitting needed

    def transform(self, X):  
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        # Create job stability column by mapping professions to job stability tiers (default to "moderate" for unknown professions)
        X_transformed = X.copy()
        X_transformed["job_stability"] = X_transformed["profession"].map(self.job_stability_map).fillna("moderate")

        return X_transformed


# Derive city tier from city
class CityTierTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, city_tier_map=None):
        if city_tier_map is None:
            raise ValueError("'city_tier_map' cannot be None. It must be a dictionary specifying the mappings from 'city' to 'city_tier'.")

        self.city_tier_map = city_tier_map 

    def fit(self, X, y=None):
        return self  # No fitting needed

    def transform(self, X):        
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        # Create city tier column by mapping cities to city tiers (default to "unknown" for unknown cities)
        X_transformed = X.copy()
        X_transformed["city_tier"] = X_transformed["city"].map(self.city_tier_map).fillna("unknown")
        
        return X_transformed


# Target encoding of state default rate 
class StateDefaultRateTargetEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        # Merge X and y
        df = X.copy()
        df["target"] = y.values
        
        # Calculate default rate by state 
        self.default_rate_by_state_ = df.groupby("state")["target"].mean()
        
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        # Create state default rate column by mapping the state to its corresponding default rate
        X_transformed = X.copy()
        X_transformed["state_default_rate"] = X_transformed["state"].map(self.default_rate_by_state_)
        
        return X_transformed


# Feature selection for downstream model training and inference 
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_keep):
        self.columns_to_keep = columns_to_keep

    def fit(self, X, y=None):
        return self  # No fitting needed

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
            
        X_transformed = X[self.columns_to_keep].copy()
        
        return X_transformed 