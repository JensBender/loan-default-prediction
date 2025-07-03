# Imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import pandas as pd


# --- Custom error classes --- 
# For missing values in critical columns of the X input DataFrame (in MissingValueChecker)
class MissingValueError(ValueError):
    pass

# For mistmatch between expected and actual X input DataFrame columns 
class ColumnMismatchError(ValueError):
    pass


# --- Custom transformer classes for data preprocessing pipeline ---
# Check missing values 
class MissingValueChecker(BaseEstimator, TransformerMixin):
    def __init__(self, critical_features, non_critical_features):
        # Validate input data type
        if not isinstance(critical_features, list):
            raise TypeError("'critical_features' must be a list of column names.")
        if not isinstance(non_critical_features, list):
            raise TypeError("'non_critical_features' must be a list of column names.")

        self.critical_features = critical_features
        self.non_critical_features = non_critical_features
    
    def fit(self, X, y=None):
        # Validate input data type
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")   
        
        # Ensure input DataFrame contains all required columns
        input_columns = set(X.columns)
        required_columns = set(self.critical_features + self.non_critical_features)
        missing_columns = required_columns - input_columns 
        if missing_columns:
            raise ColumnMismatchError(f"Input X is missing the following columns: {', '.join(missing_columns)}.")

        # Ensure input DataFrame doesn't contain any unexpected columns
        unexpected_columns = input_columns - required_columns
        if unexpected_columns:
            raise ColumnMismatchError(f"Input X contains the following columns that are neither defined in 'critical_features' nor 'non_critical_features: {', '.join(unexpected_columns)}.")
        
        # Store input feature number and names as learned attributes
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist()

        return self 

    def transform(self, X):
        # Ensure .fit() happened before
        check_is_fitted(self)
        
        # Validate input data type
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")   
        
        # Ensure input feature names and feature order is the same as during .fit()
        if X.columns.tolist() != self.feature_names_in_:
            raise ValueError("Feature names and feature order of input X must be the same as during .fit().")      

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
            raise MissingValueError(
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


# Format categorical labels in snake_case
class SnakeCaseFormatter(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        if not isinstance(columns, list) and columns is not None:
            raise TypeError("'columns' must be a list of column names or None. If None, all columns will be used.")
         
        self.columns = columns
    
    def fit(self, X, y=None):
        # Validate input data type
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")   
        
        # Store input feature number and names as learned attributes
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist()

        # Determine columns to be transformed (all if none provided)
        if self.columns is None:
            self.columns_ = X.columns.tolist()
        else:
            self.columns_ = self.columns
            # Ensure input DataFrame contains all required columns
            missing_columns = set(self.columns_) - set(X.columns)
            if missing_columns:
                raise ColumnMismatchError(f"Input X is missing the following columns: {', '.join(missing_columns)}.")
            
        return self  

    def transform(self, X):
        # Ensure .fit() happened before
        check_is_fitted(self)
        
        # Validate input data type
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")   
            
        # Ensure input feature names and feature order is the same as during .fit()
        if X.columns.tolist() != self.feature_names_in_:
            raise ValueError("Feature names and feature order of input X must be the same as during .fit().")      

        X_transformed = X.copy()
        
        for column in self.columns_:
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
        # Validate input data type
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")   
        
        # Store input feature number and names as learned attributes
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist()
        
        return self  

    def transform(self, X):
        # Ensure .fit() happened before
        check_is_fitted(self)
        
        # Validate input data type
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")   
                    
        # Ensure input feature names and feature order is the same as during .fit()
        if X.columns.tolist() != self.feature_names_in_:
            raise ValueError("Feature names and feature order of input X must be the same as during .fit().")      

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
        # Store input feature number and names as learned attributes
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist()
        
        return self  

    def transform(self, X):  
        # Ensure .fit() happened before
        check_is_fitted(self)
        
        # Validate input data type
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")   
        
        # Ensure input feature names and feature order is the same as during .fit()
        if X.columns.tolist() != self.feature_names_in_:
            raise ValueError("Feature names and feature order of input X must be the same as during .fit().")      

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
        # Validate input data type
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")   
    
        # Store input feature number and names as learned attributes
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist()
        
        return self  

    def transform(self, X):        
        # Ensure .fit() happened before
        check_is_fitted(self)
        
        # Validate input data type
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")   
        
        # Ensure input feature names and feature order is the same as during .fit()
        if X.columns.tolist() != self.feature_names_in_:
            raise ValueError("Feature names and feature order of input X must be the same as during .fit().")      

        # Create city tier column by mapping cities to city tiers (default to "unknown" for unknown cities)
        X_transformed = X.copy()
        X_transformed["city_tier"] = X_transformed["city"].map(self.city_tier_map).fillna("unknown")
        
        return X_transformed


# Target encoding of state default rate 
class StateDefaultRateTargetEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        # Validate input data type
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")   

        # Store input feature number and names as learned attributes
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist()

        # Merge X and y
        df = X.copy()
        df["target"] = y.values
        
        # Calculate default rate by state 
        self.default_rate_by_state_ = df.groupby("state")["target"].mean()
        
        return self  

    def transform(self, X):
        # Ensure .fit() happened before
        check_is_fitted(self)

        # Validate input data type
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.") 

        # Ensure input feature names and feature order is the same as during .fit()
        if X.columns.tolist() != self.feature_names_in_:
            raise ValueError("Feature names and feature order of input X must be the same as during .fit().")      

        # Create state default rate column by mapping the state to its corresponding default rate
        X_transformed = X.copy()
        X_transformed["state_default_rate"] = X_transformed["state"].map(self.default_rate_by_state_)
        
        return X_transformed


# Feature selection for downstream model training and inference 
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_keep):
        # Validate input data type
        if not isinstance(columns_to_keep, list):
            raise TypeError("columns_to_keep must be a list.")
        
        self.columns_to_keep = columns_to_keep

    def fit(self, X, y=None):
        # Validate input data type
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")   
        
        # Ensure input DataFrame contains all columns_to_keep
        missing_columns = set(self.columns_to_keep) - set(X.columns)
        if missing_columns:
            raise ColumnMismatchError(f"Input X is missing the following columns: {', '.join(missing_columns)}.")
            
        # Store input feature number and names as learned attributes
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist()

        return self  

    def transform(self, X):
        # Ensure .fit() happened before
        check_is_fitted(self)

        # Validate input data type
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        # Ensure input feature names and feature order is the same as during .fit()
        if X.columns.tolist() != self.feature_names_in_:
            raise ValueError("Feature names and feature order of input X must be the same as during .fit().")      
        
        # Create transformed DataFrame with only the selected features
        X_transformed = X[self.columns_to_keep].copy()
        
        return X_transformed 