"""
Data Service

Handles all data loading, validation, and preparation operations.
"""
from pathlib import Path
from typing import Optional, Union, Tuple
import pandas as pd

from ..utils.file_handler import FileHandler
from ..core import (
    TargetNotFoundError,
    InsufficientDataError,
    Config,
    logger,
    TimedOperation
)


class DataService:
    """Service for data loading and validation operations"""
    
    @staticmethod
    def load_and_validate(
        file_path: Union[str, Path], 
        target_column: Optional[str] = None,
        sample_size: Optional[int] = None
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Load data from file and perform basic validation.
        
        Args:
            file_path: Path to the data file (CSV or Parquet)
            target_column: Optional target column name for supervised learning
            sample_size: Optional limit on number of rows to load
            
        Returns:
            Tuple of (dataframe, file_info_dict)
            
        Raises:
            TargetNotFoundError: If target_column is specified but not found
            InsufficientDataError: If dataset has too few rows for processing
        """
        file_path = Path(file_path)
        
        with TimedOperation("data_loading", file_path=str(file_path)):
            # Load the data
            df = FileHandler.read_file(file_path, sample_size=sample_size)
            file_info = FileHandler.get_file_info(file_path)
            
            # Validate target column if specified
            if target_column and target_column not in df.columns:
                raise TargetNotFoundError(target_column, df.columns.tolist())
            
            # Check minimum data requirements
            if len(df) < Config.MIN_ROWS_REQUIRED:
                raise InsufficientDataError(
                    required_rows=Config.MIN_ROWS_REQUIRED,
                    actual_rows=len(df),
                    operation="feature engineering"
                )
            
            logger.info(
                "data_loaded",
                rows=len(df),
                columns=len(df.columns),
                target=target_column,
                file_size_mb=file_info.get('size_mb', 0)
            )
            
            return df, file_info
    
    @staticmethod
    def get_target_series(df: pd.DataFrame, target_column: Optional[str]) -> Optional[pd.Series]:
        """
        Extract target series from dataframe if target column is specified.
        
        Args:
            df: Input dataframe
            target_column: Name of target column
            
        Returns:
            Target series or None if no target specified
        """
        if target_column and target_column in df.columns:
            return df[target_column]
        return None
    
    @staticmethod
    def prepare_output_dataframe(
        df: pd.DataFrame, 
        selected_feature_names: list, 
        target_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Prepare final output dataframe with selected features and target.
        
        Args:
            df: Original dataframe
            selected_feature_names: List of selected feature names
            target_column: Optional target column to include at the end
            
        Returns:
            Dataframe with selected features and optional target column
        """
        # Remove target column from features if it was accidentally included
        features_only = [f for f in selected_feature_names if f != target_column]
        
        # Create output dataframe with selected features
        output_df = df[features_only].copy()
        
        # Add target column at the end if present
        if target_column and target_column in df.columns:
            output_df[target_column] = df[target_column]
        
        return output_df