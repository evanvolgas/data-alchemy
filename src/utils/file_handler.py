import pandas as pd
from pathlib import Path
from typing import Union, Optional
import pyarrow.parquet as pq
from ..core import FileOperationError, DataValidationError, logger, TimedOperation


class FileHandler:
    """Handle reading CSV and Parquet files with auto-detection"""
    
    @staticmethod
    def detect_file_type(file_path: Union[str, Path]) -> str:
        """Detect file type from extension"""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        logger.debug("detecting_file_type", path=str(path), extension=extension)
        
        if extension in ['.csv', '.tsv', '.txt']:
            return 'csv'
        elif extension in ['.parquet', '.pq']:
            return 'parquet'
        else:
            # Try to infer from content
            try:
                # Try reading as parquet first (binary format)
                pq.read_schema(str(path))
                logger.info("file_type_inferred", file_type="parquet", path=str(path))
                return 'parquet'
            except Exception as e:
                # Assume CSV if parquet fails
                logger.info("file_type_inferred", file_type="csv", path=str(path))
                return 'csv'
    
    @staticmethod
    def read_file(
        file_path: Union[str, Path],
        sample_size: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Read file with auto-detection of format
        
        Args:
            file_path: Path to the file
            sample_size: If provided, only read this many rows
            **kwargs: Additional arguments passed to pandas read functions
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        path = Path(file_path)
        if not path.exists():
            raise FileOperationError(
                f"File not found: {file_path}",
                {"file_path": str(file_path)}
            )
        
        with TimedOperation("file_read", file_path=str(path), sample_size=sample_size):
            file_type = FileHandler.detect_file_type(path)
            
            if file_type == 'csv':
                # Common CSV parameters
                read_kwargs = {
                    'encoding': 'utf-8',
                    'on_bad_lines': 'skip',
                    'low_memory': False
                }
                read_kwargs.update(kwargs)
                
                if sample_size:
                    read_kwargs['nrows'] = sample_size
                    
                # Try different separators
                for sep in [',', '\t', ';', '|']:
                    try:
                        df = pd.read_csv(path, sep=sep, **read_kwargs)
                        # Check if we got reasonable columns
                        if len(df.columns) > 1 or sep == '|':
                            logger.info(
                                "csv_read_success",
                                separator=sep,
                                rows=len(df),
                                columns=len(df.columns)
                            )
                            return df
                    except Exception as e:
                        logger.debug(
                            "csv_separator_failed",
                            separator=sep,
                            error=str(e)
                        )
                        continue
                        
                # If all separators fail, use comma as default
                try:
                    df = pd.read_csv(path, **read_kwargs)
                    return df
                except Exception as e:
                    raise FileOperationError(
                        f"Failed to read CSV file: {path}",
                        {"file_path": str(path), "error": str(e)}
                    )
                
            elif file_type == 'parquet':
                try:
                    df = pd.read_parquet(path, **kwargs)
                    if sample_size and len(df) > sample_size:
                        df = df.head(sample_size)
                        logger.info(
                            "parquet_sampled",
                            original_rows=len(df),
                            sample_size=sample_size
                        )
                    return df
                except Exception as e:
                    raise FileOperationError(
                        f"Failed to read Parquet file: {path}",
                        {"file_path": str(path), "error": str(e)}
                    )
            else:
                raise FileOperationError(
                    f"Unsupported file type: {file_type}",
                    {"file_type": file_type, "file_path": str(path)}
                )
    
    @staticmethod
    def save_features(
        df: pd.DataFrame,
        output_path: Union[str, Path],
        format: str = 'parquet'
    ) -> None:
        """Save feature dataframe to file
        
        Args:
            df: DataFrame to save
            output_path: Output file path
            format: Output format ('parquet' or 'csv')
        """
        path = Path(output_path)
        
        with TimedOperation("file_save", output_path=str(path), format=format):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                
                if format == 'parquet':
                    df.to_parquet(path, index=False)
                elif format == 'csv':
                    df.to_csv(path, index=False)
                else:
                    raise DataValidationError(
                        f"Unsupported output format: {format}",
                        {"format": format, "supported": ["parquet", "csv"]}
                    )
                    
                logger.info(
                    "file_saved",
                    path=str(path),
                    format=format,
                    rows=len(df),
                    columns=len(df.columns),
                    size_mb=path.stat().st_size / (1024 * 1024)
                )
            except Exception as e:
                raise FileOperationError(
                    f"Failed to save file: {path}",
                    {"output_path": str(path), "format": format, "error": str(e)}
                )
    
    @staticmethod
    def get_file_info(file_path: Union[str, Path]) -> dict:
        """Get basic file information"""
        path = Path(file_path)
        if not path.exists():
            raise FileOperationError(
                f"File not found: {file_path}",
                {"file_path": str(file_path)}
            )
        
        with TimedOperation("file_info", file_path=str(path)) as log:
            file_type = FileHandler.detect_file_type(path)
            file_size_mb = path.stat().st_size / (1024 * 1024)
            
            info = {
                'file_path': str(path),
                'file_name': path.name,
                'file_type': file_type,
                'file_size_mb': file_size_mb
            }
            
            # Try to get row count without loading full file
            if file_type == 'csv':
                try:
                    # Count lines (approximation)
                    with open(path, 'r', encoding='utf-8') as f:
                        row_count = sum(1 for line in f) - 1  # Subtract header
                    info['estimated_rows'] = row_count
                except Exception as e:
                    log.warning("csv_row_count_failed", error=str(e))
                    info['estimated_rows'] = None
            elif file_type == 'parquet':
                try:
                    parquet_file = pq.ParquetFile(str(path))
                    info['estimated_rows'] = parquet_file.metadata.num_rows
                except Exception as e:
                    log.warning("parquet_metadata_failed", error=str(e))
                    info['estimated_rows'] = None
                    
            log.info("file_info_retrieved", **info)
            return info