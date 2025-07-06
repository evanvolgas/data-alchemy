import pandas as pd
from pathlib import Path
from typing import Union, Optional
import pyarrow.parquet as pq


class FileHandler:
    """Handle reading CSV and Parquet files with auto-detection"""
    
    @staticmethod
    def detect_file_type(file_path: Union[str, Path]) -> str:
        """Detect file type from extension"""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension in ['.csv', '.tsv', '.txt']:
            return 'csv'
        elif extension in ['.parquet', '.pq']:
            return 'parquet'
        else:
            # Try to infer from content
            try:
                # Try reading as parquet first (binary format)
                pq.read_schema(str(path))
                return 'parquet'
            except:
                # Assume CSV if parquet fails
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
            raise FileNotFoundError(f"File not found: {file_path}")
        
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
                        return df
                except Exception:
                    continue
                    
            # If all separators fail, use comma as default
            return pd.read_csv(path, **read_kwargs)
            
        elif file_type == 'parquet':
            df = pd.read_parquet(path, **kwargs)
            if sample_size and len(df) > sample_size:
                return df.head(sample_size)
            return df
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
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
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'parquet':
            df.to_parquet(path, index=False)
        elif format == 'csv':
            df.to_csv(path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {format}")
    
    @staticmethod
    def get_file_info(file_path: Union[str, Path]) -> dict:
        """Get basic file information"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
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
            except:
                info['estimated_rows'] = None
        elif file_type == 'parquet':
            try:
                parquet_file = pq.ParquetFile(str(path))
                info['estimated_rows'] = parquet_file.metadata.num_rows
            except:
                info['estimated_rows'] = None
                
        return info