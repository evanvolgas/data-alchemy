"""
DataAlchemy: Multi-Agent Feature Engineering System (Refactored)

This is the refactored version using service-oriented architecture.
"""
import asyncio
from pathlib import Path
from typing import Optional, Union, Dict, Any
import pandas as pd

from .models import PerformanceMode
from .services import (
    DataService,
    OrchestrationService,
    OutputService,
    DisplayService
)
from .core import DataAlchemyError, logger, TimedOperation


class DataAlchemy:
    """
    Main orchestrator for the multi-agent feature engineering system.
    
    This class provides a high-level interface for the complete feature engineering
    pipeline, delegating specific responsibilities to focused service classes.
    
    Args:
        performance_mode: Performance mode for processing (FAST, MEDIUM, THOROUGH)
        
    Example:
        >>> alchemy = DataAlchemy(performance_mode=PerformanceMode.MEDIUM)
        >>> results = await alchemy.transform_data(
        ...     file_path="data.csv",
        ...     target_column="target",
        ...     output_path="features.parquet"
        ... )
    """
    
    def __init__(self, performance_mode: PerformanceMode = PerformanceMode.MEDIUM):
        self.performance_mode = performance_mode
        self.data_service = DataService()
        self.orchestration_service = OrchestrationService(performance_mode)
        self.output_service = OutputService()
        self.display_service = DisplayService()
    
    async def transform_data(
        self,
        file_path: Union[str, Path],
        target_column: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
        sample_size: Optional[int] = None,
        evaluation_output: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Transform raw data into engineered features using the multi-agent pipeline.
        
        This method orchestrates the complete feature engineering workflow:
        1. Load and validate input data
        2. Profile data characteristics (Scout Agent)
        3. Engineer new features (Alchemist Agent)
        4. Select optimal features (Curator Agent)
        5. Validate feature quality (Validator Agent)
        6. Save results and generate reports
        
        Args:
            file_path: Path to input CSV or Parquet file
            target_column: Target column for supervised learning (optional)
            output_path: Path to save engineered features (optional)
            sample_size: Limit number of rows for testing (optional)
            evaluation_output: Path to save detailed evaluation report (optional)
            
        Returns:
            Dictionary containing results from each pipeline stage:
            - file_info: Input file metadata
            - profile: Data profiling results
            - engineering: Feature engineering results
            - selection: Feature selection results
            - validation: Feature validation results
            
        Raises:
            DataAlchemyError: If any stage of the pipeline fails
            TargetNotFoundError: If target_column is specified but not found
            InsufficientDataError: If dataset has too few rows for processing
            
        Example:
            >>> results = await alchemy.transform_data(
            ...     file_path="sales_data.csv",
            ...     target_column="revenue",
            ...     output_path="features.parquet",
            ...     evaluation_output="evaluation.json"
            ... )
            >>> print(f"Created {len(results['selection'].selected_features)} features")
        """
        file_path = Path(file_path)
        
        with self.display_service.create_progress_context() as progress:
            
            # Stage 1: Load and validate data
            task = progress.add_task("[cyan]Loading data...", total=None)
            try:
                df, file_info = self.data_service.load_and_validate(
                    file_path=file_path,
                    target_column=target_column,
                    sample_size=sample_size
                )
                progress.update(task, completed=100)
                self.display_service.display_data_loaded(len(df), len(df.columns))
                
            except DataAlchemyError:
                raise
            except Exception as e:
                self.display_service.print_error(f"Error loading file: {e}")
                logger.error("data_loading_failed", error=str(e))
                raise DataAlchemyError(f"Failed to load data: {e}")
            
            # Stage 2-5: Execute the multi-agent pipeline
            task = progress.add_task("[cyan]Running multi-agent pipeline...", total=None)
            try:
                results = await self.orchestration_service.run_pipeline(
                    df=df,
                    file_path=str(file_path),
                    target_column=target_column
                )
                results['file_info'] = file_info
                progress.update(task, completed=100)
                
                # Display results for each stage
                self.display_service.display_profile_summary(results['profile'])
                self.display_service.display_engineering_summary(results['engineering'])
                self.display_service.display_selection_summary(results['selection'])
                self.display_service.display_validation_summary(results['validation'])
                
            except DataAlchemyError:
                raise
            except Exception as e:
                self.display_service.print_error(f"Pipeline execution failed: {e}")
                logger.error("pipeline_execution_failed", error=str(e))
                raise DataAlchemyError(f"Pipeline failed: {e}")
            
            # Stage 6: Save outputs if requested
            if output_path:
                await self._save_features(df, results, output_path, target_column, progress)
            
            if evaluation_output:
                await self._save_evaluation(results, evaluation_output, progress)
            
            # Display final analysis and summary
            self.display_service.display_top_features_analysis(
                results['selection'],
                results['engineering'],
                results['validation']
            )
            self.display_service.display_final_summary(results)
        
        return results
    
    async def _save_features(
        self, 
        df: pd.DataFrame, 
        results: Dict[str, Any], 
        output_path: Union[str, Path],
        target_column: Optional[str],
        progress
    ):
        """Save selected features to output file."""
        task = progress.add_task("[cyan]Saving selected features...", total=None)
        try:
            with TimedOperation("save_features", output_path=str(output_path)):
                selected_feature_names = [f.name for f in results['selection'].selected_features]
                output_df = self.data_service.prepare_output_dataframe(
                    df, selected_feature_names, target_column
                )
                
                self.output_service.save_features(output_df, output_path)
                progress.update(task, completed=100)
                self.display_service.print_success(f"Saved {len(selected_feature_names)} features to {output_path}")
                
                # Also save feature recipe
                recipe_path = str(output_path).replace('.parquet', '_recipe.json').replace('.csv', '_recipe.json')
                self.output_service.save_feature_recipe(results, recipe_path)
                
        except Exception as e:
            self.display_service.print_error(f"Error saving features: {e}")
            logger.error("save_features_failed", error=str(e))
    
    async def _save_evaluation(
        self, 
        results: Dict[str, Any], 
        evaluation_output: Union[str, Path],
        progress
    ):
        """Save evaluation report to file."""
        task = progress.add_task("[cyan]Saving evaluation report...", total=None)
        try:
            self.output_service.save_evaluation_report(results, evaluation_output)
            progress.update(task, completed=100)
            self.display_service.print_success(f"Saved evaluation report to {evaluation_output}")
            
        except Exception as e:
            self.display_service.print_error(f"Error saving evaluation: {e}")
            logger.error("save_evaluation_failed", error=str(e))
    
    def get_pipeline_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a summary of pipeline execution metrics.
        
        Args:
            results: Results dictionary from transform_data
            
        Returns:
            Summary dictionary with key performance metrics
            
        Example:
            >>> summary = alchemy.get_pipeline_summary(results)
            >>> print(f"Pipeline created {summary['features_created']} features")
            >>> print(f"Final validation score: {summary['validation_score']:.2f}")
        """
        return self.orchestration_service.get_pipeline_summary(results)


def run_data_alchemy(
    file_path: Union[str, Path],
    target_column: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    sample_size: Optional[int] = None,
    evaluation_output: Optional[Union[str, Path]] = None,
    performance_mode: str = "medium"
) -> Dict[str, Any]:
    """
    Convenience function to run DataAlchemy synchronously.
    
    This is a simplified interface for users who want to run the complete
    feature engineering pipeline with a single function call.
    
    Args:
        file_path: Path to input CSV or Parquet file
        target_column: Target column for supervised learning (optional)
        output_path: Path to save engineered features (optional)
        sample_size: Limit number of rows for testing (optional)
        evaluation_output: Path to save detailed evaluation report (optional)
        performance_mode: Processing mode ('fast', 'medium', 'thorough')
        
    Returns:
        Dictionary containing complete pipeline results
        
    Example:
        >>> results = run_data_alchemy(
        ...     file_path="sales_data.csv",
        ...     target_column="revenue",
        ...     output_path="features.parquet",
        ...     performance_mode="thorough"
        ... )
        >>> print(f"Selected {len(results['selection'].selected_features)} features")
        
    Note:
        This function runs the async transform_data method synchronously.
        For async applications, use DataAlchemy.transform_data directly.
    """
    try:
        mode = PerformanceMode(performance_mode.lower())
    except ValueError:
        raise ValueError(f"Invalid performance_mode: {performance_mode}. "
                        f"Must be one of: {[m.value for m in PerformanceMode]}")
    
    alchemy = DataAlchemy(performance_mode=mode)
    
    return asyncio.run(
        alchemy.transform_data(
            file_path=file_path,
            target_column=target_column,
            output_path=output_path,
            sample_size=sample_size,
            evaluation_output=evaluation_output
        )
    )