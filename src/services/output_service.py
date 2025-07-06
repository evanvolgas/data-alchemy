"""
Output Service

Handles all file output operations including features, recipes, and evaluation reports.
"""
import json
from pathlib import Path
from typing import Optional, Union, Dict, Any
import pandas as pd

from ..utils.file_handler import FileHandler
from ..models import FeatureSelectionResult, FeatureEngineeringResult, ValidationResult
from ..core import logger, TimedOperation


class OutputService:
    """Service for handling all output operations"""
    
    @staticmethod
    def save_features(
        df: pd.DataFrame,
        output_path: Union[str, Path],
        format: str = 'auto'
    ) -> None:
        """
        Save engineered features to file.
        
        Args:
            df: Dataframe with features to save
            output_path: Path where to save the features
            format: Output format ('csv', 'parquet', or 'auto' to detect from extension)
        """
        output_path = Path(output_path)
        
        with TimedOperation("save_features", output_path=str(output_path)):
            # Auto-detect format if needed
            if format == 'auto':
                format = 'parquet' if output_path.suffix == '.parquet' else 'csv'
            
            FileHandler.save_features(df, output_path, format=format)
            
            logger.info(
                "features_saved",
                feature_count=len(df.columns),
                row_count=len(df),
                output_path=str(output_path),
                format=format
            )
    
    @staticmethod
    def save_feature_recipe(
        results: Dict[str, Any],
        recipe_path: Union[str, Path]
    ) -> None:
        """
        Save a JSON recipe file that can be used to recreate the features.
        
        Args:
            results: Complete results dictionary from pipeline
            recipe_path: Path where to save the recipe file
        """
        recipe_path = Path(recipe_path)
        
        with TimedOperation("save_recipe", recipe_path=str(recipe_path)):
            engineering_result = results.get('engineering')
            selection_result = results.get('selection')
            
            if not engineering_result or not selection_result:
                logger.warning("Cannot create recipe without engineering and selection results")
                return
            
            # Extract feature creation code for selected features
            selected_feature_names = [f.name for f in selection_result.selected_features]
            feature_recipes = []
            
            for feature in engineering_result.engineered_features:
                if feature.name in selected_feature_names:
                    recipe = {
                        'name': feature.name,
                        'type': feature.type.value,
                        'source_columns': feature.source_columns,
                        'description': feature.description,
                        'python_code': feature.metadata.get('python_code', ''),
                        'formula': feature.mathematical_explanation or '',
                        'transformation_type': feature.metadata.get('transformation_type', ''),
                        'importance_score': next(
                            (sf.importance_score for sf in selection_result.selected_features 
                             if sf.name == feature.name), 0.0
                        )
                    }
                    feature_recipes.append(recipe)
            
            # Sort by importance score
            feature_recipes.sort(key=lambda x: x['importance_score'], reverse=True)
            
            recipe_data = {
                'metadata': {
                    'version': '1.0',
                    'created_by': 'DataAlchemy',
                    'total_features': len(feature_recipes),
                    'original_features': len(engineering_result.original_features),
                    'engineered_features': len(engineering_result.engineered_features)
                },
                'features': feature_recipes,
                'instructions': {
                    'description': 'This recipe contains the code to recreate the selected features',
                    'usage': 'Apply the python_code for each feature in the order listed',
                    'requirements': ['pandas', 'numpy', 'scikit-learn']
                }
            }
            
            with open(recipe_path, 'w', encoding='utf-8') as f:
                json.dump(recipe_data, f, indent=2, ensure_ascii=False)
            
            logger.info(
                "recipe_saved",
                feature_count=len(feature_recipes),
                recipe_path=str(recipe_path)
            )
    
    @staticmethod
    def save_evaluation_report(
        results: Dict[str, Any],
        evaluation_path: Union[str, Path]
    ) -> None:
        """
        Save a comprehensive evaluation report in JSON format.
        
        Args:
            results: Complete results dictionary from pipeline
            evaluation_path: Path where to save the evaluation report
        """
        evaluation_path = Path(evaluation_path)
        
        with TimedOperation("save_evaluation", evaluation_path=str(evaluation_path)):
            # Extract key metrics from each stage
            profile = results.get('profile')
            engineering = results.get('engineering')
            selection = results.get('selection')
            validation = results.get('validation')
            file_info = results.get('file_info', {})
            
            evaluation_data = {
                'summary': {
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'file_info': file_info,
                    'data_quality_score': profile.quality_score if profile else None,
                    'features_created': engineering.total_features_created if engineering else 0,
                    'features_selected': len(selection.selected_features) if selection else 0,
                    'validation_score': validation.overall_quality_score if validation else None
                },
                'data_profile': {
                    'quality_score': profile.quality_score if profile else None,
                    'suggested_task_type': profile.suggested_task_type.value if profile else None,
                    'warnings': profile.warnings if profile else [],
                    'domain_hints': profile.domain_hints if profile else [],
                    'processing_time': profile.processing_time_seconds if profile else 0
                } if profile else {},
                'feature_engineering': {
                    'total_features_created': engineering.total_features_created if engineering else 0,
                    'strategies_used': engineering.engineering_strategies_used if engineering else [],
                    'memory_estimate_mb': engineering.memory_estimate_mb if engineering else 0,
                    'warnings': engineering.warnings if engineering else [],
                    'processing_time': engineering.processing_time_seconds if engineering else 0
                } if engineering else {},
                'feature_selection': {
                    'selected_count': len(selection.selected_features) if selection else 0,
                    'removed_count': len(selection.removed_features) if selection else 0,
                    'dimensionality_reduction': selection.dimensionality_reduction if selection else {},
                    'performance_impact': selection.performance_impact_estimate if selection else {},
                    'recommendations': selection.recommendations if selection else [],
                    'processing_time': selection.processing_time_seconds if selection else 0
                } if selection else {},
                'validation': {
                    'overall_quality_score': validation.overall_quality_score if validation else None,
                    'passed_checks': [check.value for check in validation.passed_checks] if validation else [],
                    'failed_checks': [check.value for check in validation.failed_checks] if validation else [],
                    'issues_count': len(validation.issues) if validation else 0,
                    'warnings': validation.warnings if validation else [],
                    'recommendations': validation.recommendations if validation else [],
                    'processing_time': validation.processing_time_seconds if validation else 0
                } if validation else {}
            }
            
            with open(evaluation_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
            
            logger.info(
                "evaluation_saved",
                evaluation_path=str(evaluation_path)
            )