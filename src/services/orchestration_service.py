"""
Orchestration Service

Coordinates the execution of all agents in the feature engineering pipeline.
"""
from typing import Optional, Dict, Any
import pandas as pd

from ..models import (
    PerformanceMode,
    DataProfileResult,
    FeatureEngineeringResult,
    FeatureSelectionResult,
    ValidationResult
)
from ..agents.scout import ScoutAgent
from ..agents.alchemist import AlchemistAgent
from ..agents.curator import CuratorAgent
from ..agents.validator import ValidatorAgent
from ..core import (
    AgentError,
    logger,
    TimedOperation
)


class OrchestrationService:
    """Service that orchestrates the multi-agent feature engineering pipeline"""
    
    def __init__(self, performance_mode: PerformanceMode = PerformanceMode.MEDIUM):
        """
        Initialize the orchestration service.
        
        Args:
            performance_mode: Performance mode for processing (FAST, MEDIUM, THOROUGH)
        """
        self.performance_mode = performance_mode
        self.scout = ScoutAgent()
        self.alchemist = AlchemistAgent()
        self.curator = CuratorAgent()
        self.validator = ValidatorAgent()
    
    async def run_pipeline(
        self,
        df: pd.DataFrame,
        file_path: str,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete feature engineering pipeline.
        
        Args:
            df: Input dataframe
            file_path: Original file path for context
            target_column: Optional target column name
            
        Returns:
            Dictionary containing results from all pipeline stages
            
        Raises:
            AgentError: If any agent fails during pipeline execution
        """
        results = {}
        target_series = df[target_column] if target_column and target_column in df.columns else None
        
        # Stage 1: Scout Agent - Profile data
        try:
            with TimedOperation("scout_agent", performance_mode=self.performance_mode.value):
                profile_result = await self.scout.profile_data(
                    df=df,
                    file_path=file_path,
                    target_column=target_column,
                    performance_mode=self.performance_mode
                )
                results['profile'] = profile_result
                
                logger.info(
                    "scout_completed",
                    quality_score=profile_result.quality_score,
                    task_type=profile_result.suggested_task_type.value
                )
        except Exception as e:
            logger.error("scout_agent_failed", error=str(e))
            raise AgentError("Scout", str(e))
        
        # Stage 2: Alchemist Agent - Engineer features
        try:
            with TimedOperation("alchemist_agent"):
                engineering_result = await self.alchemist.engineer_features(
                    df=df,
                    profile_result=profile_result,
                    target=target_series
                )
                results['engineering'] = engineering_result
                
                logger.info(
                    "alchemist_completed",
                    total_features=engineering_result.total_features_created,
                    strategies=engineering_result.engineering_strategies_used
                )
        except Exception as e:
            logger.error("alchemist_agent_failed", error=str(e))
            raise AgentError("Alchemist", str(e))
        
        # Stage 3: Curator Agent - Select features
        try:
            with TimedOperation("curator_agent"):
                selection_result = await self.curator.select_features(
                    df=df,
                    engineering_result=engineering_result,
                    target=target_series,
                    task_type=profile_result.suggested_task_type,
                    performance_mode=self.performance_mode
                )
                results['selection'] = selection_result
                
                logger.info(
                    "curator_completed",
                    selected_features=len(selection_result.selected_features),
                    removed_features=len(selection_result.removed_features)
                )
        except Exception as e:
            logger.error("curator_agent_failed", error=str(e))
            raise AgentError("Curator", str(e))
        
        # Stage 4: Validator Agent - Validate quality
        try:
            with TimedOperation("validator_agent"):
                validation_result = await self.validator.validate_features(
                    df=df,
                    selection_result=selection_result,
                    target=target_series,
                    task_type=profile_result.suggested_task_type,
                    temporal_column=None,  # Could be detected from profile in future
                    performance_mode=self.performance_mode
                )
                results['validation'] = validation_result
                
                logger.info(
                    "validator_completed",
                    quality_score=validation_result.overall_quality_score,
                    issues_found=len(validation_result.issues)
                )
        except Exception as e:
            logger.error("validator_agent_failed", error=str(e))
            raise AgentError("Validator", str(e))
        
        return results
    
    def get_pipeline_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of the pipeline execution.
        
        Args:
            results: Results dictionary from run_pipeline
            
        Returns:
            Summary dictionary with key metrics
        """
        profile = results.get('profile')
        engineering = results.get('engineering')
        selection = results.get('selection')
        validation = results.get('validation')
        
        return {
            'data_quality_score': profile.quality_score if profile else None,
            'features_created': engineering.total_features_created if engineering else 0,
            'features_selected': len(selection.selected_features) if selection else 0,
            'validation_score': validation.overall_quality_score if validation else None,
            'strategies_used': engineering.engineering_strategies_used if engineering else [],
            'processing_times': {
                'profile': profile.processing_time_seconds if profile else 0,
                'engineering': engineering.processing_time_seconds if engineering else 0,
                'selection': selection.processing_time_seconds if selection else 0,
                'validation': validation.processing_time_seconds if validation else 0
            }
        }