"""
DataAlchemy: Multi-Agent Feature Engineering System
"""
import asyncio
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import json
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from .models import (
    PerformanceMode, 
    DataProfileResult, 
    FeatureEngineeringResult,
    FeatureSelectionResult,
    ValidationResult
)
from .agents.scout import ScoutAgent
from .agents.alchemist import AlchemistAgent
from .agents.curator import CuratorAgent
from .agents.validator import ValidatorAgent
from .utils.file_handler import FileHandler


console = Console()


class DataAlchemy:
    """Main orchestrator for the multi-agent feature engineering system"""
    
    def __init__(self, performance_mode: PerformanceMode = PerformanceMode.MEDIUM):
        self.performance_mode = performance_mode
        self.scout = ScoutAgent()
        self.alchemist = AlchemistAgent()
        self.curator = CuratorAgent()
        self.validator = ValidatorAgent()
        
    async def transform_data(
        self,
        file_path: Union[str, Path],
        target_column: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
        sample_size: Optional[int] = None,
        evaluation_output: Optional[Union[str, Path]] = None
    ) -> dict:
        """Transform raw data into engineered features
        
        Args:
            file_path: Path to input CSV/Parquet file
            target_column: Target column for supervised learning (optional)
            output_path: Path to save engineered features (optional)
            sample_size: Limit rows for testing (optional)
            evaluation_output: Path to save feature evaluation JSON (optional)
            
        Returns:
            Dictionary with results from each agent
        """
        file_path = Path(file_path)
        
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            # Load data
            task = progress.add_task("[cyan]Loading data...", total=None)
            try:
                df = FileHandler.read_file(file_path, sample_size=sample_size)
                file_info = FileHandler.get_file_info(file_path)
                progress.update(task, completed=100)
                console.print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
            except Exception as e:
                console.print(f"[red]✗ Error loading file: {e}")
                raise
            
            # Scout Agent: Profile data
            task = progress.add_task("[cyan]Scout Agent: Profiling data...", total=None)
            try:
                profile_result = await self.scout.profile_data(
                    df=df,
                    file_path=str(file_path),
                    target_column=target_column,
                    performance_mode=self.performance_mode
                )
                progress.update(task, completed=100)
                self._display_profile_summary(profile_result)
            except Exception as e:
                console.print(f"[red]✗ Scout Agent failed: {e}")
                raise
            
            # Alchemist Agent: Engineer features
            task = progress.add_task("[cyan]Alchemist Agent: Engineering features...", total=None)
            try:
                engineering_result = await self.alchemist.engineer_features(
                    df=df,
                    profile_result=profile_result,
                    target=df[target_column] if target_column and target_column in df.columns else None
                )
                progress.update(task, completed=100)
                self._display_engineering_summary(engineering_result)
            except Exception as e:
                console.print(f"[red]✗ Alchemist Agent failed: {e}")
                raise
            
            # Curator Agent: Feature selection
            task = progress.add_task("[cyan]Curator Agent: Selecting features...", total=None)
            try:
                selection_result = await self.curator.select_features(
                    df=df,
                    engineering_result=engineering_result,
                    target=df[target_column] if target_column and target_column in df.columns else None,
                    task_type=profile_result.suggested_task_type,
                    performance_mode=self.performance_mode
                )
                progress.update(task, completed=100)
                self._display_selection_summary(selection_result)
            except Exception as e:
                console.print(f"[red]✗ Curator Agent failed: {e}")
                raise
            
            # Validator Agent: Quality assurance
            task = progress.add_task("[cyan]Validator Agent: Validating features...", total=None)
            try:
                validation_result = await self.validator.validate_features(
                    df=df,
                    selection_result=selection_result,
                    target=df[target_column] if target_column and target_column in df.columns else None,
                    task_type=profile_result.suggested_task_type,
                    temporal_column=None,  # Could be detected from profile
                    performance_mode=self.performance_mode
                )
                progress.update(task, completed=100)
                self._display_validation_summary(validation_result)
            except Exception as e:
                console.print(f"[red]✗ Validator Agent failed: {e}")
                raise
            
            # Prepare results
            results = {
                'file_info': file_info,
                'profile': profile_result,
                'engineering': engineering_result,
                'selection': selection_result,
                'validation': validation_result
            }
            
            # Save features if output path provided
            if output_path:
                task = progress.add_task("[cyan]Saving selected features...", total=None)
                try:
                    # Get selected feature names (excluding target if it was accidentally included)
                    selected_features = [f.name for f in selection_result.selected_features 
                                       if f.name != target_column]
                    
                    # Create dataframe with only selected features
                    output_df = df[selected_features].copy()
                    
                    # Add target column at the end if present
                    if target_column and target_column in df.columns:
                        output_df[target_column] = df[target_column]
                    
                    output_format = 'parquet' if str(output_path).endswith('.parquet') else 'csv'
                    FileHandler.save_features(output_df, output_path, format=output_format)
                    progress.update(task, completed=100)
                    console.print(f"✓ Saved {len(selected_features)} features to {output_path}")
                except Exception as e:
                    console.print(f"[red]✗ Error saving features: {e}")
            
            # Save evaluation output if requested
            if evaluation_output:
                self._save_evaluation(results, evaluation_output)
            
            # Also save a feature recipe file
            if output_path:
                recipe_path = str(output_path).replace('.parquet', '_recipe.json').replace('.csv', '_recipe.json')
                self._save_feature_recipe(results, recipe_path)
            
            # Display top features analysis
            self._display_top_features_analysis(
                selection_result, 
                engineering_result,
                validation_result
            )
            
        return results
    
    def _display_profile_summary(self, profile: DataProfileResult):
        """Display profile results in a nice table"""
        console.print("\n[bold cyan]Data Profile Summary[/bold cyan]")
        
        # Basic info
        table = Table(title="Dataset Overview")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Rows", f"{profile.context.n_rows:,}")
        table.add_row("Columns", str(profile.context.n_columns))
        table.add_row("Memory", f"{profile.context.memory_usage_mb:.1f} MB")
        table.add_row("Task Type", profile.suggested_task_type.value)
        table.add_row("Quality Score", f"{profile.quality_score:.2f}")
        table.add_row("Processing Time", f"{profile.processing_time_seconds:.2f}s")
        
        console.print(table)
        
        # Domain hints
        if profile.domain_hints:
            console.print("\n[bold]Domain Insights:[/bold]")
            for hint in profile.domain_hints:
                console.print(f"  • {hint}")
        
        # Warnings
        if profile.warnings:
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in profile.warnings:
                console.print(f"  ⚠ {warning}")
    
    def _display_engineering_summary(self, result: FeatureEngineeringResult):
        """Display engineering results summary"""
        console.print("\n[bold cyan]Feature Engineering Summary[/bold cyan]")
        
        table = Table(title="Features Created")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Original Features", str(len(result.original_features)))
        table.add_row("Engineered Features", str(result.total_features_created))
        table.add_row("Total Features", str(len(result.all_features)))
        table.add_row("Feature Matrix Shape", f"{result.feature_matrix_shape}")
        table.add_row("Memory Estimate", f"{result.memory_estimate_mb:.1f} MB")
        table.add_row("Processing Time", f"{result.processing_time_seconds:.2f}s")
        
        console.print(table)
        
        # Strategies used
        if result.engineering_strategies_used:
            console.print("\n[bold]Strategies Applied:[/bold]")
            for strategy in result.engineering_strategies_used:
                console.print(f"  • {strategy}")
        
        # Show sample features
        console.print("\n[bold]Sample Engineered Features:[/bold]")
        for feature in result.engineered_features[:5]:
            console.print(f"  • {feature.name}: {feature.description}")
            if feature.mathematical_explanation:
                console.print(f"    [dim]{feature.mathematical_explanation}[/dim]")
    
    def _display_selection_summary(self, result: FeatureSelectionResult):
        """Display feature selection results"""
        console.print("\n[bold cyan]Feature Selection Summary[/bold cyan]")
        
        table = Table(title="Selection Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Selected Features", str(len(result.selected_features)))
        table.add_row("Removed Features", str(len(result.removed_features)))
        table.add_row("Reduction", f"{result.dimensionality_reduction['reduction_percentage']:.1f}%")
        table.add_row("Expected Accuracy", f"{result.performance_impact_estimate['expected_accuracy_retention']:.1%}")
        table.add_row("Training Speedup", f"{result.performance_impact_estimate['training_speedup']:.1f}x")
        table.add_row("Processing Time", f"{result.processing_time_seconds:.2f}s")
        
        console.print(table)
        
        # Top features
        console.print("\n[bold]Top 5 Selected Features:[/bold]")
        for feat in result.selected_features[:5]:
            console.print(f"  {feat.rank}. {feat.name} (importance: {feat.importance_score:.3f})")
        
        # Recommendations
        if result.recommendations:
            console.print("\n[bold]Selection Recommendations:[/bold]")
            for rec in result.recommendations:
                console.print(f"  • {rec}")
    
    def _display_validation_summary(self, result: ValidationResult):
        """Display validation results"""
        console.print("\n[bold cyan]Validation Summary[/bold cyan]")
        
        # Quality score with color
        quality_color = "green" if result.overall_quality_score > 0.8 else "yellow" if result.overall_quality_score > 0.6 else "red"
        
        table = Table(title="Validation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Quality Score", f"[{quality_color}]{result.overall_quality_score:.2f}[/{quality_color}]")
        table.add_row("Passed Checks", f"{len(result.passed_checks)}/{len(result.passed_checks) + len(result.failed_checks)}")
        table.add_row("Issues Found", str(len(result.issues)))
        table.add_row("Leakage Risk Features", str(len(result.leakage_risk_features)))
        table.add_row("Processing Time", f"{result.processing_time_seconds:.2f}s")
        
        console.print(table)
        
        # Issues by severity
        if result.issues:
            console.print("\n[bold]Issues Found:[/bold]")
            for issue in sorted(result.issues, key=lambda x: ["low", "medium", "high", "critical"].index(x.severity), reverse=True):
                severity_color = {"critical": "red", "high": "yellow", "medium": "cyan", "low": "dim"}[issue.severity]
                console.print(f"  [{severity_color}]{issue.severity.upper()}[/{severity_color}]: {issue.description}")
        
        # Cross-validation results
        if result.cross_validation_results:
            cv = result.cross_validation_results
            console.print(f"\n[bold]Cross-Validation:[/bold] {cv.mean_score:.3f} ± {cv.std_score:.3f}")
        
        # Final recommendations
        console.print("\n[bold]Recommendations:[/bold]")
        for rec in result.recommendations:
            console.print(f"  • {rec}")
    
    def _save_evaluation(self, results: dict, output_path: Union[str, Path]):
        """Save feature evaluation to JSON file"""
        import json
        from datetime import datetime
        
        evaluation = {
            "timestamp": datetime.now().isoformat(),
            "file_info": results['file_info'],
            "profile": {
                "quality_score": results['profile'].quality_score,
                "task_type": results['profile'].suggested_task_type.value,
                "n_rows": results['profile'].context.n_rows,
                "n_columns": results['profile'].context.n_columns,
                "domain_hints": results['profile'].domain_hints
            },
            "engineering": {
                "original_features": len(results['engineering'].original_features),
                "engineered_features": results['engineering'].total_features_created,
                "strategies_used": results['engineering'].engineering_strategies_used,
                "feature_types_created": self._get_feature_type_breakdown(results['engineering'])
            },
            "selection": {
                "selected_features": len(results['selection'].selected_features),
                "removed_features": len(results['selection'].removed_features),
                "reduction_percentage": results['selection'].dimensionality_reduction['reduction_percentage']
            },
            "validation": {
                "quality_score": results['validation'].overall_quality_score,
                "passed_checks": [check.value for check in results['validation'].passed_checks],
                "failed_checks": [check.value for check in results['validation'].failed_checks],
                "issues": [
                    {
                        "type": issue.check_type.value,
                        "severity": issue.severity,
                        "description": issue.description
                    }
                    for issue in results['validation'].issues
                ]
            },
            "features": [
                {
                    "name": feat.name,
                    "importance": feat.importance_score,
                    "rank": feat.rank,
                    "methods": [m.value for m in feat.selection_methods] if feat.selection_methods else [],
                    "details": self._get_feature_info(feat, results['engineering'])
                }
                for feat in sorted(results['selection'].selected_features, 
                                 key=lambda x: x.importance_score, reverse=True)
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(evaluation, f, indent=2)
        
        console.print(f"\n✓ Saved feature evaluation to {output_path}")
    
    def _save_feature_recipe(self, results: dict, output_path: str):
        """Save a user-friendly feature recipe"""
        # Separate original and engineered features
        original_features = []
        engineered_features = []
        
        for feat in results['selection'].selected_features:
            feat_info = self._get_feature_info(feat, results['engineering'])
            
            if feat_info['category'] == 'original':
                original_features.append({
                    'name': feat.name,
                    'importance': round(feat.importance_score, 4),
                    'data_type': feat_info.get('data_type', 'unknown')
                })
            elif feat_info['category'] == 'engineered':
                engineered_features.append({
                    'name': feat.name,
                    'importance': round(feat.importance_score, 4),
                    'type': feat_info['feature_type'],
                    'source_columns': feat_info['source_columns'],
                    'description': feat_info['description'],
                    'python_code': feat_info.get('how_to_recreate', ''),
                    'formula': feat_info.get('mathematical_explanation', '')
                })
        
        # Sort by importance
        original_features.sort(key=lambda x: x['importance'], reverse=True)
        engineered_features.sort(key=lambda x: x['importance'], reverse=True)
        
        recipe = {
            'dataset_info': {
                'total_features_selected': len(results['selection'].selected_features),
                'original_features_selected': len(original_features),
                'engineered_features_selected': len(engineered_features),
                'feature_reduction': f"{results['selection'].dimensionality_reduction['reduction_percentage']:.1f}%"
            },
            'original_features': original_features,
            'engineered_features': engineered_features,
            'usage_instructions': {
                'description': 'To recreate these features, use the Python code provided for each engineered feature.',
                'required_imports': [
                    'import pandas as pd',
                    'import numpy as np'
                ],
                'example': 'df[feature_name] = <python_code_from_recipe>'
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(recipe, f, indent=2)
        
        console.print(f"✓ Saved feature recipe to {output_path}")
    
    def _get_feature_type_breakdown(self, engineering_result: FeatureEngineeringResult) -> dict:
        """Get breakdown of engineered features by type"""
        from collections import Counter
        
        type_counts = Counter()
        for feature in engineering_result.engineered_features:
            type_counts[feature.type.value] += 1
        
        return dict(type_counts)
    
    def _get_recreation_code(self, feature) -> str:
        """Generate code to recreate the engineered feature"""
        feat_type = feature.type.value
        name = feature.name
        sources = feature.source_columns
        src = sources[0] if sources else 'column'
        
        # Map feature types to their transformation code
        transformations = {
            # Mathematical transformations
            "polynomial": {
                "_squared": f"df['{name}'] = df['{src}'] ** 2",
                "_cubed": f"df['{name}'] = df['{src}'] ** 3",
                "default": f"df['{name}'] = df['{src}'] ** N  # Check feature description"
            },
            "sqrt_transform": f"df['{name}'] = np.sqrt(df['{src}'].fillna(0))",
            "log_transform": f"df['{name}'] = np.log1p(df['{src}'])",
            
            # Feature interactions
            "interaction": f"df['{name}'] = df['{sources[0]}'] * df['{sources[1]}']" if len(sources) >= 2 else "# Need two source columns",
            "ratio": f"df['{name}'] = df['{sources[0]}'] / (df['{sources[1]}'] + 1e-8)" if len(sources) >= 2 else "# Need two source columns",
            
            # Categorical encodings
            "binned": f"df['{name}'] = pd.qcut(df['{src}'], 10, labels=False, duplicates='drop')",
            "frequency_encoded": f"freq_map = df['{src}'].value_counts().to_dict()\ndf['{name}'] = df['{src}'].map(freq_map)",
            "one_hot": f"df['{name}'] = (df['{src}'] == '{name.split('_is_')[-1]}').astype(int)" if '_is_' in name else f"df['{name}'] = pd.get_dummies(df['{src}'])",
            
            # Date/time features - using the exact transformations from alchemist.py
            "date_component": self._get_date_component_code(name, src),
            "temporal_year": f"df['{name}'] = pd.to_datetime(df['{src}']).dt.year",
            "temporal_month": f"df['{name}'] = pd.to_datetime(df['{src}']).dt.month", 
            "temporal_day": f"df['{name}'] = pd.to_datetime(df['{src}']).dt.day",
            "temporal_dayofweek": f"df['{name}'] = pd.to_datetime(df['{src}']).dt.dayofweek",
            "temporal_hour": f"df['{name}'] = pd.to_datetime(df['{src}']).dt.hour",
            "temporal_sin": f"df['{name}'] = np.sin(2 * np.pi * pd.to_datetime(df['{src}']).dt.month / 12)",
            "temporal_cos": f"df['{name}'] = np.cos(2 * np.pi * pd.to_datetime(df['{src}']).dt.month / 12)",
            "is_weekend": f"df['{name}'] = pd.to_datetime(df['{src}']).dt.dayofweek.isin([5, 6]).astype(int)"
        }
        
        # Handle polynomial special cases
        if feat_type == "polynomial":
            for suffix, code in transformations["polynomial"].items():
                if suffix != "default" and name.endswith(suffix):
                    return code
            return transformations["polynomial"]["default"]
        
        # Return the transformation or a placeholder
        return transformations.get(feat_type, f"# Feature type '{feat_type}' not mapped - check feature.description")
    
    def _get_date_component_code(self, name: str, source: str) -> str:
        """Get the appropriate date component extraction code based on feature name"""
        # Map common date component patterns
        if name.endswith("_year"):
            return f"df['{name}'] = pd.to_datetime(df['{source}']).dt.year"
        elif name.endswith("_month") and not ("sin" in name or "cos" in name):
            return f"df['{name}'] = pd.to_datetime(df['{source}']).dt.month"
        elif name.endswith("_day") and "dayofweek" not in name:
            return f"df['{name}'] = pd.to_datetime(df['{source}']).dt.day"
        elif "dayofweek" in name:
            return f"df['{name}'] = pd.to_datetime(df['{source}']).dt.dayofweek"
        elif name.endswith("_hour"):
            return f"df['{name}'] = pd.to_datetime(df['{source}']).dt.hour"
        elif "month_sin" in name:
            return f"df['{name}'] = np.sin(2 * np.pi * pd.to_datetime(df['{source}']).dt.month / 12)"
        elif "month_cos" in name:
            return f"df['{name}'] = np.cos(2 * np.pi * pd.to_datetime(df['{source}']).dt.month / 12)"
        elif "is_weekend" in name:
            return f"df['{name}'] = pd.to_datetime(df['{source}']).dt.dayofweek.isin([5, 6]).astype(int)"
        else:
            return f"# Date component extraction for '{name}' - check feature description"
    
    def _get_feature_info(self, selected_feature, engineering_result: FeatureEngineeringResult) -> dict:
        """Get detailed info about a selected feature"""
        # Check if it's an original feature
        original_feature_names = {f.name for f in engineering_result.original_features}
        
        if selected_feature.name in original_feature_names:
            # Find the original feature
            for orig in engineering_result.original_features:
                if orig.name == selected_feature.name:
                    return {
                        "category": "original",
                        "data_type": orig.type.value if hasattr(orig.type, 'value') else str(orig.type),
                        "description": orig.description
                    }
        
        # Find the engineered feature
        for eng in engineering_result.engineered_features:
            if eng.name == selected_feature.name:
                # Create how-to-recreate instructions based on feature type
                recreation_code = self._get_recreation_code(eng)
                
                return {
                    "category": "engineered",
                    "feature_type": eng.type.value,
                    "source_columns": eng.source_columns,
                    "description": eng.description,
                    "mathematical_explanation": eng.mathematical_explanation,
                    "computational_complexity": eng.computational_complexity,
                    "how_to_recreate": recreation_code
                }
        
        # Default if not found
        return {
            "category": "unknown",
            "description": f"Feature {selected_feature.name}"
        }
    
    def _display_top_features_analysis(
        self,
        selection_result: FeatureSelectionResult,
        engineering_result: FeatureEngineeringResult,
        validation_result: ValidationResult
    ):
        """Display analysis of top regular and engineered features"""
        console.print("\n[bold cyan]Top Features Analysis[/bold cyan]")
        
        # Separate original and engineered features
        selected_features = selection_result.selected_features
        
        original_feature_names = {f.name for f in engineering_result.original_features}
        
        original_selected = [f for f in selected_features if f.name in original_feature_names]
        engineered_selected = [f for f in selected_features if f.name not in original_feature_names]
        
        # Display top 5 original features
        console.print("\n[bold]Top 5 Original Features:[/bold]")
        if original_selected:
            for i, feat in enumerate(original_selected[:5]):
                console.print(f"\n  {i+1}. [green]{feat.name}[/green]")
                console.print(f"     Importance: {feat.importance_score:.3f}")
                if feat.selection_methods:
                    console.print(f"     Selection methods: {', '.join(m.value for m in feat.selection_methods)}")
                
                # Find original feature info
                orig_feat = next((f for f in engineering_result.original_features if f.name == feat.name), None)
                if orig_feat:
                    console.print(f"     Type: {orig_feat.type.value}")
                    if orig_feat.description:
                        console.print(f"     Description: {orig_feat.description}")
                
                # Add stability info if available
                if feat.name in validation_result.feature_stability_scores:
                    stability = validation_result.feature_stability_scores[feat.name]
                    console.print(f"     Stability score: {stability:.3f}")
        else:
            console.print("  No original features selected")
        
        # Display top 5 engineered features
        console.print("\n[bold]Top 5 Engineered Features:[/bold]")
        if engineered_selected:
            for i, feat in enumerate(engineered_selected[:5]):
                console.print(f"\n  {i+1}. [yellow]{feat.name}[/yellow]")
                console.print(f"     Importance: {feat.importance_score:.3f}")
                if feat.selection_methods:
                    console.print(f"     Selection methods: {', '.join(m.value for m in feat.selection_methods)}")
                
                # Find engineered feature info
                eng_feat = next((f for f in engineering_result.engineered_features if f.name == feat.name), None)
                if eng_feat:
                    console.print(f"     Type: {eng_feat.type.value}")
                    console.print(f"     Source columns: {', '.join(eng_feat.source_columns)}")
                    if eng_feat.description:
                        console.print(f"     Description: {eng_feat.description}")
                    if eng_feat.mathematical_explanation:
                        console.print(f"     Formula: {eng_feat.mathematical_explanation}")
                    console.print(f"     Complexity: {eng_feat.computational_complexity}")
                
                # Add stability info
                if feat.name in validation_result.feature_stability_scores:
                    stability = validation_result.feature_stability_scores[feat.name]
                    console.print(f"     Stability score: {stability:.3f}")
        else:
            console.print("  No engineered features selected")
        
        # Summary statistics
        console.print(f"\n[bold]Feature Summary:[/bold]")
        console.print(f"  Original features selected: {len(original_selected)}/{len(engineering_result.original_features)}")
        console.print(f"  Engineered features selected: {len(engineered_selected)}/{engineering_result.total_features_created}")
        
        if selected_features:
            avg_importance = sum(f.importance_score for f in selected_features) / len(selected_features)
            console.print(f"  Average importance score: {avg_importance:.3f}")


def run_data_alchemy(
    file_path: Union[str, Path],
    target_column: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    performance_mode: str = "medium",
    sample_size: Optional[int] = None,
    evaluation_output: Optional[Union[str, Path]] = None
) -> dict:
    """Convenience function to run DataAlchemy synchronously"""
    mode = PerformanceMode(performance_mode.lower())
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