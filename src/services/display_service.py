"""
Display Service

Handles all console display and user interface operations.
"""
from typing import Dict, Any
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from ..models import (
    DataProfileResult,
    FeatureEngineeringResult,
    FeatureSelectionResult,
    ValidationResult
)


class DisplayService:
    """Service for handling console display and user interface"""
    
    def __init__(self):
        self.console = Console()
    
    def create_progress_context(self):
        """Create a progress context manager for tracking operations."""
        return Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=self.console
        )
    
    def print_success(self, message: str):
        """Print a success message."""
        self.console.print(f"✓ {message}")
    
    def print_error(self, message: str):
        """Print an error message."""
        self.console.print(f"[red]✗ {message}")
    
    def print_warning(self, message: str):
        """Print a warning message."""
        self.console.print(f"[yellow]⚠ {message}")
    
    def display_data_loaded(self, rows: int, columns: int):
        """Display data loading success message."""
        self.print_success(f"Loaded {rows:,} rows, {columns} columns")
    
    def display_profile_summary(self, profile: DataProfileResult):
        """Display data profile results in a formatted table."""
        self.console.print("\\n[bold cyan]Data Profile Summary[/bold cyan]")
        
        # Basic info table
        table = Table(title="Dataset Overview")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Rows", f"{profile.context.n_rows:,}")
        table.add_row("Columns", str(profile.context.n_columns))
        table.add_row("Memory", f"{profile.context.memory_usage_mb:.1f} MB")
        table.add_row("Task Type", profile.suggested_task_type.value)
        table.add_row("Quality Score", f"{profile.quality_score:.2f}")
        table.add_row("Processing Time", f"{profile.processing_time_seconds:.2f}s")
        
        self.console.print(table)
        
        # Domain insights
        if profile.domain_hints:
            self.console.print("\\n[bold]Domain Insights:[/bold]")
            for hint in profile.domain_hints:
                self.console.print(f"  • {hint}")
        
        # Warnings
        if profile.warnings:
            self.console.print("\\n[bold yellow]Warnings:[/bold yellow]")
            for warning in profile.warnings:
                self.console.print(f"  ⚠ {warning}")
    
    def display_engineering_summary(self, result: FeatureEngineeringResult):
        """Display feature engineering results summary."""
        self.console.print("\\n[bold cyan]Feature Engineering Summary[/bold cyan]")
        
        table = Table(title="Features Created")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Original Features", str(len(result.original_features)))
        table.add_row("Engineered Features", str(result.total_features_created))
        table.add_row("Total Features", str(len(result.all_features)))
        table.add_row("Feature Matrix Shape", f"{result.feature_matrix_shape}")
        table.add_row("Memory Estimate", f"{result.memory_estimate_mb:.1f} MB")
        table.add_row("Processing Time", f"{result.processing_time_seconds:.2f}s")
        
        self.console.print(table)
        
        # Strategies used
        if result.engineering_strategies_used:
            self.console.print("\\n[bold]Strategies Applied:[/bold]")
            for strategy in result.engineering_strategies_used:
                self.console.print(f"  • {strategy}")
        
        # Sample features
        self.console.print("\\n[bold]Sample Engineered Features:[/bold]")
        for feature in result.engineered_features[:5]:
            self.console.print(f"  • {feature.name}: {feature.description}")
            if feature.mathematical_explanation:
                self.console.print(f"    [dim]{feature.mathematical_explanation}[/dim]")
    
    def display_selection_summary(self, result: FeatureSelectionResult):
        """Display feature selection results."""
        self.console.print("\\n[bold cyan]Feature Selection Summary[/bold cyan]")
        
        table = Table(title="Selection Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Selected Features", str(len(result.selected_features)))
        table.add_row("Removed Features", str(len(result.removed_features)))
        table.add_row("Reduction", f"{result.dimensionality_reduction['reduction_percentage']:.1f}%")
        table.add_row("Expected Accuracy", f"{result.performance_impact_estimate['expected_accuracy_retention']:.1%}")
        table.add_row("Training Speedup", f"{result.performance_impact_estimate['training_speedup']:.1f}x")
        table.add_row("Processing Time", f"{result.processing_time_seconds:.2f}s")
        
        self.console.print(table)
        
        # Top features
        self.console.print("\\n[bold]Top 5 Selected Features:[/bold]")
        for feat in result.selected_features[:5]:
            self.console.print(f"  {feat.rank}. {feat.name} (importance: {feat.importance_score:.3f})")
        
        # Recommendations
        if result.recommendations:
            self.console.print("\\n[bold]Selection Recommendations:[/bold]")
            for rec in result.recommendations:
                self.console.print(f"  • {rec}")
    
    def display_validation_summary(self, result: ValidationResult):
        """Display validation results."""
        self.console.print("\\n[bold cyan]Validation Summary[/bold cyan]")
        
        # Quality score with appropriate color
        quality_color = ("green" if result.overall_quality_score > 0.8 
                        else "yellow" if result.overall_quality_score > 0.6 
                        else "red")
        
        table = Table(title="Validation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Quality Score", f"[{quality_color}]{result.overall_quality_score:.2f}[/{quality_color}]")
        table.add_row("Passed Checks", f"{len(result.passed_checks)}/{len(result.passed_checks) + len(result.failed_checks)}")
        table.add_row("Issues Found", str(len(result.issues)))
        table.add_row("Leakage Risk Features", str(len(result.leakage_risk_features)))
        table.add_row("Processing Time", f"{result.processing_time_seconds:.2f}s")
        
        self.console.print(table)
        
        # Issues by severity
        if result.issues:
            self.console.print("\\n[bold]Issues Found:[/bold]")
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            sorted_issues = sorted(result.issues, key=lambda x: severity_order.get(x.severity, 99))
            
            for issue in sorted_issues:
                severity_colors = {"critical": "red", "high": "yellow", "medium": "cyan", "low": "dim"}
                color = severity_colors.get(issue.severity, "white")
                self.console.print(f"  [{color}]{issue.severity.upper()}[/{color}]: {issue.description}")
        
        # Cross-validation results
        if result.cross_validation_results:
            cv = result.cross_validation_results
            self.console.print(f"\\n[bold]Cross-Validation:[/bold] {cv.mean_score:.3f} ± {cv.std_score:.3f}")
        
        # Recommendations
        self.console.print("\\n[bold]Recommendations:[/bold]")
        for rec in result.recommendations:
            self.console.print(f"  • {rec}")
    
    def display_top_features_analysis(
        self,
        selection_result: FeatureSelectionResult,
        engineering_result: FeatureEngineeringResult,
        validation_result: ValidationResult
    ):
        """Display analysis of top selected features."""
        self.console.print("\\n[bold cyan]Top Features Analysis[/bold cyan]")
        
        # Feature breakdown by type
        feature_types = {}
        for feature in selection_result.selected_features[:10]:
            # Find corresponding engineered feature for type info
            eng_feature = next(
                (f for f in engineering_result.all_features if f.name == feature.name),
                None
            )
            
            if eng_feature:
                feature_type = eng_feature.type.value
                feature_types[feature_type] = feature_types.get(feature_type, 0) + 1
        
        if feature_types:
            self.console.print("\\n[bold]Top 10 Features by Type:[/bold]")
            for ftype, count in sorted(feature_types.items(), key=lambda x: x[1], reverse=True):
                self.console.print(f"  • {ftype}: {count}")
        
        # Feature quality insights
        high_importance = [f for f in selection_result.selected_features if f.importance_score > 0.1]
        self.console.print(f"\\n[bold]Quality Insights:[/bold]")
        self.console.print(f"  • {len(high_importance)} features with high importance (>0.1)")
        self.console.print(f"  • {len(selection_result.selected_features)} total features selected")
        self.console.print(f"  • {validation_result.overall_quality_score:.1%} overall validation quality")
    
    def display_final_summary(self, results: Dict[str, Any]):
        """Display final pipeline summary."""
        self.console.print("\\n[bold green]🎉 DataAlchemy Pipeline Complete![/bold green]")
        
        profile = results.get('profile')
        engineering = results.get('engineering')
        selection = results.get('selection')
        validation = results.get('validation')
        
        summary_table = Table(title="Pipeline Summary")
        summary_table.add_column("Stage", style="cyan")
        summary_table.add_column("Key Metric", style="green")
        summary_table.add_column("Status", style="bold")
        
        if profile:
            summary_table.add_row("Data Profile", f"Quality: {profile.quality_score:.2f}", "✓ Complete")
        
        if engineering:
            summary_table.add_row("Feature Engineering", f"{engineering.total_features_created} features created", "✓ Complete")
        
        if selection:
            summary_table.add_row("Feature Selection", f"{len(selection.selected_features)} features selected", "✓ Complete")
        
        if validation:
            quality_status = "✓ Excellent" if validation.overall_quality_score > 0.8 else "⚠ Good" if validation.overall_quality_score > 0.6 else "⚠ Needs Review"
            summary_table.add_row("Validation", f"Score: {validation.overall_quality_score:.2f}", quality_status)
        
        self.console.print(summary_table)