"""
MangroVision - Detection Quality Validation Module
Validate tree crown detection results for omission & commission errors
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import typer

console = Console()
app = typer.Typer()


class DetectionValidator:
    """Validate tree crown detection quality"""
    
    def __init__(self, geojson_path: str):
        """
        Initialize validator with detection results
        
        Args:
            geojson_path: Path to GeoJSON file containing detected crowns
        """
        self.geojson_path = geojson_path
        self.features = self._load_geojson()
        
    def _load_geojson(self) -> List[dict]:
        """Load GeoJSON features"""
        with open(self.geojson_path, 'r') as f:
            geojson = json.load(f)
        return geojson.get('features', [])
    
    def calculate_area_statistics(self) -> Dict:
        """
        Calculate crown area statistics in m²
        Validates if detected sizes are realistic for the region
        
        Returns:
            Dictionary with area statistics
        """
        if not self.features:
            return {
                'count': 0,
                'total_area_m2': 0.0,
                'mean_area_m2': 0.0,
                'median_area_m2': 0.0,
                'min_area_m2': 0.0,
                'max_area_m2': 0.0,
                'std_area_m2': 0.0
            }
        
        areas = []
        for feature in self.features:
            props = feature.get('properties', {})
            area = props.get('area', 0.0)  # Area in m²
            if area > 0:
                areas.append(area)
        
        if not areas:
            return {'count': 0, 'total_area_m2': 0.0}
        
        areas_np = np.array(areas)
        
        return {
            'count': len(areas),
            'total_area_m2': float(np.sum(areas_np)),
            'mean_area_m2': float(np.mean(areas_np)),
            'median_area_m2': float(np.median(areas_np)),
            'min_area_m2': float(np.min(areas_np)),
            'max_area_m2': float(np.max(areas_np)),
            'std_area_m2': float(np.std(areas_np)),
            'percentile_25_m2': float(np.percentile(areas_np, 25)),
            'percentile_75_m2': float(np.percentile(areas_np, 75))
        }
    
    def validate_crown_sizes(self, 
                            min_expected_m2: float = 0.5,
                            max_expected_m2: float = 50.0) -> Dict:
        """
        Validate if detected crown sizes are realistic
        
        Typical tropical/mangrove tree crown sizes:
        - Young trees: 0.5 - 5 m²
        - Mature trees: 5 - 30 m²
        - Very large trees: 30 - 50 m²
        
        Args:
            min_expected_m2: Minimum expected crown area
            max_expected_m2: Maximum expected crown area
            
        Returns:
            Dictionary with validation results
        """
        areas = []
        for feature in self.features:
            props = feature.get('properties', {})
            area = props.get('area', 0.0)
            if area > 0:
                areas.append(area)
        
        if not areas:
            return {
                'valid': False,
                'reason': 'No detections with area information'
            }
        
        too_small = sum(1 for a in areas if a < min_expected_m2)
        too_large = sum(1 for a in areas if a > max_expected_m2)
        valid = sum(1 for a in areas if min_expected_m2 <= a <= max_expected_m2)
        
        total = len(areas)
        valid_percent = (valid / total) * 100 if total > 0 else 0
        
        return {
            'total_detections': total,
            'valid_size': valid,
            'too_small': too_small,
            'too_large': too_large,
            'valid_percent': valid_percent,
            'expected_range_m2': f"{min_expected_m2} - {max_expected_m2}",
            'passed': valid_percent >= 70.0  # At least 70% should be valid sizes
        }
    
    def detect_commission_errors(self, max_reasonable_area_m2: float = 100.0) -> Dict:
        """
        Detect commission errors (false positives - detecting non-trees)
        
        Commission errors typically have:
        - Extremely large areas (buildings, water bodies)
        - Very irregular shapes
        - Unrealistic size for region
        
        Args:
            max_reasonable_area_m2: Maximum reasonable crown area
            
        Returns:
            Dictionary with suspected commission errors
        """
        suspected_errors = []
        
        for idx, feature in enumerate(self.features):
            props = feature.get('properties', {})
            area = props.get('area', 0.0)
            
            # Flag: Too large (likely building, field, water)
            if area > max_reasonable_area_m2:
                suspected_errors.append({
                    'feature_id': idx,
                    'area_m2': area,
                    'reason': f'Unreasonably large ({area:.1f} m²), likely not a tree',
                    'severity': 'high'
                })
        
        total_detections = len(self.features)
        error_rate = (len(suspected_errors) / total_detections * 100) if total_detections > 0 else 0
        
        return {
            'total_detections': total_detections,
            'suspected_commission_errors': len(suspected_errors),
            'commission_rate_percent': error_rate,
            'errors': suspected_errors[:10]  # Show first 10
        }
    
    def detect_omission_hints(self, expected_density_per_hectare: float = None) -> Dict:
        """
        Provide hints about potential omission errors (missed trees)
        
        Note: True omission detection requires ground truth data.
        This provides indirect indicators based on:
        - Unusually low tree density for the area
        - Gaps in coverage (requires spatial analysis)
        
        Args:
            expected_density_per_hectare: Expected trees per hectare (optional)
            
        Returns:
            Dictionary with omission hints
        """
        total_trees = len(self.features)
        
        # Get bounding box area (if we have coordinate system)
        try:
            stats = self.calculate_area_statistics()
            total_canopy_area = stats['total_area_m2']
            
            # Estimate surveyed area (very rough - total_canopy / typical_canopy_coverage)
            # Assuming ~40% canopy coverage for mangroves
            estimated_area_m2 = total_canopy_area / 0.4 if total_canopy_area > 0 else 0
            estimated_area_ha = estimated_area_m2 / 10000
            
            density_per_ha = total_trees / estimated_area_ha if estimated_area_ha > 0 else 0
            
            hints = []
            
            # Typical mangrove density: 500-2000 trees/ha depending on maturity
            if expected_density_per_hectare and density_per_ha < expected_density_per_hectare * 0.5:
                hints.append(
                    f"Detected density ({density_per_ha:.0f} trees/ha) is much lower than "
                    f"expected ({expected_density_per_hectare:.0f} trees/ha). "
                    "Possible omission errors (missed trees)."
                )
            elif density_per_ha < 500 and total_trees > 10:
                hints.append(
                    f"Low tree density detected ({density_per_ha:.0f} trees/ha). "
                    "Consider: (1) Lower confidence threshold, (2) Check if area includes sparse vegetation"
                )
            
            return {
                'total_trees_detected': total_trees,
                'estimated_area_ha': estimated_area_ha,
                'density_per_ha': density_per_ha,
                'omission_hints': hints,
                'note': 'True omission detection requires ground truth validation'
            }
            
        except Exception as e:
            return {
                'total_trees_detected': total_trees, 
                'error': str(e),
                'note': 'Could not estimate omission - requires ground truth data'
            }
    
    def generate_report(self, 
                       min_expected_area: float = 0.5,
                       max_expected_area: float = 50.0) -> None:
        """
        Generate comprehensive validation report
        
        Args:
            min_expected_area: Minimum expected crown area in m²
            max_expected_area: Maximum expected crown area in m²
        """
        console.print(Panel.fit(
            "[bold cyan]🔍 Detection Quality Validation Report[/bold cyan]",
            border_style="cyan"
        ))
        
        # Area Statistics
        console.print("\n[bold]📊 Crown Area Statistics[/bold]")
        stats = self.calculate_area_statistics()
        
        area_table = Table(show_header=False)
        area_table.add_column("Metric", style="cyan")
        area_table.add_column("Value", style="yellow")
        
        area_table.add_row("Total Detections", str(stats['count']))
        area_table.add_row("Total Canopy Area", f"{stats['total_area_m2']:.2f} m²")
        area_table.add_row("Mean Crown Area", f"{stats['mean_area_m2']:.2f} m²")
        area_table.add_row("Median Crown Area", f"{stats['median_area_m2']:.2f} m²")
        area_table.add_row("Std Deviation", f"{stats['std_area_m2']:.2f} m²")
        area_table.add_row("Min Crown Area", f"{stats['min_area_m2']:.2f} m²")
        area_table.add_row("Max Crown Area", f"{stats['max_area_m2']:.2f} m²")
        area_table.add_row("25th Percentile", f"{stats.get('percentile_25_m2', 0):.2f} m²")
        area_table.add_row("75th Percentile", f"{stats.get('percentile_75_m2', 0):.2f} m²")
        
        console.print(area_table)
        
        # Size Validation
        console.print("\n[bold]✓ Crown Size Validation[/bold]")
        size_val = self.validate_crown_sizes(min_expected_area, max_expected_area)
        
        size_table = Table(show_header=False)
        size_table.add_column("Metric", style="cyan")
        size_table.add_column("Value", style="yellow")
        
        size_table.add_row("Expected Size Range", size_val['expected_range_m2'] + " m²")
        size_table.add_row("Valid Sizes", f"{size_val['valid_size']} ({size_val['valid_percent']:.1f}%)")
        size_table.add_row("Too Small (< {0:.1f} m²)".format(min_expected_area), str(size_val['too_small']))
        size_table.add_row("Too Large (> {0:.1f} m²)".format(max_expected_area), str(size_val['too_large']))
        size_table.add_row("Validation Passed", "✅ Yes" if size_val['passed'] else "❌ No")
        
        console.print(size_table)
        
        # Commission Errors
        console.print("\n[bold]❌ Commission Error Detection (False Positives)[/bold]")
        commission = self.detect_commission_errors(max_expected_area)
        
        comm_table = Table(show_header=False)
        comm_table.add_column("Metric", style="cyan")
        comm_table.add_column("Value", style="yellow")
        
        comm_table.add_row("Total Detections", str(commission['total_detections']))
        comm_table.add_row("Suspected Errors", str(commission['suspected_commission_errors']))
        comm_table.add_row("Commission Rate", f"{commission['commission_rate_percent']:.2f}%")
        
        console.print(comm_table)
        
        if commission['errors']:
            console.print("\n[dim]Top suspected commission errors:[/dim]")
            for error in commission['errors'][:5]:
                console.print(f"  • Feature #{error['feature_id']}: {error['reason']}")
        
        # Omission Hints
        console.print("\n[bold]⚠️  Omission Error Hints (Missed Trees)[/bold]")
        omission = self.detect_omission_hints()
        
        omiss_table = Table(show_header=False)
        omiss_table.add_column("Metric", style="cyan")
        omiss_table.add_column("Value", style="yellow")
        
        omiss_table.add_row("Trees Detected", str(omission['total_trees_detected']))
        if 'estimated_area_ha' in omission:
            omiss_table.add_row("Estimated Area", f"{omission.get('estimated_area_ha', 0):.2f} ha")
            omiss_table.add_row("Density", f"{omission.get('density_per_ha', 0):.0f} trees/ha")
        
        console.print(omiss_table)
        
        if omission.get('omission_hints'):
            console.print("\n[yellow]Omission hints:[/yellow]")
            for hint in omission['omission_hints']:
                console.print(f"  ⚠️  {hint}")
        
        console.print(f"\n[dim]{omission.get('note', '')}[/dim]")
        
        # Summary
        console.print("\n[bold green]📝 Summary & Recommendations[/bold green]\n")
        
        if size_val['passed'] and commission['commission_rate_percent'] < 10:
            console.print("[green]✅ Detection quality appears GOOD[/green]")
            console.print("   - Crown sizes are realistic")
            console.print("   - Low commission error rate")
        else:
            console.print("[yellow]⚠️  Detection quality may need improvement[/yellow]")
            if not size_val['passed']:
                console.print("   - Many detections have unrealistic sizes")
                console.print("   → Adjust min_area threshold or retrain model")
            if commission['commission_rate_percent'] > 10:
                console.print(f"   - High commission rate ({commission['commission_rate_percent']:.1f}%)")
                console.print("   → Increase confidence threshold or add post-processing filters")


@app.command()
def validate(
    geojson_path: str = typer.Option(
        "output_geojson/detected_crowns.geojson",
        "--input", "-i",
        help="Path to GeoJSON file with detection results"
    ),
    min_area: float = typer.Option(
        0.5,
        "--min-area",
        help="Minimum expected crown area in m²"
    ),
    max_area: float = typer.Option(
        50.0,
        "--max-area",
        help="Maximum expected crown area in m²"
    )
):
    """
    Validate tree crown detection results for quality and errors
    """
    geojson_file = Path(geojson_path)
    
    if not geojson_file.exists():
        console.print(f"[bold red]❌ Error: File not found: {geojson_path}[/bold red]")
        raise typer.Exit(1)
    
    validator = DetectionValidator(geojson_path)
    validator.generate_report(min_area, max_area)


if __name__ == "__main__":
    app()
