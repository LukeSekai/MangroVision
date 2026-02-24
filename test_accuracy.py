"""
MangroVision Accuracy Test
Shows detection quality metrics and visual comparison
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add parent to path
sys.path.append(str(Path(__file__).parent))

from canopy_detection.detectree2_detector import Detectree2Detector
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def test_detection_accuracy(image_path: str = None):
    """
    Test detection quality on a sample image
    Shows metrics to verify accuracy
    """
    
    console.print(Panel.fit(
        "[bold cyan]🌳 MangroVision Detection Accuracy Test[/bold cyan]\n"
        "[white]This will show you how well the system detects canopies[/white]",
        border_style="cyan"
    ))
    
    # Use test image
    if image_path is None:
        test_images = [
            "drone_images/dataset_with_gps/frame_0000.jpg",
            "drone_images/dataset_with_gps/frame_0050.jpg",
            "drone_images/dataset_with_gps/frame_0100.jpg"
        ]
        
        for img_path in test_images:
            if Path(img_path).exists():
                image_path = img_path
                break
    
    if not Path(image_path).exists():
        console.print(f"[red]❌ Test image not found: {image_path}[/red]")
        return
    
    console.print(f"\n[cyan]📸 Testing on: {image_path}[/cyan]\n")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        console.print("[red]❌ Failed to load image[/red]")
        return
    
    h, w = image.shape[:2]
    console.print(f"   Image size: {w}x{h} pixels")
    
    # Initialize detector
    console.print(f"\n[cyan]🔧 Initializing detector...[/cyan]")
    # VERY LOW confidence (0.25) for dense canopy - detects shadowed/partial trees
    detector = Detectree2Detector(confidence_threshold=0.25, device='cpu')
    detector.setup_model()
    
    # Run detection
    console.print(f"\n[cyan]🌳 Running detection...[/cyan]")
    polygons, mask, metadata = detector.detect_from_image(image)
    
    # Calculate accuracy metrics
    console.print(f"\n[bold green]📊 Detection Results:[/bold green]\n")
    
    # Create results table
    results_table = Table(title="Detection Metrics", show_header=True, header_style="bold cyan")
    results_table.add_column("Metric", style="yellow", width=30)
    results_table.add_column("Value", style="green", width=40)
    
    # Basic counts
    results_table.add_row("✅ Trees Detected", f"{len(polygons)} individual crowns")
    results_table.add_row("🎯 Detection Method", metadata.get('detection_method', 'N/A'))
    results_table.add_row("🔲 Tiles Processed", str(metadata.get('num_tiles', 'N/A')))
    
    # Coverage metrics
    total_pixels = h * w
    canopy_pixels = np.count_nonzero(mask)
    coverage_percent = 100 * canopy_pixels / total_pixels
    results_table.add_row("🌿 Vegetation Coverage", f"{coverage_percent:.1f}%")
    
    # Detection quality indicators
    veg_coverage = metadata.get('vegetation_coverage_percent', 0)
    raw_detections = metadata.get('total_ai_detections', 0)
    filtered_non_veg = metadata.get('filtered_non_vegetation', 0)
    
    results_table.add_row("🟢 HSV Vegetation Detected", f"{veg_coverage:.1f}%")
    results_table.add_row("🤖 Raw AI Detections", str(raw_detections))
    results_table.add_row("❌ Filtered (non-vegetation)", str(filtered_non_veg))
    
    # Calculate precision estimate
    if raw_detections > 0:
        precision_estimate = 100 * (len(polygons) / raw_detections)
        results_table.add_row("📈 Precision Estimate", f"{precision_estimate:.1f}% (valid detections)")
    
    console.print(results_table)
    
    # Quality assessment
    console.print(f"\n[bold cyan]🎯 Quality Assessment:[/bold cyan]\n")
    
    quality_issues = []
    quality_good = []
    
    # Check if enough trees detected
    if coverage_percent > 50 and len(polygons) < 20:
        quality_issues.append("⚠️  Few trees detected in dense vegetation (may be missing canopies)")
    elif len(polygons) > 0:
        quality_good.append("✅ Detecting individual tree crowns")
    
    # Check filtering rate
    if raw_detections > 0:
        filter_rate = 100 * filtered_non_veg / raw_detections
        if filter_rate > 30:
            quality_issues.append(f"⚠️  High false positive rate ({filter_rate:.0f}% filtered)")
        else:
            quality_good.append(f"✅ Good precision ({100-filter_rate:.0f}% valid detections)")
    
    # Check coverage alignment
    if abs(coverage_percent - veg_coverage) > 10:
        quality_issues.append("⚠️  HSV and AI detection mismatch (calibration issue)")
    else:
        quality_good.append("✅ HSV and AI detection aligned well")
    
    # Display results
    for item in quality_good:
        console.print(f"  {item}")
    
    for item in quality_issues:
        console.print(f"  {item}")
    
    # Overall assessment
    console.print(f"\n[bold cyan]📋 Overall Assessment:[/bold cyan]\n")
    
    if len(quality_issues) == 0:
        console.print("  [bold green]✅ GOOD - Detection quality looks accurate![/bold green]")
        console.print("  [dim]The system is detecting canopies correctly[/dim]")
    elif len(quality_issues) <= 2:
        console.print("  [bold yellow]⚠️  FAIR - Some detection issues found[/bold yellow]")
        console.print("  [dim]Consider adjusting confidence threshold or downloading better model[/dim]")
    else:
        console.print("  [bold red]❌ NEEDS IMPROVEMENT - Multiple issues detected[/bold red]")
        console.print("  [dim]Recommended: Download official detectree2 pre-trained model[/dim]")
    
    # Recommendations
    console.print(f"\n[bold cyan]💡 Recommendations:[/bold cyan]\n")
    
    if len(polygons) < 10 and coverage_percent > 30:
        console.print("  📥 [yellow]Download better model for higher detection rate[/yellow]")
        console.print("      See: DETECTREE2_ACCURACY_GUIDE.md")
    
    if filter_rate > 20:
        console.print("  🎯 [yellow]Increase confidence threshold to reduce false positives[/yellow]")
        console.print("      Try: 0.6 or 0.7 in Streamlit sidebar")
    
    console.print(f"\n  📊 [cyan]Compare with GitHub examples:[/cyan]")
    console.print(f"      https://github.com/PatBall1/detectree2")
    console.print(f"      Expected: 90-95% detection rate in dense canopy")
    
    # Save visualization
    console.print(f"\n[cyan]💾 Saving visualization...[/cyan]")
    
    # Create visualization
    vis = image.copy()
    
    # Draw detections
    for poly in polygons:
        try:
            coords = np.array(poly.exterior.coords, dtype=np.int32)
            cv2.polylines(vis, [coords], True, (0, 255, 0), 2)
        except:
            continue
    
    output_path = Path("detection_accuracy_test.jpg")
    cv2.imwrite(str(output_path), vis)
    
    console.print(f"   ✅ Saved to: {output_path}")
    console.print(f"\n[bold green]✅ Test Complete![/bold green]")


if __name__ == "__main__":
    test_detection_accuracy()
