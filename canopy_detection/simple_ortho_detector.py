"""
MangroVision - Simple Orthophoto Tree Detector (No Samgeo Required)
Direct detectron2 implementation for tree crown detection
"""

import sys
from pathlib import Path
import json
import cv2
import numpy as np
from typing import List, Dict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table

console = Console()


def detect(
    input_image: str = "MAP/odm_orthophoto/odm_orthophoto.tif",
    output_json: str = "output_geojson/simple_detection.json",
    confidence: float = 0.5
):
    """
    Detect tree crowns using the existing detectree2_detector.py
    This works with your current installation - no samgeo needed!
    """
    
    console.print(Panel.fit(
        "[bold green]🌳 MangroVision Tree Crown Detector[/bold green]\n"
        "[cyan]Using existing detectree2 implementation[/cyan]",
        border_style="green"
    ))
    
    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from canopy_detection.detectree2_detector import Detectree2Detector
    from canopy_detection.ortho_matcher import ortho_pixel_to_gps
    
    # Check if input exists
    input_path = Path(input_image)
    if not input_path.exists():
        console.print(f"[bold red]❌ Error: Input file not found: {input_image}[/bold red]")
        sys.exit(1)
    
    # Load image
    console.print(f"\n[cyan]Loading image: {input_path}[/cyan]")
    
    try:
        # Read image (OpenCV can handle TIF, JPG, PNG)
        image = cv2.imread(str(input_path))
        if image is None:
            console.print(f"[bold red]❌ Error: Cannot read image file[/bold red]")
            sys.exit(1)
        
        h, w = image.shape[:2]
        console.print(f"[green]✓[/green] Image loaded: {w}x{h} pixels")
        
    except Exception as e:
        console.print(f"[bold red]❌ Error loading image: {e}[/bold red]")
        sys.exit(1)
    
    # Initialize detector
    console.print(f"\n[cyan]Initializing detector...[/cyan]")
    
    try:
        detector = Detectree2Detector(
            confidence_threshold=confidence,
            device='cpu'
        )
        detector.setup_model()
        console.print(f"[green]✓[/green] Detector initialized")
    except Exception as e:
        console.print(f"[bold red]❌ Error initializing detector: {e}[/bold red]")
        sys.exit(1)
    
    # Run detection
    console.print(f"\n[bold cyan]Running tree crown detection...[/bold cyan]")
    console.print("[dim]This may take several minutes for large images[/dim]\n")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Detecting tree crowns...", total=None)
            
            polygons, mask, metadata = detector.detect_from_image(image)
            
            progress.update(task, description="[green]✓ Detection complete")
        
        tree_count = len(polygons)
        console.print(f"\n[green]✓[/green] Detected {tree_count} tree crowns")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Detection failed: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)
    
    # Convert polygons to JSON
    console.print(f"\n[cyan]Saving results...[/cyan]")
    
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'type': 'FeatureCollection',
        'features': []
    }
    
    for idx, polygon in enumerate(polygons):
        # Get polygon coordinates (in pixels)
        coords_px = list(polygon.exterior.coords)
        
        # Convert pixel coordinates to GPS (lon, lat)
        coords_gps = []
        for px, py in coords_px:
            lat, lon = ortho_pixel_to_gps(px, py)
            coords_gps.append([lon, lat])  # GeoJSON format is [lon, lat]
        
        # Calculate area (in pixels - for reference)
        area_px = polygon.area
        
        feature = {
            'type': 'Feature',
            'id': idx,
            'geometry': {
                'type': 'Polygon',
                'coordinates': [coords_gps]
            },
            'properties': {
                'tree_id': idx + 1,
                'area_pixels': round(area_px, 2),
                'confidence': confidence
            }
        }
        results['features'].append(feature)
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    console.print(f"[green]✓[/green] Results saved to: {output_path}")
    
    # Display summary
    console.print()
    summary_table = Table(title="🌳 Detection Summary", show_header=True, header_style="bold green")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="magenta")
    
    summary_table.add_row("Input Image", str(input_path))
    summary_table.add_row("Image Size", f"{w}x{h} pixels")
    summary_table.add_row("Trees Detected", str(tree_count))
    summary_table.add_row("Confidence Threshold", f"{confidence:.2f}")
    summary_table.add_row("Output File", str(output_path))
    
    console.print(summary_table)
    
    console.print(f"\n[bold green]✅ Detection complete![/bold green]")
    console.print(f"\n[dim]Results include GPS coordinates for each tree crown.[/dim]")
    console.print(f"[dim]View them in the Streamlit app's 'View Orthophoto Tree Crown Detection' mode.[/dim]")


if __name__ == "__main__":
    detect()
