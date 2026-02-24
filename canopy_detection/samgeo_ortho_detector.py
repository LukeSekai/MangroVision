"""
MangroVision - Samgeo Detectree2 Orthophoto Tree Crown Detector
Process high-resolution GeoTIFF orthophotos to detect individual tree crowns
Uses the segment-geospatial (samgeo) wrapper for detectree2
"""

import sys
from pathlib import Path
import json
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

console = Console()
app = typer.Typer()


def display_results(results: dict):
    """Display detection results in a beautiful table"""
    
    # Create results table
    table = Table(title="🌳 Tree Crown Detection Results", show_header=True, header_style="bold green")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Trees Detected", str(results['tree_count']))
    table.add_row("Total Canopy Area", f"{results['total_area_m2']:.2f} m²")
    table.add_row("Average Crown Area", f"{results['avg_crown_area_m2']:.2f} m²")
    table.add_row("Output GeoJSON", results['geojson_path'])
    table.add_row("Model Used", results['metadata']['model'])
    table.add_row("Device", results['metadata']['device'])
    table.add_row("Tile Size", results['metadata']['tile_size'])
    
    console.print(table)


@app.command()
def detect(
    input_tif: str = typer.Option(
        "MAP/odm_orthophoto/odm_orthophoto.tif",
        "--input", "-i",
        help="Path to input GeoTIFF orthophoto"
    ),
    output_geojson: str = typer.Option(
        "output_geojson/detected_crowns.geojson",
        "--output", "-o",
        help="Path to save output GeoJSON"
    ),
    model_path: str = typer.Option(
        None,
        "--model", "-m",
        help="Path to detectree2 model weights (.pth file)"
    ),
    tile_width: int = typer.Option(
        100,
        "--tile-width", "-tw",
        help="Width of inference tiles (pixels)"
    ),
    tile_height: int = typer.Option(
        100,
        "--tile-height", "-th",
        help="Height of inference tiles (pixels)"
    ),
    device: str = typer.Option(
        "cpu",
        "--device", "-d",
        help="Device to use: 'cpu' or 'cuda'"
    ),
    min_area: float = typer.Option(
        0.5,
        "--min-area", "-ma",
        help="Minimum crown area in m² to keep"
    ),
    confidence: float = typer.Option(
        0.5,
        "--confidence", "-c",
        help="Minimum detection confidence (0-1)"
    )
):
    """
    Detect tree crowns from high-resolution GeoTIFF orthophoto.
    
    Uses detectree2 with samgeo wrapper for accurate tropical tree detection.
    Results are saved as GeoJSON with crown polygons and area calculations.
    """
    
    console.print(Panel.fit(
        "[bold green]🌳 MangroVision Tree Crown Detector[/bold green]\n"
        "[cyan]Using Samgeo + Detectree2 for accurate mangrove detection[/cyan]",
        border_style="green"
    ))
    
    # Check if input file exists
    input_path = Path(input_tif)
    if not input_path.exists():
        console.print(f"[bold red]❌ Error: Input file not found: {input_tif}[/bold red]")
        raise typer.Exit(1)
    
    # Check if samgeo is installed
    try:
        from samgeo.detectree2 import TreeCrownDelineator
        console.print("[green]✓[/green] Samgeo detectree2 wrapper found")
    except ImportError:
        console.print("[bold red]❌ Error: segment-geospatial not installed[/bold red]")
        console.print("\nInstall with:")
        console.print("  [cyan]pip install segment-geospatial[/cyan]")
        console.print("\nSee INSTALL_SAMGEO_DETECTREE2.md for complete guide")
        raise typer.Exit(1)
    
    # Determine model path
    if model_path is None:
        model_dir = Path(__file__).parent.parent / 'models'
        tropical_models = [
            model_dir / '230103_randresize_full.pth',
            model_dir / '230717_tropical_base.pth'
        ]
        for model in tropical_models:
            if model.exists():
                model_path = str(model)
                console.print(f"[green]✓[/green] Using model: {model.name}")
                break
        
        if model_path is None:
            console.print("[bold red]❌ Error: No detectree2 model found[/bold red]")
            console.print(f"\nExpected location: {model_dir}")
            console.print("\nDownload from: https://zenodo.org/record/7123579")
            console.print("Save as: models/230103_randresize_full.pth")
            raise typer.Exit(1)
    
    # Create output directory
    output_path = Path(output_geojson)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Display configuration
    config_table = Table(show_header=False, box=None)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="yellow")
    config_table.add_row("Input GeoTIFF", str(input_path))
    config_table.add_row("Output GeoJSON", str(output_path))
    config_table.add_row("Tile Size", f"{tile_width}x{tile_height}")
    config_table.add_row("Device", device)
    config_table.add_row("Min Area", f"{min_area} m²")
    config_table.add_row("Confidence", str(confidence))
    
    console.print("\n[bold]Configuration:[/bold]")
    console.print(config_table)
    console.print()
    
    # Initialize delineator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Initializing TreeCrownDelineator...", total=None)
        
        try:
            delineator = TreeCrownDelineator(
                model_path=model_path,
                device=device
            )
            progress.update(task, description="[green]✓ Model loaded")
        except Exception as e:
            console.print(f"[bold red]❌ Error loading model: {e}[/bold red]")
            raise typer.Exit(1)
    
    # Run detection
    console.print("\n[bold cyan]Running tree crown detection...[/bold cyan]")
    console.print("[dim]This may take several minutes for large orthophotos[/dim]\n")
    
    try:
        delineator.predict(
            image_path=str(input_path),
            output_path=str(output_path),
            tile_width=tile_width,
            tile_height=tile_height,
            output_format='geojson',
            min_area=min_area,
            iou_threshold=0.5,  # Non-maximum suppression
            threshold=confidence  # Detection confidence
        )
    except Exception as e:
        console.print(f"\n[bold red]❌ Detection failed: {e}[/bold red]")
        raise typer.Exit(1)
    
    # Calculate statistics
    console.print("\n[cyan]Calculating statistics...[/cyan]")
    
    with open(output_path, 'r') as f:
        geojson = json.load(f)
    
    features = geojson.get('features', [])
    tree_count = len(features)
    
    if tree_count == 0:
        console.print("[yellow]⚠️  No trees detected. Try:[/yellow]")
        console.print("  - Lower confidence threshold: --confidence 0.3")
        console.print("  - Lower minimum area: --min-area 0.1")
        console.print("  - Check if orthophoto covers vegetation area")
        raise typer.Exit(0)
    
    # Calculate areas
    areas = []
    for feature in features:
        props = feature.get('properties', {})
        area = props.get('area', 0.0)
        if area > 0:
            areas.append(area)
    
    total_area = sum(areas) if areas else 0.0
    avg_area = total_area / len(areas) if areas else 0.0
    
    results = {
        'tree_count': tree_count,
        'total_area_m2': total_area,
        'avg_crown_area_m2': avg_area,
        'geojson_path': str(output_path),
        'metadata': {
            'model': Path(model_path).name,
            'device': device,
            'tile_size': f"{tile_width}x{tile_height}"
        }
    }
    
    # Display results
    console.print()
    display_results(results)
    
    console.print(f"\n[bold green]✅ Detection complete![/bold green]")
    console.print(f"\n[dim]View results in Streamlit:[/dim]")
    console.print(f"  [cyan]streamlit run app.py[/cyan]")
    

@app.command()
def quick_test():
    """
    Quick test detection on a small tile (for testing installation)
    """
    console.print("[bold cyan]Running quick test detection...[/bold cyan]\n")
    
    # Use a smaller tile for testing
    detect(
        input_tif="MAP/odm_orthophoto/odm_orthophoto.tif",
        output_geojson="output_geojson/test_detection.geojson",
        tile_width=50,
        tile_height=50,
        device="cpu",
        min_area=0.1,
        confidence=0.5
    )


if __name__ == "__main__":
    # If no command specified, run detect by default
    import sys
    if len(sys.argv) == 1:
        sys.argv.append("detect")
    app()
