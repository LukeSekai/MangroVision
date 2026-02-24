"""
MangroVision - Test Samgeo + Detectree2 Installation
Verify that all components are correctly installed
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def test_pytorch():
    """Test PyTorch installation"""
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        device = "CUDA" if cuda_available else "CPU"
        console.print(f"[green]✓[/green] PyTorch {version} installed ({device})")
        return True
    except ImportError:
        console.print("[red]✗[/red] PyTorch not installed")
        console.print("  Install with: pip install torch torchvision")
        return False


def test_detectron2():
    """Test Detectron2 installation"""
    try:
        import detectron2
        version = detectron2.__version__
        console.print(f"[green]✓[/green] Detectron2 {version} installed")
        return True
    except ImportError:
        console.print("[red]✗[/red] Detectron2 not installed")
        console.print("  Install with: pip install git+https://github.com/facebookresearch/detectron2.git")
        return False


def test_samgeo():
    """Test Samgeo installation"""
    try:
        import samgeo
        console.print(f"[green]✓[/green] Samgeo installed")
        return True
    except ImportError:
        console.print("[red]✗[/red] Segment-geospatial (samgeo) not installed")
        console.print("  Install with: pip install segment-geospatial")
        return False


def test_samgeo_detectree2():
    """Test Samgeo detectree2 module"""
    try:
        from samgeo.detectree2 import TreeCrownDelineator
        console.print(f"[green]✓[/green] TreeCrownDelineator available")
        return True
    except ImportError as e:
        console.print(f"[red]✗[/red] TreeCrownDelineator not available: {e}")
        return False


def test_gdal():
    """Test GDAL installation (optional but recommended)"""
    try:
        from osgeo import gdal
        version = gdal.__version__
        console.print(f"[green]✓[/green] GDAL {version} installed")
        return True
    except ImportError:
        console.print("[yellow]⚠[/yellow] GDAL not installed (optional)")
        console.print("  For better GeoTIFF support, install with: conda install -c conda-forge gdal")
        return None  # Not critical


def test_geopandas():
    """Test GeoPandas installation"""
    try:
        import geopandas
        console.print(f"[green]✓[/green] GeoPandas installed")
        return True
    except ImportError:
        console.print("[red]✗[/red] GeoPandas not installed")
        console.print("  Install with: pip install geopandas")
        return False


def test_model_files():
    """Test if model files exist"""
    model_dir = Path(__file__).parent.parent / 'models'
    
    models = {
        '230103_randresize_full.pth': 'New tropical model (recommended)',
        '230717_tropical_base.pth': 'Current tropical model'
    }
    
    found_any = False
    for model_name, description in models.items():
        model_path = model_dir / model_name
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            console.print(f"[green]✓[/green] {model_name} found ({size_mb:.1f} MB)")
            found_any = True
        else:
            console.print(f"[yellow]⚠[/yellow] {model_name} not found ({description})")
    
    return found_any


def test_orthophoto_exists():
    """Test if orthophoto exists"""
    ortho_path = Path(__file__).parent.parent / 'MAP' / 'odm_orthophoto' / 'odm_orthophoto.tif'
    
    if ortho_path.exists():
        size_mb = ortho_path.stat().st_size / (1024 * 1024)
        console.print(f"[green]✓[/green] Orthophoto found ({size_mb:.1f} MB)")
        return True
    else:
        console.print(f"[yellow]⚠[/yellow] Orthophoto not found at {ortho_path}")
        console.print("  This is optional - only needed for orthophoto-based detection")
        return None


def main():
    console.print(Panel.fit(
        "[bold cyan]🌳 MangroVision - Samgeo + Detectree2 Installation Test[/bold cyan]",
        border_style="cyan"
    ))
    
    console.print("\n[bold]Testing Core Dependencies:[/bold]\n")
    
    results = {}
    results['pytorch'] = test_pytorch()
    results['detectron2'] = test_detectron2()
    results['samgeo'] = test_samgeo()
    results['samgeo_detectree2'] = test_samgeo_detectree2()
    results['geopandas'] = test_geopandas()
    
    console.print("\n[bold]Testing Optional Dependencies:[/bold]\n")
    results['gdal'] = test_gdal()
    
    console.print("\n[bold]Testing Data Files:[/bold]\n")
    results['models'] = test_model_files()
    results['orthophoto'] = test_orthophoto_exists()
    
    # Summary
    console.print("\n" + "="*70 + "\n")
    
    critical_components = ['pytorch', 'detectron2', 'samgeo', 'samgeo_detectree2', 'geopandas']
    critical_passed = all(results.get(comp, False) for comp in critical_components)
    
    if critical_passed:
        console.print(Panel.fit(
            "[bold green]✅ ALL CRITICAL COMPONENTS INSTALLED SUCCESSFULLY![/bold green]\n\n"
            "[green]You can now run tree crown detection with:[/green]\n"
            "  [cyan]python canopy_detection/samgeo_ortho_detector.py[/cyan]",
            border_style="green"
        ))
        
        if not results.get('models', False):
            console.print("\n[yellow]⚠️  Note: No model files found[/yellow]")
            console.print("Download from: https://zenodo.org/record/7123579")
            console.print("Save to: models/230103_randresize_full.pth")
    else:
        console.print(Panel.fit(
            "[bold red]❌ INSTALLATION INCOMPLETE[/bold red]\n\n"
            "[yellow]Install missing components shown above with [red]✗[/red] marks[/yellow]\n\n"
            "See [cyan]INSTALL_SAMGEO_DETECTREE2.md[/cyan] for complete installation guide",
            border_style="red"
        ))
    
    console.print("\n[dim]For detailed installation instructions, see: INSTALL_SAMGEO_DETECTREE2.md[/dim]\n")


if __name__ == "__main__":
    main()
