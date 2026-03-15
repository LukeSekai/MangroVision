import subprocess
import sys
import time
from pathlib import Path
import webbrowser

def print_header():
    print("="*60)
    print("🌿 MangroVision - Map-Based Detection System")
    print("="*60)
    print()

def check_tiles_exist():
    """Check if tiles directory exists"""
    tiles_path = Path("MAP")
    if not tiles_path.exists():
        print("⚠️  Warning: 'MAP' folder not found!")
        print("Please create a 'MAP' folder and add your XYZ tiles from QGIS")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    else:
        print("✅ Found MAP tiles directory")
    return True

def start_tile_server():
    """Start simple HTTP server for tiles"""
    print("\n📡 Starting tile server on port 8080...")
    
    tiles_path = Path("MAP")
    if tiles_path.exists():
        try:
            # Start HTTP server for tiles
            process = subprocess.Popen(
                [sys.executable, "-m", "http.server", "8080"],
                cwd=str(tiles_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            time.sleep(1)
            print("✅ Tile server running at http://localhost:8080")
            return process
        except Exception as e:
            print(f"❌ Failed to start tile server: {e}")
            return None
    else:
        print("⚠️  MAP folder not found, skipping tile server")
        return None

def start_backend():
    """Start FastAPI backend"""
    print("\n🔧 Starting backend API on port 8000...")
    
    try:
        process = subprocess.Popen(
            [sys.executable, "map_backend.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(2)
        print("✅ Backend API running at http://localhost:8000")
        return process
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def open_frontend():
    """Open frontend in browser"""
    print("\n🌐 Opening frontend in browser...")
    
    frontend_path = Path("map_frontend.html").absolute()
    try:
        webbrowser.open(f"file://{frontend_path}")
        print("✅ Frontend opened in browser")
        return True
    except Exception as e:
        print(f"⚠️  Could not auto-open browser: {e}")
        print(f"Please manually open: {frontend_path}")
        return False

def main():
    print_header()
    
    # Check prerequisites
    if not check_tiles_exist():
        return
    
    # Start servers
    tile_server = start_tile_server()
    backend = start_backend()
    
    if backend is None:
        print("\n❌ Failed to start backend. Please check:")
        print("  1. FastAPI is installed: pip install fastapi uvicorn")
        print("  2. All dependencies are installed: pip install -r requirements.txt")
        print("  3. map_backend.py exists in current directory")
        return
    
    # Open frontend
    time.sleep(1)
    open_frontend()
    
    print("\n" + "="*60)
    print("🎉 System Started Successfully!")
    print("="*60)
    print("\n📋 Services Running:")
    if tile_server:
        print("  • Tile Server:  http://localhost:8080")
    print("  • Backend API:  http://localhost:8000")
    print("  • Frontend:     map_frontend.html (opened in browser)")
    
    print("\n💡 Usage:")
    print("  1. Upload an image from your testing site")
    print("  2. Click 'Analyze & Map Plantable Areas'")
    print("  3. View detected zones on the map")
    
    print("\n⚠️  Important:")
    print("  • Configure orthophoto bounds in map_backend.py")
    print("  • See MAP_GUIDE.md for detailed setup")
    
    print("\n🛑 Press Ctrl+C to stop all servers")
    print("="*60)
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        
        if tile_server:
            tile_server.terminate()
            print("✅ Tile server stopped")
        
        if backend:
            backend.terminate()
            print("✅ Backend stopped")
        
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
