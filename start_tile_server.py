"""
Simple HTTP Server with CORS for serving map tiles
"""
import http.server
import socketserver
from pathlib import Path

PORT = 8080

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP Handler with CORS headers"""
    
    def end_headers(self):
        # Add CORS headers to allow requests from Streamlit
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Cache-Control', 'max-age=3600')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

if __name__ == "__main__":
    # Change to MAP directory
    import os
    map_dir = Path(__file__).parent / "MAP"
    if map_dir.exists():
        os.chdir(map_dir)
        print(f"📁 Serving tiles from: {map_dir}")
    else:
        print(f"⚠️  Warning: MAP folder not found at {map_dir}")
        print("   Make sure you run this from the MangroVision directory")
    
    with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
        print(f"🗺️  Tile server running on http://localhost:{PORT}")
        print(f"📡 CORS enabled for all origins")
        print(f"🛑 Press Ctrl+C to stop\n")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n🛑 Tile server stopped")
