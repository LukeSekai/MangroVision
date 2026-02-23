# Map-Based Plantable Area Detection - Complete Setup Guide

## 🎯 What This System Does

1. **Display your orthophoto** as an interactive web map using XYZ tiles
2. **Upload a field photo** from your testing site
3. **Detect plantable areas** using your existing MangroVision detector
4. **Convert pixel coordinates to GPS coordinates** (lat/lon)
5. **Display markers on the map** showing exactly where to plant

---

## 📋 Prerequisites Checklist

- [x] Orthophoto generated from WebODM
- [x] XYZ tiles created from QGIS
- [x] MangroVision detector working (you already have this!)
- [ ] Backend server running (FastAPI)
- [ ] Tile server running
- [ ] Frontend HTML file open in browser

---

## 🚀 Step-by-Step Setup

### **Step 1: Install Additional Dependencies**

```powershell
# Activate your venv
.\venv\Scripts\Activate.ps1

# Install FastAPI and server
pip install fastapi uvicorn python-multipart

# For orthophoto metadata extraction (optional)
pip install gdal rasterio
```

---

### **Step 2: Configure Your Orthophoto Metadata**

You need to tell the system where your orthophoto is located geographically.

#### Option A: Extract from GeoTIFF (Recommended)

If you have your orthophoto as a GeoTIFF from WebODM:

```powershell
python extract_orthophoto_info.py path\to\your\orthophoto.tif
```

This will print all the values you need. Copy them to `map_backend.py`.

#### Option B: Get from QGIS Manually

1. Open your orthophoto in QGIS
2. Right-click the layer → **Properties** → **Information**
3. Find the **Extent** values:
   - North (top): Maximum Y
   - South (bottom): Minimum Y  
   - East (right): Maximum X
   - West (left): Minimum X

4. Update `map_backend.py` line 47-60:

```python
class OrthophotoMetadata:
    def __init__(self):
        self.bounds = {
            'north': YOUR_NORTH_VALUE,    # Top latitude
            'south': YOUR_SOUTH_VALUE,    # Bottom latitude  
            'east': YOUR_EAST_VALUE,      # Right longitude
            'west': YOUR_WEST_VALUE       # Left longitude
        }
        
        self.width_px = YOUR_ORTHO_WIDTH   # From QGIS
        self.height_px = YOUR_ORTHO_HEIGHT # From QGIS
        self.gsd_cm = YOUR_GSD_VALUE       # Ground Sample Distance
```

**Example values (Leganes area - UPDATE WITH YOUR VALUES):**
```python
'north': 10.7234,
'south': 10.7198,
'east': 122.5689,
'west': 122.5645
```

---

### **Step 3: Serve Your XYZ Tiles**

Your tiles need to be accessible via HTTP. You have several options:

#### Option A: Simple Python HTTP Server

```powershell
# Navigate to where your tiles are
cd path\to\your\tiles

# Start simple server on port 8080
python -m http.server 8080
```

Your tiles should be accessible at: `http://localhost:8080/tiles/{z}/{x}/{y}.png`

#### Option B: Use QGIS QTiles Plugin Server

If you used QTiles plugin, it may have created a viewer with a server.

#### Option C: Node.js http-server (faster)

```powershell
npm install -g http-server
cd path\to\tiles
http-server -p 8080 --cors
```

**Verify tiles are working:**
Open browser: `http://localhost:8080/tiles/18/12345/67890.png`  
(Use actual tile coordinates from your tiles folder)

---

### **Step 4: Start the Backend API**

```powershell
# Make sure venv is activated
.\venv\Scripts\Activate.ps1

# Start the FastAPI backend
python map_backend.py
```

You should see:
```
🌿 Starting MangroVision Map API...
📍 Configure your orthophoto bounds in OrthophotoMetadata class!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Test it's working:**
Open browser: `http://localhost:8000/api/map/metadata`  
You should see JSON with your map bounds.

---

### **Step 5: Configure Frontend Tile URL**

Open `map_frontend.html` and update line 356:

```javascript
// Change this URL to match your tile server
tileLayer = L.tileLayer('http://localhost:8080/tiles/{z}/{x}/{y}.png', {
    maxZoom: 20,
    minZoom: 15,
    attribution: 'MangroVision Orthophoto',
    tms: false  // Set to true if using TMS tile scheme
});
```

**Check your tile naming convention:**
- Standard: `tiles/z/x/y.png` → `tms: false`
- TMS: `tiles/z/x/y.png` but Y is inverted → `tms: true`

---

### **Step 6: Open the Frontend**

Simply open `map_frontend.html` in your browser:

```powershell
# Open with default browser
start map_frontend.html

# Or open manually in Chrome/Edge
```

---

## 🎮 **How to Use the System**

### Workflow:

1. **Open** `map_frontend.html` in browser
2. **See your orthophoto** displayed as the map
3. **Upload a photo** from your testing site (drag & drop or click)
4. **Click "Analyze & Map Plantable Areas"**
5. **Wait** for processing (2-5 seconds)
6. **View results:**
   - Green markers 🌱 = Plantable zones
   - Red markers ⚠️ = Danger zones (canopies)
7. **Click markers** to see exact coordinates
8. **Export** coordinates for field work

---

## 🔧 **Understanding the Coordinate Transformation**

### Key Concept: Pixel-to-Geographic Mapping

When you upload an image, the system needs to know WHERE in the real world that image was taken.

**The Math:**

```python
# Your uploaded image has pixels (0,0) to (width, height)
# Your orthophoto covers geographic area (west, south) to (east, north)

# To convert pixel (x, y) to (lat, lon):

x_fraction = pixel_x / image_width
y_fraction = pixel_y / image_height

longitude = west + x_fraction * (east - west)
latitude = north - y_fraction * (north - south)  # Note: Y is inverted!
```

**Important:** This assumes your uploaded image was taken from roughly the same area shown in the orthophoto. For images from different perspectives, you'd need more advanced georeferencing.

---

## 📊 **System Architecture Explained**

```
USER BROWSER (map_frontend.html)
    ↓
    1. User uploads image
    ↓
BACKEND API (map_backend.py:8000)
    ↓
    2. HexagonDetector analyzes image
    ↓
    3. Finds plantable zones (pixel coordinates)
    ↓
    4. GeoTransformer converts pixels → lat/lon
    ↓
    5. Returns JSON with coordinates
    ↓
FRONTEND receives coordinates
    ↓
    6. Leaflet.js adds markers to map
    ↓
USER sees plantable zones on orthophoto map
```

---

## 📁 **File Structure**

```
MangroVision/
├── map_backend.py              # ⭐ FastAPI backend server
├── map_frontend.html           # ⭐ Web interface
├── extract_orthophoto_info.py  # Tool to extract metadata
│
├── canopy_detection/           # Your existing detector
│   ├── canopy_detector_hexagon.py
│   └── gsd_calculator.py
│
└── tiles/                      # Your XYZ tiles (create folder)
    └── 18/
        └── 123456/
            └── 789012.png
```

---

## 🧪 **Testing**

### Test 1: Backend Health Check
```
http://localhost:8000/
```
Should return: `{"status": "online"}`

### Test 2: Map Metadata
```
http://localhost:8000/api/map/metadata
```
Should return your orthophoto bounds

### Test 3: Simple Detection
Use the `/api/detect-simple` endpoint first to verify coordinate transformation without complex detection.

### Test 4: Full Detection
Upload an image through the frontend and verify markers appear.

---

## 🐛 **Troubleshooting**

### Problem: Map is blank
- ✅ Check tile server is running (`http://localhost:8080`)
- ✅ Verify tile URL in `map_frontend.html` line 356
- ✅ Check browser console (F12) for errors
- ✅ Try with `tms: true` if tiles don't show

### Problem: Markers in wrong location
- ✅ Verify orthophoto bounds in `OrthophotoMetadata`
- ✅ Check that coordinates are in correct order (lat, lon vs lon, lat)
- ✅ Ensure CRS matches (degrees vs meters)

### Problem: CORS errors
- ✅ Backend CORS is enabled (line 24 in map_backend.py)
- ✅ Tile server allows CORS
- ✅ Use `http://`, not `file://` for frontend

### Problem: No plantable areas detected
- ✅ Check uploaded image is from the testing site
- ✅ Verify detector parameters (altitude, drone model)
- ✅ Test with `detect-simple` endpoint first

---

## 🎓 **Advanced: Coordinate Reference Systems (CRS)**

If your orthophoto uses UTM (meters) instead of WGS84 (lat/lon):

1. Install: `pip install pyproj`

2. Update `OrthophotoMetadata`:
```python
self.crs_from = "EPSG:32651"  # UTM Zone 51N (Philippines)
self.crs_to = "EPSG:4326"     # WGS84 (for Leaflet)
```

3. Use `pyproj.Transformer` in `GeoTransformer` class to convert

---

## 📸 **Field Usage Workflow**

1. **At testing site:** Take photos with your phone
2. **Back at computer:** Upload photos to system
3. **System analyzes:** Detects plantable zones
4. **View on map:** See exactly where to plant
5. **Export coordinates:** Take tablet/GPS to field
6. **Navigate to markers:** Plant mangroves at exact locations!

---

## 🚀 **Next Steps**

1. ✅ Configure orthophoto metadata
2. ✅ Start tile server
3. ✅ Start backend (`python map_backend.py`)
4. ✅ Open frontend (`map_frontend.html`)
5. ✅ Test with a sample image
6. 📱 Consider mobile-friendly version
7. 📊 Add database to save planting locations
8. 📷 Add real-time camera capture
9. 🌐 Deploy to cloud for field access

---

## 💡 **Pro Tips**

- **Cache tiles:** Generate all zoom levels in QGIS for smooth panning
- **Mobile access:** Host backend on local network for field access
- **Offline mode:** Pre-cache tiles for areas without internet
- **GPS integration:** Use device GPS to show "You are here"
- **Export formats:** Add KML/GeoJSON export for field devices

---

## 📚 **Key Technologies Used**

- **Backend:** FastAPI (Python) - REST API
- **Frontend:** Leaflet.js - Interactive maps
- **Detection:** Your existing HexagonDetector
- **GIS:** Shapely, PyProj - Geometric operations
- **Tiles:** XYZ format - Standard web map tiles
- **Coordinates:** WGS84 (EPSG:4326) - GPS coordinates

---

**Your system is ready! 🌿 Start mapping plantable areas!**
