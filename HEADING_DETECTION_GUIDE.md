# Camera Heading Detection - User Guide

## Problem
Drone images don't include camera heading (compass direction) in GPS EXIF data. Without knowing which way the camera was facing, GPS markers appear rotated on the map.

## Solutions (In Order of Preference)

### ✅ Solution 1: Flight Log / SRT File (RECOMMENDED)

**DJI Drones** (Mavic, Phantom, Mini, Air series) automatically record heading data in `.SRT` subtitle files.

**Where to find SRT files:**
1. Connect drone SD card or DJI Smart Controller to computer
2. Navigate to drone's storage:
   - **DJI Mini/Air/Mavic**: `DCIM/100MEDIA/` folder
   - Look for files like: `DJI_0001.SRT`, `DJI_0002.SRT`, etc.
   - Each SRT file corresponds to a video or image with the same number
3. Upload the `.SRT` file alongside your image in MangroVision

**Supported formats:**
- `.SRT` - DJI subtitle files (most common)
- `.TXT` - Flight log exports from DJI apps
- `.LOG` - Some drone telemetry logs
- `.CSV` - Exported flight data

**What the file contains:**
```
GPS coordinates, altitude, gimbal angles, compass heading (yaw)
```

---

### ✅ Solution 2: Multiple Images (GPS Trajectory)

If you have **2 or more consecutive images** from the same flight:

1. The system calculates the direction of travel from GPS coordinates
2. Assumes camera faces forward in the direction of flight
3. Automatically determines heading

**How to use:**
- Upload multiple images in sequence
- System will analyze GPS trajectory
- Heading will be calculated and applied to all images from that flight

---

### ⚠️ Solution 3: Manual Override (Last Resort)

If flight logs are unavailable:

1. Compare Visual Results with orthophoto map
2. Identify same landmarks (tower, building, path)
3. Manually adjust heading slider until GPS markers align

**Compass directions:**
- `0°` = North (top of image faces north)
- `90°` = East (top of image faces east)
- `180°` = South (top of image faces south)
- `270°` = West (top of image faces west)

---

## How to Get Flight Logs

### DJI Drones
**Option A: From SD Card**
- SRT files automatically created alongside videos/photos
- Located in `DCIM/100MEDIA/` folder
- Same filename as image/video (e.g., `DJI_0049.SRT` for `DJI_0049.JPG`)

**Option B: From DJI Fly/Pilot App**
1. Open DJI Fly or DJI Pilot app
2. Go to "Flight Records" or "Flight Logs"
3. Select your flight
4. Export as TXT or CSV file
5. Upload to MangroVision

### Other Drone Brands
- **Autel**: Check flight logs in Autel Explorer app
- **Parrot**: Pix4Dcapture records heading in metadata
- **Yuneec**: DataPilot app exports telemetry

---

## Why This Matters for Your Study

**Academic Rigor:**
- Automatic heading detection eliminates manual guessing
- Reduces human error in GPS coordinate transformation
- Provides reproducible, verifiable results

**Field Deployment:**
- Accurate GPS coordinates essential for field workers
- Wrong heading = markers 50-100 meters off target
- Your thesis depends on precise planting zone locations

**Professional Quality:**
- Shows you understand coordinate transformations
- Demonstrates proper handling of geospatial data
- Elevates project from "student demo" to "professional system"

---

## Quick Start

1. **Before fieldwork**: Ensure drone records SRT files (check settings)
2. **After flight**: Copy both images AND SRT files from SD card
3. **In MangroVision**: 
   - Upload image
   - Upload corresponding SRT file
   - System automatically extracts heading
   - GPS markers accurately placed on map ✓

---

## Troubleshooting

**Q: My drone doesn't create SRT files**
A: Check drone video settings - SRT is usually created for videos, not single photos. Alternative: Take multiple photos to enable GPS trajectory analysis.

**Q: SRT file shows "no heading found"**
A: Some compressed SRT formats may not include all telemetry. Try exporting flight log from DJI app instead.

**Q: I only have one image and no flight log**
A: Use manual override. Identify landmarks visible in both image and map, adjust slider until they align.

**Q: Can I use this for non-DJI drones?**
A: Yes! Upload any flight log format that contains heading/yaw/compass data. The parser will attempt to extract it.

---

## Technical Details

**What we're extracting:**
- **Yaw (gimbal heading)**: Direction camera is pointing
- **Compass heading**: Direction drone is facing
- **GPS trajectory**: Flight path direction

**Why GPS EXIF isn't enough:**
- Standard GPS tags: `latitude`, `longitude`, `altitude`
- Missing: `GPSImgDirection` (rarely populated)
- Solution: External telemetry from flight logs

**Coordinate transformation:**
```
1. Detect hexagon at pixel (x, y) in image
2. Convert to offset from center in meters
3. Rotate by camera heading angle
4. Convert to GPS (lat, lon)
5. Place marker on map
```

Without correct heading → markers rotated → wrong locations in field.

---

## Example Workflow

**Your defense/demonstration:**

1. "We uploaded a drone image of the mangrove area"
2. "The system extracted GPS coordinates from EXIF"
3. "We uploaded the corresponding SRT file from the drone"
4. "The system automatically detected camera heading: 87° (East)"
5. "Canopy detection identified 47 safe planting zones"
6. "GPS coordinates calculated and displayed on orthophoto map"
7. "Field workers use these coordinates for precise planting" ✓

**Thesis committee impressed** ✓

---

*This approach demonstrates advanced understanding of:*
- Geospatial coordinate systems
- Drone telemetry integration
- Automated data processing
- Real-world deployment considerations
