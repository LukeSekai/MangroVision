# Landmark-Based Alignment - Quick Guide

## 🎯 Purpose
Automatically calculate the correct camera heading by clicking the same landmark in both the Visual Results image and the orthophoto map.

## ✅ Why This Works Better Than Manual Adjustment
- **Accurate**: Mathematical calculation, not guessing
- **Fast**: 2 clicks + calculate = done
- **Reproducible**: Same landmarks always give same result
- **Scientific**: Demonstrates understanding of coordinate transformations

---

## 📖 Step-by-Step Instructions

### Step 1: Run Initial Analysis
1. Upload your drone image
2. Click "Analyze Image" button
3. Wait for Visual Results and map to appear
4. **Notice**: GPS markers may not align with hexagon positions if heading is wrong

---

### Step 2: Identify a Good Landmark

**Best Landmarks:**
- ✅ Tower or antenna (sharp vertical structure)
- ✅ Building corner (distinct right angle)
- ✅ Path intersection (clear junction point)
- ✅ Isolated tree with distinctive shape
- ✅ Pier or dock edge

**Avoid:**
- ❌ Ambiguous vegetation clusters
- ❌ Points too close to image center (less accurate)
- ❌ Moving objects (water, vehicles)

**Pro Tip**: Choose landmark near image edges for best accuracy

---

### Step 3: Get Pixel Coordinates from Visual Results

**Method A: Visual Estimation** (Quick but approximate)
1. Look at the "Detected Zones" image in Visual Results
2. Use the helper expander "📏 How to find pixel coordinates"
3. Estimate based on image dimensions:
   - Example image: 4000 × 3000 pixels
   - If tower is in upper-left: X ≈ 1000, Y ≈ 750
   - If tower is in upper-right: X ≈ 3000, Y ≈ 750

**Method B: Exact Coordinates** (Most accurate)
1. Click "📥 Download Visualization" button
2. Open downloaded image in image viewer:
   - **Paint** (Windows): Shows X,Y in bottom-left corner when hovering
   - **GIMP/Photoshop**: Shows coordinates in info panel
   - **ImageJ** (scientific): Precise coordinate display
3. Hover over your landmark
4. Record X and Y pixel values

---

### Step 4: Get GPS Coordinates from Map

**Finding the landmark on the orthophoto map:**

1. Scroll down to the "🗺️ GPS-Tagged Locations on Map" section
2. Look for the same landmark you identified in Visual Results
3. Get coordinates:

   **Method A: Right-click** (easiest)
   - Right-click on the landmark in the folium map
   - Some browsers show lat/lon in context menu
   
   **Method B: Browser inspector** (if right-click doesn't work)
   - Right-click map → Inspect Element → Console
   - Type: `map.getCenter()` (shows coordinates)
   
   **Method C: Manual estimation**
   - Use the image center marker coordinates as reference
   - Estimate offset from center
   
   **Method D: Use a separate map viewer**
   - Copy the GPS coordinates from image center marker
   - Open Google Maps / OpenStreetMap at that location
   - Find your landmark and note its GPS coordinates

---

### Step 5: Calculate Heading

1. Scroll to **"🎯 Landmark-Based Alignment"** section (below the map)
2. Click expander: **"📍 Click here if GPS markers don't match..."**
3. Enter coordinates:

   **Left column (Visual Results):**
   - Pixel X coordinate: `1234`
   - Pixel Y coordinate: `567`
   
   **Right column (Map):**
   - Latitude: `10.781234`
   - Longitude: `122.625678`

4. Click **"🧮 Calculate Heading from Landmark"** button

5. System calculates and displays:
   ```
   ✅ Calculated heading: 87.3°
   Direction: East
   ```

---

### Step 6: Apply the Heading and Re-run

1. Scroll back up to **"🧭 Camera Heading Detection"** section
2. Enable **"Method 3: Manual Adjustment"** checkbox
3. Enter the calculated heading (e.g., `87`) in the slider
4. Click **"Analyze Image"** button again
5. New results will show GPS markers correctly aligned! ✓

---

## 🎓 For Your Thesis Defense

**What to say:**
> "Initially, GPS markers were misaligned because GPS EXIF doesn't contain camera heading. We implemented landmark-based alignment where users click corresponding points in the image and map. The system calculates rotation using coordinate transformation mathematics, eliminating manual guessing. This ensures accurate GPS tagging for field deployment."

**Why this impresses:**
- Shows you understand geospatial coordinate systems
- Demonstrates problem-solving (when auto-detection failed)
- Provides user-friendly solution
- Mathematically rigorous approach
- Practical for field deployment

---

## 🔧 Troubleshooting

**Q: Calculated heading doesn't improve alignment**
- Try a landmark farther from image center
- Ensure landmark is correctly identified in both views
- Check pixel coordinates are correct (Y increases downward)

**Q: Can't find landmark on map**
- Ensure orthophoto tile server is running
- Check image GPS coordinates are within map bounds
- Try zooming map in/out

**Q: How many landmarks needed?**
- **1 landmark** is sufficient for rotation calculation
- **2-3 landmarks** can improve accuracy (system uses median)
- More than 3 is unnecessary and slower

**Q: What if no clear landmarks visible?**
- Use the manual slider as fallback
- Try comparing edge features (coastline, paths)
- Consider using shadow direction if sunny

---

## 📐 Behind the Scenes (Technical Details)

**What the system does:**

1. **Convert pixels to meters:**
   ```
   offset_x_m = (px - image_center_x) * GSD
   offset_y_m = -(py - image_center_y) * GSD  # Y-flip
   ```

2. **Convert GPS to meters:**
   ```
   offset_x_m = (lon - center_lon) * meters_per_degree_lon
   offset_y_m = (lat - center_lat) * meters_per_degree_lat
   ```

3. **Calculate angles:**
   ```
   image_angle = atan2(image_offset_y, image_offset_x)
   map_angle = atan2(map_offset_y, map_offset_x)
   rotation = map_angle - image_angle
   ```

4. **Normalize to 0-360°:**
   ```
   heading = rotation % 360
   ```

This heading is then used to rotate ALL hexagon coordinates before converting to GPS.

---

## 🚀 Best Practices

1. **Choose distinctive landmarks** - sharp corners, isolated structures
2. **Use landmarks near edges** - better angle resolution
3. **Be precise with coordinates** - especially pixel values
4. **Re-run analysis** after calculating heading - don't just note the number
5. **Save the heading value** - same heading works for all images from same flight

---

## 💡 Pro Tips

- If you have multiple images from the same flight, you only need to do this ONCE
- The calculated heading applies to all images taken during that flight (camera doesn't rotate mid-flight)
- Record the heading in your field notes for future reference
- For thesis documentation, include screenshots showing before/after alignment

---

**This feature demonstrates advanced GIS understanding and provides accurate, reproducible results for your research!** ✨
