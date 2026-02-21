"""
Working Mangrove Canopy Detection System with Accurate GSD
Fixed: 6-meter altitude, 90-degree angle
"""

import os
import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon
from matplotlib.patches import Patch

import config
from gsd_calculator import GSDCalculator


class MangroveCanopyDetector:
    """
    Canopy detector with accurate Ground Sample Distance (GSD) calculations
    Fixed at 6-meter altitude, 90-degree overhead angle
    """
    
    def __init__(self):
        self.buffer_distance = config.BUFFER_DISTANCE_METERS
        self.altitude_m = config.FLIGHT_ALTITUDE_METERS
        
        print(f"Initializing Mangrove Canopy Detector...")
        print(f"Flight altitude: {self.altitude_m}m (90\u00b0 overhead angle)")
        print(f"Buffer distance: {self.buffer_distance}m around canopies")
        print(f"Planting spacing: {self.buffer_distance}m between points\n")
        
    def detect_green_regions(self, image, min_area=500):
        """Detect green regions (potential tree canopies)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        canopy_polygons = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                points = contour.reshape(-1, 2)
                if len(points) >= 3:
                    poly = Polygon(points)
                    if poly.is_valid:
                        canopy_polygons.append(poly)
        
        return canopy_polygons, mask
    
    def create_buffer_zones(self, canopy_polygons, buffer_pixels):
        """Create buffer zones around detected canopies"""
        if not canopy_polygons:
            return None
        
        buffered_zones = [poly.buffer(buffer_pixels) for poly in canopy_polygons]
        merged_buffer = unary_union(buffered_zones)
        return merged_buffer
    
    def identify_plantable_zones(self, image_shape, danger_zones):
        """Identify plantable zones = Total area - Danger zones"""
        height, width = image_shape[:2]
        
        total_area = Polygon([
            (0, 0), (width, 0), (width, height), (0, height)
        ])
        
        if danger_zones and danger_zones.is_valid:
            try:
                plantable_zones = total_area.difference(danger_zones)
            except:
                plantable_zones = total_area
        else:
            plantable_zones = total_area
        
        total_pixels = width * height
        danger_pixels = danger_zones.area if danger_zones else 0
        plantable_pixels = plantable_zones.area if plantable_zones else total_pixels
        
        stats = {
            'total_pixels': total_pixels,
            'danger_pixels': int(danger_pixels),
            'plantable_pixels': int(plantable_pixels),
            'danger_percent': (danger_pixels / total_pixels) * 100,
            'plantable_percent': (plantable_pixels / total_pixels) * 100
        }
        
        return plantable_zones, stats
    
    def generate_planting_points(self, plantable_zones, spacing_pixels, image_shape):
        """Generate planting points 1 meter apart"""
        if not plantable_zones or plantable_zones.is_empty:
            return []
        
        minx, miny, maxx, maxy = plantable_zones.bounds
        
        if image_shape:
            height, width = image_shape[:2]
            minx = max(minx, 0)
            miny = max(miny, 0)
            maxx = min(maxx, width)
            maxy = min(maxy, height)
        
        planting_points = []
        x = minx + spacing_pixels / 2
        
        while x < maxx:
            y = miny + spacing_pixels / 2
            while y < maxy:
                point = Point(x, y)
                if plantable_zones.contains(point):
                    planting_points.append((x, y))
                y += spacing_pixels
            x += spacing_pixels
        
        return planting_points
    
    def visualize_zones(self, image, canopy_polygons, buffer_zones, planting_points, stats, gsd, output_path):
        """Create 3-panel visualization"""
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # Panel 1: Original
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Drone Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Panel 2: Canopies + Buffers
        img_buffers = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        axes[1].imshow(img_buffers)
        
        for poly in canopy_polygons:
            if poly.exterior:
                coords = np.array(poly.exterior.coords)
                polygon_patch = MPLPolygon(coords, fill=True, 
                                          facecolor='red', alpha=0.5,
                                          edgecolor='darkred', linewidth=2)
                axes[1].add_patch(polygon_patch)
        
        if buffer_zones:
            if hasattr(buffer_zones, 'geoms'):
                for poly in buffer_zones.geoms:
                    if poly.exterior:
                        coords = np.array(poly.exterior.coords)
                        polygon_patch = MPLPolygon(coords, fill=True,
                                                  facecolor='orange', alpha=0.3,
                                                  edgecolor='orange', linewidth=1)
                        axes[1].add_patch(polygon_patch)
            else:
                if buffer_zones.exterior:
                    coords = np.array(buffer_zones.exterior.coords)
                    polygon_patch = MPLPolygon(coords, fill=True,
                                              facecolor='orange', alpha=0.3,
                                              edgecolor='orange', linewidth=1)
                    axes[1].add_patch(polygon_patch)
        
        axes[1].set_title(f'Danger Zones (Canopy + 1m Buffer)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Panel 3: Planting Points
        img_planting = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        axes[2].imshow(img_planting)
        
        if buffer_zones:
            if hasattr(buffer_zones, 'geoms'):
                for poly in buffer_zones.geoms:
                    if poly.exterior:
                        coords = np.array(poly.exterior.coords)
                        polygon_patch = MPLPolygon(coords, fill=True,
                                                  facecolor='red', alpha=0.3,
                                                  edgecolor='red', linewidth=1)
                        axes[2].add_patch(polygon_patch)
            else:
                if buffer_zones.exterior:
                    coords = np.array(buffer_zones.exterior.coords)
                    polygon_patch = MPLPolygon(coords, fill=True,
                                              facecolor='red', alpha=0.3,
                                              edgecolor='red', linewidth=1)
                    axes[2].add_patch(polygon_patch)
        
        if planting_points:
            xs, ys = zip(*planting_points)
            axes[2].scatter(xs, ys, c='lime', s=20, alpha=0.9, 
                          edgecolors='darkgreen', linewidths=0.5, marker='o')
        
        stats_text = f"Altitude: {self.altitude_m}m\n"
        stats_text += f"GSD: {gsd*100:.2f} cm/pixel\n"
        stats_text += f"Danger: {stats['danger_percent']:.1f}%\n"
        stats_text += f"Plantable: {stats['plantable_percent']:.1f}%\n"
        stats_text += f"Points: {len(planting_points)}"
        axes[2].text(10, 70, stats_text,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                    fontsize=10, fontweight='bold', family='monospace')
        
        axes[2].set_title(f'Planting Points (1m apart)', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        legend_elements = [
            Patch(facecolor='red', alpha=0.5, edgecolor='darkred', 
                  label=f'Canopies ({len(canopy_polygons)})'),
            Patch(facecolor='orange', alpha=0.4, 
                  label=f'1m Buffer (No plant)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', 
                      markersize=8, markeredgecolor='darkgreen',
                      label=f'{len(planting_points)} Points (1m spacing)')
        ]
        axes[2].legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
        
        plt.suptitle(f'Mangrove Planting Zone Analysis - {self.altitude_m}m altitude, 90\u00b0 angle',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\u2713 Saved visualization: {output_path}")
        plt.close()
    
    def process_image(self, image_path, output_dir=None):
        """Complete pipeline with accurate GSD"""
        if output_dir is None:
            output_dir = config.OUTPUT_DIR
        
        print(f"\n{'='*70}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"{'='*70}\n")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        print(f"\u2713 Loaded: {image.shape[1]}x{image.shape[0]} pixels")
        
        # Calculate accurate GSD
        print(f"\nCalculating Ground Sample Distance (GSD)...")
        if config.GSD_METERS_PER_PIXEL:
            gsd = config.GSD_METERS_PER_PIXEL
            print(f"\u2713 Using manual GSD: {gsd:.6f} m/pixel ({gsd*100:.3f} cm/pixel)")
        else:
            gsd, specs = GSDCalculator.calculate_gsd_from_drone(
                altitude_m=self.altitude_m,
                drone_model=config.DRONE_MODEL
            )
            print(f"\u2713 Calculated GSD: {gsd:.6f} m/pixel ({gsd*100:.3f} cm/pixel)")
            print(f"  Drone model: {config.DRONE_MODEL}")
        
        ground_width_m = gsd * image.shape[1]
        ground_height_m = gsd * image.shape[0]
        print(f"  Coverage: {ground_width_m:.2f}m \u00d7 {ground_height_m:.2f}m = {ground_width_m * ground_height_m:.2f} m\u00b2")
        
        # Convert 1 meter to pixels
        buffer_pixels = self.buffer_distance / gsd
        print(f"  1 meter = {buffer_pixels:.1f} pixels at this altitude")
        
        # Detect canopies
        print(f"\nDetecting tree canopies...")
        canopy_polygons, mask = self.detect_green_regions(image)
        print(f"\u2713 Detected {len(canopy_polygons)} canopy regions")
        
        # Create buffer zones
        print(f"Creating 1m buffer zones...")
        buffer_zones = self.create_buffer_zones(canopy_polygons, buffer_pixels)
        
        # Identify plantable zones
        print(f"Calculating plantable zones...")
        plantable_zones, stats = self.identify_plantable_zones(image.shape, buffer_zones)
        
        # Generate planting points
        print(f"Generating planting points (1m spacing)...")
        planting_points = self.generate_planting_points(
            plantable_zones, buffer_pixels, image.shape
        )
        
        print(f"\n>> Results:")
        print(f"  Canopies: {len(canopy_polygons)}")
        print(f"  Danger zones: {stats['danger_percent']:.1f}%")
        print(f"  Plantable area: {stats['plantable_percent']:.1f}%")
        print(f"  Planting points: {len(planting_points)}")
        
        # Save results
        base_name = Path(image_path).stem
        
        if config.SAVE_VISUALIZATIONS:
            output_path = os.path.join(output_dir, f"{base_name}_zones.png")
            self.visualize_zones(image, canopy_polygons, buffer_zones, 
                               planting_points, stats, gsd, output_path)
        
        if config.SAVE_JSON:
            json_path = os.path.join(output_dir, f"{base_name}_results.json")
            results = {
                'image_path': image_path,
                'image_size': {'width': image.shape[1], 'height': image.shape[0]},
                'flight_altitude_meters': self.altitude_m,
                'camera_angle_degrees': 90,
                'gsd_meters_per_pixel': gsd,
                'gsd_cm_per_pixel': gsd * 100,
                'ground_coverage_meters': {
                    'width': ground_width_m,
                    'height': ground_height_m,
                    'area': ground_width_m * ground_height_m
                },
                'buffer_distance_meters': self.buffer_distance,
                'buffer_distance_pixels': buffer_pixels,
                'num_canopies_detected': len(canopy_polygons),
                'num_planting_points': len(planting_points),
                'statistics': stats
            }
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\u2713 Saved JSON: {json_path}")
        
        print(f"\n{'='*70}")
        print(">> Processing complete!")
        print(f"{'='*70}\n")
        
        return {
            'canopy_polygons': canopy_polygons,
            'buffer_zones': buffer_zones,
            'plantable_zones': plantable_zones,
            'planting_points': planting_points,
            'stats': stats,
            'gsd': gsd
        }


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("MANGROVE PLANTING ZONE ANALYZER")
    print(f"Fixed: {config.FLIGHT_ALTITUDE_METERS}m altitude, 90\u00b0 overhead angle")
    print("="*70 + "\n")
    
    detector = MangroveCanopyDetector()
    
    # Find images
    image_files = []
    for ext in config.IMAGE_EXTENSIONS:
        image_files.extend(Path(config.DRONE_IMAGES_DIR).glob(f"*{ext}"))
    
    if not image_files:
        print(f"X No images found in: {config.DRONE_IMAGES_DIR}")
        return
    
    print(f">> Found {len(image_files)} image(s)\n")
    
    # Process all images
    for image_path in image_files:
        try:
            detector.process_image(str(image_path))
        except Exception as e:
            print(f"X Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print(">> ALL COMPLETE!")
    print("="*70)
    print(f"\n>> Results: {config.OUTPUT_DIR}")
    print("\nRed = Danger zones (canopy + 1m buffer)")
    print("Green dots = Planting points (1m apart)")


if __name__ == "__main__":
    main()
