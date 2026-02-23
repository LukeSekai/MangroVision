"""
Flight log and SRT parser for extracting camera heading/yaw data.
Supports DJI .SRT subtitle files and flight log formats.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
import json


class FlightLogParser:
    """Parse DJI flight logs and SRT files to extract heading data."""
    
    def __init__(self):
        self.heading_data = []
    
    def parse_srt_file(self, srt_path: str) -> Optional[float]:
        """
        Parse DJI SRT subtitle file to extract camera heading (yaw).
        
        DJI .SRT format contains telemetry data including:
        - GPS coordinates (lon, lat, alt)
        - Altitude, speeds, camera settings
        - If explicit yaw missing: calculate from GPS trajectory
        
        Args:
            srt_path: Path to .SRT file
            
        Returns:
            Average heading in degrees (0-360, where 0=North, 90=East, etc.)
        """
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Method 1: Try to find explicit heading/yaw fields
            yaw_patterns = [
                r'<font[^>]*>yaw\s*:\s*([-\d.]+)',  # yaw: 123.4
                r'yaw\(([-\d.]+)\)',  # yaw(123.4)
                r'heading\s*:\s*([-\d.]+)',  # heading: 123.4
                r'compass\s*:\s*([-\d.]+)',  # compass: 123.4
                r'gimbal_yaw\(([-\d.]+)\)',  # gimbal_yaw(123.4)
            ]
            
            headings = []
            for pattern in yaw_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    for match in matches:
                        try:
                            heading = float(match)
                            heading = heading % 360
                            if heading < 0:
                                heading += 360
                            headings.append(heading)
                        except ValueError:
                            continue
            
            if headings:
                headings.sort()
                median_heading = headings[len(headings) // 2]
                print(f"✓ Extracted {len(headings)} explicit heading values from SRT")
                print(f"✓ Median heading: {median_heading:.1f}°")
                return median_heading
            
            # Method 2: Extract GPS trajectory and calculate heading
            # DJI format: GPS (lon, lat, alt)
            gps_pattern = r'GPS\s*\(([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)'
            gps_matches = re.findall(gps_pattern, content)
            
            if len(gps_matches) >= 2:
                print(f"  No explicit heading found. Analyzing GPS trajectory ({len(gps_matches)} points)...")
                
                # Convert to lat/lon (DJI format is lon, lat, alt)
                gps_points = []
                for lon_str, lat_str, alt_str in gps_matches:
                    try:
                        lon = float(lon_str)
                        lat = float(lat_str)
                        # Only use points with valid coordinates (not 0,0)
                        if abs(lat) > 0.001 and abs(lon) > 0.001:
                            gps_points.append((lat, lon))
                    except ValueError:
                        continue
                
                if len(gps_points) >= 2:
                    # Calculate heading from GPS trajectory
                    from canopy_detection.gps_trajectory import GPSTrajectoryAnalyzer
                    analyzer = GPSTrajectoryAnalyzer()
                    
                    for lat, lon in gps_points:
                        analyzer.add_point(lat, lon)
                    
                    calculated_heading = analyzer.get_flight_heading()
                    
                    if calculated_heading is not None:
                        print(f"✓ Calculated heading from GPS trajectory: {calculated_heading:.1f}°")
                        return calculated_heading
            
            print("  Could not extract heading from SRT file")
            return None
            
        except Exception as e:
            print(f"Error parsing SRT file: {e}")
            return None
    
    def parse_txt_log(self, log_path: str) -> Optional[float]:
        """
        Parse DJI flight log .TXT file.
        
        Args:
            log_path: Path to flight log .txt file
            
        Returns:
            Average heading in degrees
        """
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for heading/yaw/compass entries
            patterns = [
                r'"yaw"\s*:\s*([-\d.]+)',
                r'"heading"\s*:\s*([-\d.]+)',
                r'"compass"\s*:\s*([-\d.]+)',
                r'yaw=([-\d.]+)',
                r'heading=([-\d.]+)',
            ]
            
            headings = []
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    try:
                        heading = float(match)
                        heading = heading % 360
                        if heading < 0:
                            heading += 360
                        headings.append(heading)
                    except ValueError:
                        continue
            
            if headings:
                headings.sort()
                median_heading = headings[len(headings) // 2]
                print(f"✓ Extracted {len(headings)} heading values from log")
                print(f"✓ Median heading: {median_heading:.1f}°")
                return median_heading
            
            return None
            
        except Exception as e:
            print(f"Error parsing log file: {e}")
            return None
    
    def extract_heading(self, file_path: str) -> Optional[float]:
        """
        Auto-detect file type and extract heading.
        
        Args:
            file_path: Path to .SRT or .TXT flight log
            
        Returns:
            Heading in degrees (0-360) or None if not found
        """
        path = Path(file_path)
        
        if not path.exists():
            print(f"File not found: {file_path}")
            return None
        
        ext = path.suffix.lower()
        
        if ext == '.srt':
            return self.parse_srt_file(file_path)
        elif ext in ['.txt', '.log', '.csv']:
            return self.parse_txt_log(file_path)
        else:
            # Try both parsers
            heading = self.parse_srt_file(file_path)
            if heading is None:
                heading = self.parse_txt_log(file_path)
            return heading


def snap_to_cardinal(heading: float, tolerance: float = 5.0) -> float:
    """
    Snap heading to nearest cardinal direction if close.
    
    Args:
        heading: Heading in degrees (0-360)
        tolerance: Degrees within which to snap to cardinal
        
    Returns:
        Snapped heading
    """
    cardinals = [0, 90, 180, 270]
    
    for cardinal in cardinals:
        diff = abs(heading - cardinal)
        if min(diff, 360 - diff) <= tolerance:
            print(f"  Snapping {heading:.1f}° → {cardinal}° (cardinal direction)")
            return cardinal
    
    return heading


if __name__ == "__main__":
    # Test with example
    parser = FlightLogParser()
    
    # Example: parser.extract_heading("DJI_0001.SRT")
    print("Flight log parser ready.")
    print("Usage: parser.extract_heading('path/to/flight.srt')")
