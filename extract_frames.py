import cv2
import os

# ================= CONFIGURATION =================
# List all your video files here (relative to script location or absolute paths)
VIDEO_FILES = ["flight_1.mp4", "flight_2.mp4", "flight_3.mp4", "flight_4.mp4"]
OUTPUT_FOLDER = "dataset_frames"     
INTERVAL_SECONDS = 2.5               # Extract 1 frame every 2.5 seconds (~60-65% overlap)
JPEG_QUALITY = 100                   # Maximum quality (100 = lossless, critical for SfM)
# =================================================

def extract_frames_from_list(video_list, output_folder, interval, jpeg_quality=100):
    """
    Extract frames from drone videos for photogrammetry/SfM processing.
    
    Args:
        video_list: List of video file paths
        output_folder: Directory to save extracted frames
        interval: Time interval in seconds between extracted frames
        jpeg_quality: JPEG compression quality (0-100, 100=best)
    """
    # Create output folder with robust path handling
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {os.path.abspath(output_folder)}")

    total_images_saved = 0

    # Process each video file
    for video_filename in video_list:
        print(f"\n{'='*60}")
        print(f"Processing: {video_filename}")
        print(f"{'='*60}")
        
        # Validate video file exists
        if not os.path.isfile(video_filename):
            print(f"⚠️  ERROR: File not found: {video_filename}")
            print(f"    Current working directory: {os.getcwd()}")
            continue

        cap = cv2.VideoCapture(video_filename)
        
        if not cap.isOpened():
            print(f"⚠️  ERROR: Could not open video: {video_filename}")
            continue

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video FPS: {fps:.2f}")
        print(f"Total frames: {total_frames}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Expected extractions: ~{int(duration / interval)}")
        
        # Calculate frame interval (ensure at least 1 frame is skipped)
        frame_interval = max(1, int(round(fps * interval)))
        print(f"Extracting every {frame_interval} frames (every {frame_interval/fps:.2f}s)")
        
        frame_count = 0
        saved_count_video = 0

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break 

                # Extract frame at exact intervals (skip frame 0 to avoid temporal gap)
                # This ensures consistent 1.0s spacing: frame 30, 60, 90... at 30fps
                if frame_count > 0 and frame_count % frame_interval == 0:
                    # Extract clean filename without extension
                    base_name = os.path.splitext(os.path.basename(video_filename))[0]
                    
                    # Create sequential filename with proper path handling
                    filename = f"{base_name}_frame_{saved_count_video+1:04d}.jpg"
                    save_path = os.path.join(output_folder, filename)
                    
                    # Write with maximum JPEG quality for SfM feature matching
                    cv2.imwrite(save_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                    saved_count_video += 1
                    total_images_saved += 1
                    
                    # Progress indicator every 10 frames
                    if saved_count_video % 10 == 0:
                        print(f"  Extracted {saved_count_video} frames...", end='\r')

                frame_count += 1

        finally:
            # Ensure resources are released even if error occurs
            cap.release()
        
        print(f"\n✓ Finished {video_filename}: Extracted {saved_count_video} images")

    print(f"\n{'='*60}")
    print(f"✓ ALL DONE! Total images extracted: {total_images_saved}")
    print(f"  Output location: {os.path.abspath(output_folder)}")
    print(f"{'='*60}")

# Run the extraction
if __name__ == "__main__":
    # Get script directory to resolve relative video paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.join(script_dir, "flight_videos")
    
    # Build full paths to video files
    video_paths = [os.path.join(video_dir, vf) for vf in VIDEO_FILES]
    
    print("="*60)
    print("DRONE FRAME EXTRACTION FOR PHOTOGRAMMETRY")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Interval: {INTERVAL_SECONDS}s (for 80% overlap)")
    print(f"  - JPEG Quality: {JPEG_QUALITY}")
    print(f"  - Videos to process: {len(video_paths)}")
    print(f"  - Working directory: {os.getcwd()}")
    print("="*60)
    
    extract_frames_from_list(video_paths, OUTPUT_FOLDER, INTERVAL_SECONDS, JPEG_QUALITY)