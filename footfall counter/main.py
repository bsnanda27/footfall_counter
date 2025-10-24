#!/usr/bin/env python3

from counter import FootfallCounter
import sys
import os
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path> [line_position]")
        print("Example: python main.py video.mp4 0.5")
        sys.exit(1)
    
    video_path = sys.argv[1]
    line_position = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    # Create results directory if it doesn't exist
    results_dir = Path("../results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    counter = FootfallCounter(video_path, line_position)
    
    # Prepare output video path with input filename
    input_name = Path(video_path).stem
    output_video_path = results_dir / f"{input_name}_footfall_output.mp4"
    csv_file = results_dir / f"{input_name}_footfall_results.csv"
    
    print(f"[*] Output video will be saved to: {output_video_path}")
    print(f"[*] CSV results will be saved to: {csv_file}")
    
    # Run processing with video output
    counter.run(display=True, output_video_path=str(output_video_path))
    
    # Export CSV to results folder
    counter.export_csv(str(csv_file))
    
    print(f"\n[*] Summary:")
    print(f"    Entry Zone: {counter.in_zone}")
    print(f"    Entry Cross: {counter.in_crossed}")
    print(f"    Exit Zone: {counter.out_zone}")
    print(f"    Exit Cross: {counter.out_crossed}")
    print(f"    Events: {len(counter.events)}")
    print(f"    CSV: {csv_file}")
    print(f"    Video: {output_video_path}")


if __name__ == "__main__":
    main()