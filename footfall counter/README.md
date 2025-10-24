# Footfall Counter

A professional real-time footfall counting system that uses YOLOv8 for detection and tracking to count people crossing a designated line in video. The system allows multiple counts per person and provides detailed logging with visual overlays.

## Overview

This project implements a robust footfall counting solution that:
- Detects and tracks individuals using YOLOv8 (nano model)
- Counts footfall entering and exiting zones based on crossing a horizontal line
- Allows multiple counts per person for accurate tracking
- Outputs processed video with detailed visual overlays
- Exports event logs as CSV files
- Provides a FastAPI server for API-based processing

## Features

- **Real-time Detection & Tracking**: YOLOv8-based person detection with persistent ID tracking
- **Multi-Zone Counting**: Configurable entry/exit zones with margin-based boundaries
- **Debounce Logic**: Prevents duplicate counts with distance-based tracking (20 pixels)
- **Visual Overlays**: 
  - Color-coded zones (green for exit, red for entry)
  - Bounding boxes with confidence scores and direction indicators
  - Person tracking trails (up to 30 points)
  - Real-time statistics panel with zone and crossing breakdowns
- **CSV Export**: Detailed event logging with timestamps, frame numbers, and event types
- **FastAPI Server**: HTTP endpoints for video upload and result retrieval
- **FPS Monitoring**: Real-time processing performance metrics

## Technical Architecture

### Core Components

**`counter.py`** - Main processing engine
- `PersonTrack` class: Manages individual person tracking state with debounce logic
- `FootfallCounter` class: Main counting logic and frame processing
- Zone crossing detection with multi-pass counting support
- Debounce distance: 20 pixels (configurable)

**`main.py`** - Command-line interface
- Entry point for batch video processing
- Handles video loading and output file generation with proper naming

**`api.py`** - FastAPI web server
- HTTP endpoints for video upload and processing
- Result file serving and management

## Installation

### Prerequisites
- Python 3.7+
- FFmpeg (for video processing)

### Setup

1. Clone or download the project

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install opencv-python ultralytics fastapi uvicorn numpy
```

4. Verify YOLOv8 model download:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Usage

### Command Line Processing

Process a video file with the footfall counting system:

```bash
python main.py <videopath> [lineposition]
```

**Arguments:**
- `<videopath>`: Path to input video file (e.g., `video.mp4`, `./videos/sample.mp4`)
- `[lineposition]`: Optional horizontal line position (0.0 = top, 1.0 = bottom, default = 0.5)

**Examples:**

```bash
# Basic usage with default line at 50% height
python main.py sample_video.mp4

# Custom line position at 60% height
python main.py sample_video.mp4 0.6

# With full path
python main.py /path/to/video.mp4 0.5
```

### Output Files

After processing, output files are saved in `../results/` directory with the following naming convention:

| File Pattern | Description |
|------|-------------|
| `{input_name}_footfall_output.mp4` | Video with overlays showing detection boxes, zones, trails, and statistics |
| `{input_name}_footfall_results.csv` | Event log with time, frame, person ID, and event type |

### CSV Output Format

```csv
Time,Frame,Person_ID,Event,Type
0.12,3,1,ENTRY,initial
0.45,11,1,EXIT,crossed
0.67,16,2,ENTRY,crossed
```

Columns:
- **Time**: Event timestamp in seconds (2 decimal places)
- **Frame**: Frame number where event occurred
- **Person_ID**: Unique tracking ID
- **Event**: ENTRY or EXIT
- **Type**: "initial" (detected in zone) or "crossed" (crossed line)

### FastAPI Server

Start the API server:

```bash
python api.py
```
or
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Server runs at `http://localhost:8000`

#### API Endpoints

**Upload and Process Video**
```
POST /count
```
Upload a video file and get footfall count summary
- Returns: `{status: "success", summary: {...}, csv_file: "filename.csv", video_file: "filename.mp4"}`

**Download Results**
```
GET /download/{filename}
```
Download a processed CSV or video file

**List Results**
```
GET /results
```
List all available result files
- Returns: `{csv_files: [...], video_files: [...], total_files: N}`

**Health Check**
```
GET /health
```
Server status check

## Configuration

### Adjustable Parameters (in `counter.py`)

```python
# Zone margins (pixels from counting line)
self.exit_margin = 30      # Distance above line for exit zone
self.entry_margin = 30     # Distance below line for entry zone

# Tracking
self.debounce_distance = 20  # Minimum pixel movement to count again

# Detection (in process_frame method)
conf=0.4                  # Confidence threshold
iou=0.5                   # IoU threshold
classes=[0]               # Person class only

# Size filtering
width < 20 or height < 40  # Minimum bounding box size
```

### Counting Logic

- **Exit Zone**: Above counting line (y < line_y - exit_margin)
- **Entry Zone**: Below counting line (y > line_y + entry_margin)
- **Count Conditions**:
  1. Person detected in zone initially (zone detection)
  2. Person crosses line with sufficient movement (line crossing)
  3. Debounce logic prevents duplicate counts within 20 pixels

## Output Video Details

The processed video includes:

- **Colored Zones**: Green (exit) and red (entry) semi-transparent overlays
- **Detection Boxes**: Color-coded by last direction (green=OUT, red=IN, white=unknown)
- **Center Points**: Small circles marking person center
- **Tracking Trails**: Lines showing recent movement history (last 30 positions)
- **Statistics Panel** (top-left):
  - Entry counts (zone detection + line crossing separately)
  - Exit counts (zone detection + line crossing separately) 
  - FPS (processing speed)
  - Clear breakdown of zone vs crossing events

## Performance

- **Model**: YOLOv8 Nano (fastest, optimized for real-time)
- **Typical FPS**: 30-60 FPS on modern hardware (CPU-dependent)
- **Memory**: ~1GB RAM typical usage
- **GPU Support**: Automatic CUDA detection if available

## Troubleshooting

### Video Won't Load
```
Cannot open: videopath
```
Ensure the file path is correct and the video format is supported (MP4, AVI, MOV).

### Low FPS
- Use smaller video resolution
- Reduce confidence threshold (0.4 → 0.3)
- Check for GPU acceleration availability

### Incorrect Counts
- Adjust line position parameter
- Increase margin values if people are partially in zones
- Check video quality and lighting conditions

### Results Directory Issues
The system automatically creates `../results/` directory. Ensure parent directory has write permissions.

## Example Workflow

```bash
# 1. Process a video
python main.py entrance_video.mp4 0.5

# 2. Check output files in results directory
ls ../results/

# 3. View statistics from CSV
cat ../results/entrance_video_footfall_results.csv

# 4. Play processed video
# Output will be: ../results/entrance_video_footfall_output.mp4
```

## API Usage Example

Using Python requests:

```python
import requests

# Upload video
with open("video.mp4", "rb") as f:
    files = {"video": f}
    response = requests.post("http://localhost:8000/count", files=files)
    result = response.json()
    print(f"Entry: {result['summary']['total_entries']}")
    print(f"Exit: {result['summary']['total_exits']}")
    csv_file = result['csv_file']
    video_file = result['video_file']

# Download CSV results
response = requests.get(f"http://localhost:8000/download/{csv_file}")
with open(f"local_{csv_file}", "wb") as f:
    f.write(response.content)

# Download processed video
response = requests.get(f"http://localhost:8000/download/{video_file}")
with open(f"local_{video_file}", "wb") as f:
    f.write(response.content)
```

## Requirements

| Dependency | Purpose | Version |
|------------|---------|---------|
| opencv-python | Video I/O and image processing | Latest |
| ultralytics | YOLOv8 object detection and tracking | Latest |
| fastapi | Web server framework | Latest |
| uvicorn | ASGI server | Latest |
| numpy | Numerical operations | Latest |

## Summary Statistics Output

The system provides detailed counting statistics:

- **Entry Zone Detection**: People first detected in entry zone
- **Entry Line Crossing**: People crossing from exit to entry zone  
- **Exit Zone Detection**: People first detected in exit zone
- **Exit Line Crossing**: People crossing from entry to exit zone
- **Total Events**: All detection and crossing events logged

## Limitations

- Requires clear video with good visibility
- Performance depends on hardware capability  
- Crowded scenes with occlusions may have reduced accuracy
- Works best with horizontal line detection zones
- Minimum bounding box size filtering may miss small/distant people

## Future Enhancements

- Multi-line counting zones
- Configurable zone shapes (polygons)
- Advanced analytics (heatmaps, dwell time)
- Enhanced GPU acceleration options
- Web UI dashboard
- Database integration for results persistence