"""
FastAPI Server for Footfall Counter
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import tempfile
from pathlib import Path
from counter import FootfallCounter
import traceback

app = FastAPI(title="Footfall Counter API", version="1.0")

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)


@app.post("/count")
async def count_footfall(video: UploadFile = File(...)):
    """Upload video and get footfall counts"""
    try:
        # Save video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            content = await video.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Process
        counter = FootfallCounter(tmp_path)
        
        # Prepare output paths
        video_stem = Path(video.filename).stem
        csv_name = f"{video_stem}_footfall_results.csv"
        video_name = f"{video_stem}_footfall_output.mp4"
        csv_path = results_dir / csv_name
        video_path = results_dir / video_name
        
        # Run processing with video output
        counter.run(display=False, output_video_path=str(video_path))
        
        # Export CSV
        counter.export_csv(str(csv_path))
        
        # Cleanup
        Path(tmp_path).unlink()
        
        return {
            "status": "success",
            "summary": counter.get_summary(),
            "csv_file": csv_name,
            "video_file": video_name
        }
    
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download CSV or video file"""
    file_path = results_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine media type based on extension
    if filename.endswith('.csv'):
        media_type = 'text/csv'
    elif filename.endswith('.mp4'):
        media_type = 'video/mp4'
    else:
        media_type = 'application/octet-stream'
    
    return FileResponse(path=file_path, filename=filename, media_type=media_type)


@app.get("/results")
async def list_results():
    """List all result files (CSV and video)"""
    csv_files = list(results_dir.glob("*.csv"))
    video_files = list(results_dir.glob("*.mp4"))
    
    return {
        "csv_files": [f.name for f in csv_files],
        "video_files": [f.name for f in video_files],
        "total_files": len(csv_files) + len(video_files)
    }


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)