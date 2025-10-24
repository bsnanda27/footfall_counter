"""
Footfall Counter
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import time
import csv


class PersonTrack:
    """Track state for each person"""
    def __init__(self, track_id, position, line_y, entry_margin, exit_margin):
        self.track_id = track_id
        self.positions = deque(maxlen=50)
        self.positions.append(position)
        
        cy = position[1]
        self.line_y = line_y
        self.entry_margin = entry_margin
        self.exit_margin = exit_margin
        
        # Zone boundaries
        self.exit_zone_bottom = line_y - exit_margin
        self.entry_zone_top = line_y + entry_margin
        
        # Track state - prevent duplicates with debouncing
        self.last_counted_direction = None
        self.last_counted_y = cy
        self.debounce_distance = 20  # Must move this far to count again
    
    def get_zone(self, cy):
        """Get zone for position: above line (EXIT), below line (ENTRY)"""
        if cy < self.line_y:
            return 'above'  # Exit zone
        else:
            return 'below'  # Entry zone


class FootfallCounter:
    """Professional footfall counting - allows multiple counts per person"""
    
    def __init__(self, video_path, line_position=0.5):
        """Initialize counter"""
        print("="*70)
        print("FOOTFALL COUNTER - FIXED (Multiple counts allowed)")
        print("="*70)
        
        # Load YOLO model
        print("[*] Loading model...")
        self.model = YOLO("yolov8n.pt")
        
        # Open video
        print(f"[*] Opening: {video_path}")
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open: {video_path}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[*] Video: {self.width}x{self.height} @ {self.fps} FPS")
        print(f"[*] Total frames: {self.total_frames}")
        
        # Counting line with zones
        self.line_y = int(self.height * line_position)
        self.exit_margin = 30
        self.entry_margin = 30
        
        # Zone boundaries
        self.exit_zone_bottom = self.line_y - self.exit_margin
        self.entry_zone_top = self.line_y + self.entry_margin
        
        # Track management
        self.tracks = {}
        self.track_trails = defaultdict(lambda: deque(maxlen=30))
        
        # Counting
        self.in_count = 0
        self.out_count = 0
        
        # Separate counts for crossed vs zone detection
        self.in_crossed = 0
        self.out_crossed = 0
        self.in_zone = 0
        self.out_zone = 0
        
        # Events
        self.events = []
        
        # Stats
        self.frame_count = 0
        self.processing_fps = 0
        
        print(f"[*] Exit Zone: Y < {self.exit_zone_bottom}")
        print(f"[*] Counting Line: Y = {self.line_y}")
        print(f"[*] Entry Zone: Y > {self.entry_zone_top}")
        print("[*] Ready!\n")
    
    def process_frame(self, frame):
        """Process frame and return detections"""
        # Run YOLOv8 tracking
        results = self.model.track(
            frame,
            persist=True,
            conf=0.4,
            iou=0.5,
            classes=[0],
            verbose=False
        )
        
        detections = []
        
        if results[0].boxes is None or results[0].boxes.id is None:
            return detections
        
        # Extract tracking data
        boxes = results[0].boxes
        track_ids = boxes.id.int().cpu().tolist()
        bboxes = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        
        for track_id, bbox, conf in zip(track_ids, bboxes, confidences):
            x1, y1, x2, y2 = map(int, bbox)
            
            # Filter by size
            width = x2 - x1
            height = y2 - y1
            if width < 20 or height < 40:
                continue
            
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            detections.append({
                'id': track_id,
                'bbox': (x1, y1, x2, y2),
                'center': (cx, cy),
                'conf': float(conf)
            })
            
            # Create or update track
            if track_id not in self.tracks:
                self.tracks[track_id] = PersonTrack(
                    track_id, (cx, cy), 
                    self.line_y, 
                    self.entry_margin, 
                    self.exit_margin
                )
                # Count initial position
                self._count_initial(track_id, cy)
            else:
                self.tracks[track_id].positions.append((cx, cy))
                # Check for zone crossing (FIXED: ALLOWS MULTIPLE COUNTS)
                self._check_zone_crossing(track_id, cy)
            
            # Store trail
            self.track_trails[track_id].append((cx, cy))
        
        # Clean up lost tracks
        active_ids = set(track_ids)
        lost_ids = set(self.tracks.keys()) - active_ids
        for lost_id in lost_ids:
            if lost_id in self.tracks:
                del self.tracks[lost_id]
        
        return detections
    
    def _count_initial(self, track_id, cy):
        """Count person if detected in zone initially"""
        track = self.tracks[track_id]
        
        # If in exit zone
        if cy < self.exit_zone_bottom:
            self.out_count += 1
            self.out_zone += 1
            track.last_counted_direction = 'OUT'
            
            self.events.append({
                'time': self.frame_count / self.fps,
                'frame': self.frame_count,
                'id': track_id,
                'event': 'EXIT',
                'type': 'initial'
            })
            print(f"[EXIT-INITIAL] ID:{track_id}")
            return
        
        # If in entry zone
        if cy > self.entry_zone_top:
            self.in_count += 1
            self.in_zone += 1
            track.last_counted_direction = 'IN'
            
            self.events.append({
                'time': self.frame_count / self.fps,
                'frame': self.frame_count,
                'id': track_id,
                'event': 'ENTRY',
                'type': 'initial'
            })
            print(f"[ENTRY-INITIAL] ID:{track_id}")
            return
    
    def _check_zone_crossing(self, track_id, cy):
        """Check if person crossed zones"""
        track = self.tracks[track_id]
        
        if len(track.positions) < 2:
            return
        
        prev_cy = track.positions[-2][1]
        curr_cy = cy
        
        prev_zone = track.get_zone(prev_cy)
        curr_zone = track.get_zone(curr_cy)
        
        # ENTRY: Person crosses from above line to below line (coming IN)
        if prev_zone == 'above' and curr_zone == 'below':
            # Only count if they weren't just counted as IN or have moved far enough
            if track.last_counted_direction != 'IN' and abs(curr_cy - track.last_counted_y) > track.debounce_distance:
                self.in_count += 1
                self.in_crossed += 1
                track.last_counted_direction = 'IN'
                track.last_counted_y = curr_cy
                
                self.events.append({
                    'time': self.frame_count / self.fps,
                    'frame': self.frame_count,
                    'id': track_id,
                    'event': 'ENTRY',
                    'type': 'crossed'
                })
                print(f"[ENTRY] ID:{track_id} crossed from above to below")
        
        # EXIT: Person crosses from below line to above line (going OUT)
        elif prev_zone == 'below' and curr_zone == 'above':
            # Only count if they weren't just counted as OUT or have moved far enough
            if track.last_counted_direction != 'OUT' and abs(curr_cy - track.last_counted_y) > track.debounce_distance:
                self.out_count += 1
                self.out_crossed += 1
                track.last_counted_direction = 'OUT'
                track.last_counted_y = curr_cy
                
                self.events.append({
                    'time': self.frame_count / self.fps,
                    'frame': self.frame_count,
                    'id': track_id,
                    'event': 'EXIT',
                    'type': 'crossed'
                })
                print(f"[EXIT] ID:{track_id} crossed from below to above")
    
    def draw_frame(self, frame, detections):
        """Draw professional overlay"""
        overlay = frame.copy()
        
        # Draw zones
        cv2.rectangle(overlay, (0, 0), (self.width, self.exit_zone_bottom),
                     (0, 255, 0), -1)
        
        cv2.rectangle(overlay, (0, self.entry_zone_top), (self.width, self.height),
                     (0, 0, 255), -1)
        
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        
        # Draw boundaries
        cv2.line(frame, (0, self.exit_zone_bottom), (self.width, self.exit_zone_bottom),
                (0, 255, 0), 3)
        
        cv2.line(frame, (0, self.line_y), (self.width, self.line_y),
                (0, 255, 255), 3)
        
        cv2.line(frame, (0, self.entry_zone_top), (self.width, self.entry_zone_top),
                (0, 0, 255), 3)
        
        # Zone labels
        cv2.putText(frame, "EXIT ZONE (OUT)", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, "ENTRY ZONE (IN)", (20, self.height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw detections
        for det in detections:
            track_id = det['id']
            x1, y1, x2, y2 = det['bbox']
            cx, cy = det['center']
            conf = det['conf']
            
            # Color based on last count direction
            if track_id in self.tracks:
                track = self.tracks[track_id]
                if track.last_counted_direction == 'OUT':
                    color = (0, 255, 0)
                    label = f"ID:{track_id} [OUT]"
                elif track.last_counted_direction == 'IN':
                    color = (0, 0, 255)
                    label = f"ID:{track_id} [IN]"
                else:
                    color = (255, 255, 255)
                    label = f"ID:{track_id}"
            else:
                color = (255, 255, 255)
                label = f"ID:{track_id}"
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw center
            cv2.circle(frame, (cx, cy), 4, color, -1)
            
            # Draw trail
            if len(self.track_trails[track_id]) > 1:
                points = list(self.track_trails[track_id])
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], color, 2)
        
        # Draw stats
        self._draw_stats_panel(frame)
        
        return frame
    
    def _draw_stats_panel(self, frame):
        """Draw statistics panel in top left corner"""
        # Compact panel - top left corner
        panel_w = 220
        panel_h = 95
        x_start = 10
        y_start = 10
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y_start), (x_start + panel_w, y_start + panel_h), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border
        cv2.rectangle(frame, (x_start, y_start), (x_start + panel_w, y_start + panel_h), 
                     (0, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "FOOTFALL", (x_start + 10, y_start + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # ENTRY section
        cv2.putText(frame, "ENTRY", (x_start + 10, y_start + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.putText(frame, f"Zone: {self.in_zone}", (x_start + 10, y_start + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 255), 1)
        
        cv2.putText(frame, f"Cross: {self.in_crossed}", (x_start + 10, y_start + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 150, 255), 1)
        
        # EXIT section
        cv2.putText(frame, "EXIT", (x_start + 110, y_start + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.putText(frame, f"Zone: {self.out_zone}", (x_start + 110, y_start + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
        
        cv2.putText(frame, f"Cross: {self.out_crossed}", (x_start + 110, y_start + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 255, 150), 1)
        
        # FPS in corner
        fps_text = f"FPS: {self.processing_fps:.1f}"
        cv2.putText(frame, fps_text, (x_start + 10, y_start + 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def export_csv(self, filename):
        """Export to CSV"""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'Frame', 'Person_ID', 'Event', 'Type'])
            
            for evt in self.events:
                writer.writerow([
                    f"{evt['time']:.2f}",
                    evt['frame'],
                    evt['id'],
                    evt['event'],
                    evt['type']
                ])
        
        print(f"\n[*] CSV exported: {filename}")
    
    def get_summary(self):
        """Get summary"""
        return {
            'total_entries': self.in_count,
            'total_exits': self.out_count,
            'total_frames': self.frame_count,
            'events': len(self.events)
        }
    
    def run(self, display=True, output_video_path=None):
        """Main processing loop"""
        print("[*] Processing...\n")
        
        fps_time = time.time()
        fps_counter = 0
        
        # Video writer for output
        video_writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, self.fps, (self.width, self.height))
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Process frame
            detections = self.process_frame(frame)
            
            # Draw
            frame = self.draw_frame(frame, detections)
            
            # Save to output video
            if video_writer:
                video_writer.write(frame)
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_time > 1.0:
                self.processing_fps = fps_counter / (time.time() - fps_time)
                fps_time = time.time()
                fps_counter = 0
            
            # Display
            if display:
                cv2.imshow("Footfall Counter", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Cleanup
        self.cap.release()
        if video_writer:
            video_writer.release()
        if display:
            cv2.destroyAllWindows()
        
        # Final stats
        print("\n" + "="*70)
        print("FINAL STATISTICS")
        print("="*70)
        print(f"Entry Zone Detection:   {self.in_zone}")
        print(f"Entry Line Crossing:    {self.in_crossed}")
        print(f"Exit Zone Detection:    {self.out_zone}")
        print(f"Exit Line Crossing:     {self.out_crossed}")
        print(f"Total Events:           {len(self.events)}")
        print(f"Frames Processed:       {self.frame_count}")
        print("="*70)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Footfall Counter")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--line", type=float, default=0.5, help="Line position (0.0-1.0)")
    
    args = parser.parse_args()
    
    counter = FootfallCounter(args.video, args.line)
    counter.run(display=True)
    
    # Export
    csv_file = args.video.replace('.mp4', '_results.csv')
    counter.export_csv(csv_file)


if __name__ == "__main__":
    main()