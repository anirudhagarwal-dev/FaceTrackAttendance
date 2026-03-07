"""
Face Recognition Based Attendance System
=========================================
Main application file that runs the attendance system.

Features:
- Real-time face detection and recognition
- Automatic attendance marking to CSV and SQLite
- Duplicate prevention (one entry per student per day)
- FPS display and performance optimization

Usage:
    python main.py

Controls:
    - Press 'q' to quit
    - Press 'r' to reload dataset
    - Press 's' to show today's attendance

Author: Face Recognition Attendance System
"""

import os
import sys
import csv
import time
import cv2
import pandas as pd
from datetime import datetime
from typing import Set, Optional

from face_utils import FaceRecognizer, draw_face_box, draw_status_bar, FaceMatch
from database import AttendanceDatabase


class AttendanceSystem:
    """
    Main attendance system that combines face recognition with attendance tracking.
    """

    def __init__(
        self,
        dataset_path: str = "dataset",
        attendance_folder: str = "attendance",
        camera_index: int = 0,
        tolerance: float = 0.6,
        resize_factor: float = 0.25,
        process_every_n_frames: int = 2
    ):
        """
        Initialize the attendance system.
        
        Args:
            dataset_path: Path to student images folder
            attendance_folder: Path to store attendance files
            camera_index: Camera device index (0 for default webcam)
            tolerance: Face matching tolerance (lower = stricter)
            resize_factor: Frame resize factor for faster processing
            process_every_n_frames: Process face recognition every N frames
        """
        self.dataset_path = dataset_path
        self.attendance_folder = attendance_folder
        self.camera_index = camera_index
        self.resize_factor = resize_factor
        self.process_every_n_frames = process_every_n_frames
        
        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(attendance_folder, exist_ok=True)
        
        print("[INFO] Initializing face recognizer...")
        self.recognizer = FaceRecognizer(
            dataset_path=dataset_path,
            tolerance=tolerance,
            model="hog" 
        )
        
        print("[INFO] Initializing database...")
        self.db = AttendanceDatabase(
            db_path=os.path.join(attendance_folder, "attendance.db")
        )
        
        self.csv_path = self._get_csv_path()
        
        self.marked_today: Set[str] = set()
        
        self.cap: Optional[cv2.VideoCapture] = None
        
        self.frame_count = 0
        self.fps = 0.0
        self.fps_start_time = time.time()

    def _get_csv_path(self, date: Optional[str] = None) -> str:
        """Get CSV file path for a given date."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.attendance_folder, f"attendance_{date}.csv")

    def _initialize_csv(self) -> None:
        """Initialize CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Date", "Time", "Confidence"])
            print(f"[INFO] Created attendance file: {self.csv_path}")

    def _load_marked_today(self) -> None:
        """Load students already marked today from CSV."""
        self.marked_today = set()
        
        if os.path.exists(self.csv_path):
            try:
                df = pd.read_csv(self.csv_path)
                if 'Name' in df.columns:
                    self.marked_today = set(df['Name'].tolist())
                    print(f"[INFO] Loaded {len(self.marked_today)} attendance records from today")
            except Exception as e:
                print(f"[WARNING] Could not load existing attendance: {e}")

    def mark_attendance(self, name: str, confidence: float) -> bool:
        """
        Mark attendance for a student.
        
        Args:
            name: Student name
            confidence: Recognition confidence
            
        Returns:
            True if attendance was marked, False if already marked
        """
        if name == "Unknown":
            return False
        
        if name in self.marked_today:
            return False
        
        if self.db.is_already_marked(name):
            self.marked_today.add(name)
            return False
        
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        self.db.mark_attendance(name, confidence)
        
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([name, date_str, time_str, f"{confidence:.4f}"])
        
        self.marked_today.add(name)
        
        print(f"[ATTENDANCE] Marked: {name} at {time_str} (Confidence: {confidence*100:.1f}%)")
        return True

    def start(self) -> None:
        """Start the attendance system."""
        print("\n" + "="*60)
        print("  FACE RECOGNITION ATTENDANCE SYSTEM")
        print("="*60 + "\n")
        
        student_count = self.recognizer.load_dataset()
        
        if student_count == 0:
            print("\n[WARNING] No students found in dataset!")
            print(f"[INFO] Add student folders with images to: {self.dataset_path}")
            print("[INFO] Structure: dataset/StudentName/image1.jpg")
            print("\n[INFO] Starting anyway for testing...")
        
        self._initialize_csv()
        self._load_marked_today()
        
        print(f"\n[INFO] Opening camera (index: {self.camera_index})...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print("[ERROR] Could not open camera!")
            print("[INFO] Try changing camera_index in main() or check camera connection")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("[INFO] Camera opened successfully!")
        print("\nControls:")
        print("  - Press 'q' to quit")
        print("  - Press 'r' to reload dataset")
        print("  - Press 's' to show today's attendance")
        print("\n" + "-"*60 + "\n")
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        finally:
            self.cleanup()

    def _main_loop(self) -> None:
        """Main processing loop."""
        last_matches = []
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                print("[ERROR] Failed to grab frame")
                break
            
            self.frame_count += 1
            
            if self.frame_count % self.process_every_n_frames == 0:
                last_matches = self.recognizer.detect_and_recognize(
                    frame, 
                    resize_factor=self.resize_factor
                )
                
                for match in last_matches:
                    if match.name != "Unknown":
                        self.mark_attendance(match.name, match.confidence)
            
            for match in last_matches:
                frame = draw_face_box(frame, match, show_confidence=True)
            
            if self.frame_count % 30 == 0:
                elapsed = time.time() - self.fps_start_time
                self.fps = 30 / elapsed if elapsed > 0 else 0
                self.fps_start_time = time.time()
            
            frame = draw_status_bar(
                frame,
                student_count=self.recognizer.get_student_count(),
                attendance_count=len(self.marked_today),
                fps=self.fps
            )
            
            cv2.imshow("Face Recognition Attendance", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n[INFO] Quit requested")
                break
            elif key == ord('r'):
                print("\n[INFO] Reloading dataset...")
                self.recognizer.load_dataset()
            elif key == ord('s'):
                self._show_today_attendance()

    def _show_today_attendance(self) -> None:
        """Display today's attendance in console."""
        print("\n" + "="*40)
        print("  TODAY'S ATTENDANCE")
        print("="*40)
        
        today = datetime.now().strftime("%Y-%m-%d")
        records = self.db.get_attendance_by_date(today)
        
        if not records:
            print("  No attendance recorded today")
        else:
            for i, record in enumerate(records, 1):
                print(f"  {i}. {record[0]} - {record[2]}")
        
        print("="*40 + "\n")

    def cleanup(self) -> None:
        """Clean up resources."""
        print("\n[INFO] Cleaning up...")
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        self.db.close()
        
        print("\n" + "="*60)
        print("  SESSION SUMMARY")
        print("="*60)
        print(f"  Students registered: {self.recognizer.get_student_count()}")
        print(f"  Attendance marked today: {len(self.marked_today)}")
        print(f"  Attendance file: {self.csv_path}")
        print("="*60 + "\n")


def export_attendance_report(
    start_date: str,
    end_date: str,
    output_file: str = "attendance_report.csv"
) -> None:
    """
    Export attendance report for a date range.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_file: Output CSV file path
    """
    db = AttendanceDatabase()
    records = db.get_attendance_report(start_date, end_date)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time", "Confidence"])
        
        for record in records:
            writer.writerow([record[0], record[1], record[2], f"{record[3]:.4f}"])
    
    print(f"[INFO] Report exported to: {output_file}")
    db.close()


def main():
    """Main entry point."""
    CONFIG = {
        "dataset_path": "dataset",
        "attendance_folder": "attendance",
        "camera_index": 1,
        "tolerance": 0.6, 
        "resize_factor": 0.25, 
        "process_every_n_frames": 2 
    }
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("""
Face Recognition Attendance System
===================================

Usage:
    python main.py              Start the attendance system
    python main.py --help       Show this help message
    python main.py --export     Export attendance report

Configuration:
    Edit the CONFIG dictionary in main() to customize settings.
    
Adding Students:
    1. Create a folder in 'dataset/' with the student's name
    2. Add 3-5 clear face images of the student
    3. Restart the system or press 'r' to reload
            """)
            return
        
        elif sys.argv[1] == "--export":
            from datetime import timedelta
            today = datetime.now()
            week_ago = today - timedelta(days=7)
            
            export_attendance_report(
                week_ago.strftime("%Y-%m-%d"),
                today.strftime("%Y-%m-%d"),
                "attendance_report.csv"
            )
            return
    
    system = AttendanceSystem(**CONFIG)
    system.start()


if __name__ == "__main__":
    main()
