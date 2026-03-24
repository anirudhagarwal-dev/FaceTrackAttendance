"""
Face Recognition Utilities Module
Handles face detection, encoding, and recognition operations.
"""

import os
import cv2
import numpy as np
import face_recognition
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class FaceMatch:
    """Represents a face match result."""
    name: str
    confidence: float
    location: Tuple[int, int, int, int] 


class FaceRecognizer:
    """
    Handles face recognition operations including:
    - Loading and encoding known faces from dataset
    - Detecting faces in video frames
    - Matching detected faces with known encodings
    """

    SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

    def __init__(
        self,
        dataset_path: str = "dataset",
        tolerance: float = 0.6,
        model: str = "hog"
    ):
        """
        Initialize the face recognizer.
        
        Args:
            dataset_path: Path to the dataset folder containing student subfolders
            tolerance: Face matching tolerance (lower = stricter, 0.6 is default)
            model: Face detection model ('hog' for CPU, 'cnn' for GPU)
        """
        self.dataset_path = dataset_path
        self.tolerance = tolerance
        self.model = model
        
        self.known_encodings: List[np.ndarray] = []
        self.known_names: List[str] = []
        
        self._encoding_cache: Dict[str, np.ndarray] = {}

    def load_dataset(self) -> int:
        """
        Load and encode all faces from the dataset folder.
        
        Expected structure:
        dataset/
            Student1/
                image1.jpg
                image2.jpg
            Student2/
                image1.jpg
        
        Returns:
            Number of students loaded
        """
        self.known_encodings = []
        self.known_names = []
        
        if not os.path.exists(self.dataset_path):
            print(f"[WARNING] Dataset folder not found: {self.dataset_path}")
            os.makedirs(self.dataset_path, exist_ok=True)
            return 0
        
        student_count = 0
        
        for student_name in os.listdir(self.dataset_path):
            student_folder = os.path.join(self.dataset_path, student_name)
            
            if not os.path.isdir(student_folder):
                continue
            
            encodings = self._load_student_encodings(student_name, student_folder)
            
            if encodings:
                for encoding in encodings:
                    self.known_encodings.append(encoding)
                    self.known_names.append(student_name)
                
                student_count += 1
                print(f"[INFO] Loaded {len(encodings)} image(s) for: {student_name}")
        
        print(f"[INFO] Total students loaded: {student_count}")
        print(f"[INFO] Total face encodings: {len(self.known_encodings)}")
        
        return student_count

    def _load_student_encodings(
        self, 
        student_name: str, 
        folder_path: str
    ) -> List[np.ndarray]:
        """
        Load face encodings for a single student.
        
        Args:
            student_name: Name of the student
            folder_path: Path to student's image folder
            
        Returns:
            List of face encodings for the student
        """
        encodings = []
        
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(self.SUPPORTED_FORMATS):
                continue
            
            image_path = os.path.join(folder_path, filename)
            
            try:
                image = face_recognition.load_image_file(image_path)
            
                face_encodings = face_recognition.face_encodings(image)
                
                if face_encodings:
                    encodings.append(face_encodings[0])
                else:
                    print(f"[WARNING] No face found in: {image_path}")
                    
            except Exception as e:
                print(f"[ERROR] Failed to process {image_path}: {e}")
        
        return encodings

    def detect_and_recognize(
        self, 
        frame: np.ndarray,
        resize_factor: float = 0.25
    ) -> List[FaceMatch]:
        """
        Detect and recognize faces in a video frame.
        
        Args:
            frame: BGR image from OpenCV
            resize_factor: Factor to resize frame for faster processing
            
        Returns:
            List of FaceMatch objects for detected faces
        """
        matches = []
        
        if not self.known_encodings:
            return matches
        
        small_frame = cv2.resize(
            frame, 
            (0, 0), 
            fx=resize_factor, 
            fy=resize_factor
        )
        
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(
            rgb_frame, 
            model=self.model
        )
        
        if not face_locations:
            return matches
        

        face_encodings = face_recognition.face_encodings(
            rgb_frame, 
            face_locations
        )
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            top, right, bottom, left = face_location
            scale = int(1 / resize_factor)
            scaled_location = (
                top * scale,
                right * scale,
                bottom * scale,
                left * scale
            )
            
            name, confidence = self._find_best_match(face_encoding)
            
            matches.append(FaceMatch(
                name=name,
                confidence=confidence,
                location=scaled_location
            ))
        
        return matches

    def _find_best_match(
        self, 
        face_encoding: np.ndarray
    ) -> Tuple[str, float]:
        """
        Find the best matching face from known encodings.
        
        Args:
            face_encoding: Encoding of the face to match
            
        Returns:
            Tuple of (name, confidence) where confidence is 0-1
        """
        if not self.known_encodings:
            return ("Unknown", 0.0)
        
        face_distances = face_recognition.face_distance(
            self.known_encodings, 
            face_encoding
        )
        
        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]
        
        if best_distance <= self.tolerance:
            name = self.known_names[best_match_index]
            confidence = 1.0 - best_distance
            return (name, confidence)
        
        return ("Unknown", 0.0)

    def add_student(
        self, 
        name: str, 
        image_path: str
    ) -> bool:
        """
        Add a new student image to the recognizer (runtime only).
        
        Args:
            name: Student name
            image_path: Path to student's image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                self.known_encodings.append(encodings[0])
                self.known_names.append(name)
                return True
            
            print(f"[WARNING] No face found in: {image_path}")
            return False
            
        except Exception as e:
            print(f"[ERROR] Failed to add student {name}: {e}")
            return False

    def get_student_count(self) -> int:
        """Get the number of unique students loaded."""
        return len(set(self.known_names))

    def get_encoding_count(self) -> int:
        """Get the total number of face encodings loaded."""
        return len(self.known_encodings)

    def get_student_names(self) -> List[str]:
        """Get list of unique student names."""
        return list(set(self.known_names))


def draw_face_box(
    frame: np.ndarray,
    face_match: FaceMatch,
    color: Tuple[int, int, int] = (0, 255, 0),
    unknown_color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
    show_confidence: bool = True
) -> np.ndarray:
    """
    Draw bounding box and label on a face.
    
    Args:
        frame: BGR image
        face_match: FaceMatch object with detection info
        color: Box color for known faces (BGR)
        unknown_color: Box color for unknown faces (BGR)
        thickness: Line thickness
        show_confidence: Whether to show confidence percentage
        
    Returns:
        Frame with drawn annotations
    """
    top, right, bottom, left = face_match.location
    name = face_match.name
    confidence = face_match.confidence
    
    box_color = unknown_color if name == "Unknown" else color
    
    cv2.rectangle(frame, (left, top), (right, bottom), box_color, thickness)
    
    if show_confidence and name != "Unknown":
        label = f"{name} ({confidence*100:.1f}%)"
    else:
        label = name
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    label_thickness = 1
    (label_width, label_height), baseline = cv2.getTextSize(
        label, font, font_scale, label_thickness
    )
    
    cv2.rectangle(
        frame,
        (left, bottom),
        (left + label_width + 10, bottom + label_height + 10),
        box_color,
        cv2.FILLED
    )
    
    cv2.putText(
        frame,
        label,
        (left + 5, bottom + label_height + 5),
        font,
        font_scale,
        (255, 255, 255),
        label_thickness
    )
    
    return frame


def draw_status_bar(
    frame: np.ndarray,
    student_count: int,
    attendance_count: int,
    fps: float = 0.0
) -> np.ndarray:
    """
    Draw a status bar at the top of the frame. //status bar
    
    Args:
        frame: BGR image
        student_count: Number of registered students
        attendance_count: Number of students marked today
        fps: Current frames per second
        
    Returns:
        Frame with status bar
    """
    height, width = frame.shape[:2]
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 35), (50, 50, 50), cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    status_text = f"Students: {student_count} | Marked Today: {attendance_count} | FPS: {fps:.1f} | Press 'q' to quit"
    
    cv2.putText(
        frame,
        status_text,
        (10, 25),
        font,
        0.6,
        (255, 255, 255),
        1
    )
    
    return frame


if __name__ == "__main__":
    print("Testing Face Recognition Module...")
    
    recognizer = FaceRecognizer(dataset_path="dataset")
    count = recognizer.load_dataset()
    
    print(f"\nLoaded {count} students")
    print(f"Student names: {recognizer.get_student_names()}")
    print(f"Total encodings: {recognizer.get_encoding_count()}")
    print("\nFace recognition module test completed!")
