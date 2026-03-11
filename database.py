"""
Database Module for Face Recognition Attendance System
Handles SQLite database operations for storing attendance records.
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Tuple


class AttendanceDatabase:
    """
    Manages SQLite database operations for the attendance system.
    Provides methods to create tables, add records, and query attendance data.
    """

    def __init__(self, db_path: str = "attendance/attendance.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to the SQLite database file
        """
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.connection: Optional[sqlite3.Connection] = None
        self._create_tables()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
        return self.connection

    def _create_tables(self) -> None:
        """Create necessary tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name, date)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                folder_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()

    def is_already_marked(self, name: str, date: Optional[str] = None) -> bool:
        """
        Check if attendance is already marked for a student on a given date.
        
        Args:
            name: Student name
            date: Date string (YYYY-MM-DD format). Defaults to today.
            
        Returns:
            True if attendance already exists, False otherwise
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT COUNT(*) FROM attendance WHERE name = ? AND date = ?",
            (name, date)
        )
        count = cursor.fetchone()[0]
        return count > 0

    def mark_attendance(
        self, 
        name: str, 
        confidence: float = 0.0,
        date: Optional[str] = None,
        time: Optional[str] = None
    ) -> bool:
        """
        Mark attendance for a student.
        
        Args:
            name: Student name
            confidence: Face match confidence (0-1)
            date: Date string (YYYY-MM-DD). Defaults to today.
            time: Time string (HH:MM:SS). Defaults to current time.
            
        Returns:
            True if attendance was marked, False if already exists
        """
        now = datetime.now()
        if date is None:
            date = now.strftime("%Y-%m-%d")
        if time is None:
            time = now.strftime("%H:%M:%S")
        
        if self.is_already_marked(name, date):
            return False
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO attendance (name, date, time, confidence) VALUES (?, ?, ?, ?)",
                (name, date, time, confidence)
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_attendance_by_date(self, date: str) -> List[Tuple]:
        """
        Get all attendance records for a specific date.
        
        Args:
            date: Date string (YYYY-MM-DD)
            
        Returns:
            List of attendance records
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT name, date, time, confidence FROM attendance WHERE date = ? ORDER BY time",
            (date,)
        )
        return cursor.fetchall()

    def get_attendance_by_student(self, name: str) -> List[Tuple]:
        """
        Get all attendance records for a specific student.
        
        Args:
            name: Student name
            
        Returns:
            List of attendance records
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT name, date, time, confidence FROM attendance WHERE name = ? ORDER BY date DESC",
            (name,)
        )
        return cursor.fetchall()

    def get_all_students_today(self) -> List[str]:
        """
        Get list of all students who marked attendance today.
        
        Returns:
            List of student names
        """
        today = datetime.now().strftime("%Y-%m-%d")
        records = self.get_attendance_by_date(today)
        return [record[0] for record in records]

    def register_student(self, name: str, folder_path: str) -> bool:
        """
        Register a new student in the database.
        
        Args:
            name: Student name
            folder_path: Path to student's image folder
            
        Returns:
            True if registered successfully, False if already exists
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO students (name, folder_path) VALUES (?, ?)",
                (name, folder_path)
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_all_students(self) -> List[str]:
        """Get list of all registered students."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM students ORDER BY name")
        return [row[0] for row in cursor.fetchall()]

    def get_attendance_report(
        self, 
        start_date: str, 
        end_date: str
    ) -> List[Tuple]:
        """
        Get attendance report for a date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            List of attendance records
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT name, date, time, confidence 
            FROM attendance 
            WHERE date BETWEEN ? AND ? 
            ORDER BY date, time
            """,
            (start_date, end_date)
        )
        return cursor.fetchall()

    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def get_database(db_path: str = "attendance/attendance.db") -> AttendanceDatabase:
    """Get a database instance."""
    return AttendanceDatabase(db_path)


if __name__ == "__main__":
    db = AttendanceDatabase()
    
    print("Testing database operations...")
    
    result = db.mark_attendance("Test Student", confidence=0.95)
    print(f"Mark attendance: {'Success' if result else 'Already exists'}")
    
    result = db.mark_attendance("Test Student")
    print(f"Mark again: {'Success' if result else 'Already exists (expected)'}")
    
    today = datetime.now().strftime("%Y-%m-%d")
    records = db.get_attendance_by_date(today)
    print(f"\nToday's attendance ({today}):")
    for record in records:
        print(f"  - {record[0]} at {record[2]}")
    
    db.close()
    print("\nDatabase test completed!")
