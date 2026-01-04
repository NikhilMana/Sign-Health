import sqlite3
import bcrypt
from pathlib import Path
from datetime import datetime

class Database:
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_db()
    
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                full_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Consultations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS consultations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                doctor_id INTEGER NOT NULL,
                transcript TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES users(id),
                FOREIGN KEY (doctor_id) REFERENCES users(id)
            )
        ''')
        
        conn.commit()
        conn.close()

class User:
    def __init__(self, id, email, role, full_name):
        self.id = id
        self.email = email
        self.role = role
        self.full_name = full_name
        self.is_authenticated = True
        self.is_active = True
        self.is_anonymous = False
    
    def get_id(self):
        return str(self.id)
    
    @staticmethod
    def hash_password(password):
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    @staticmethod
    def check_password(password, password_hash):
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    @staticmethod
    def create_user(db, email, password, full_name, role):
        conn = db.get_connection()
        cursor = conn.cursor()
        
        password_hash = User.hash_password(password)
        
        try:
            cursor.execute(
                'INSERT INTO users (email, password_hash, role, full_name) VALUES (?, ?, ?, ?)',
                (email, password_hash, role, full_name)
            )
            conn.commit()
            user_id = cursor.lastrowid
            conn.close()
            return User(user_id, email, role, full_name)
        except sqlite3.IntegrityError:
            conn.close()
            return None
    
    @staticmethod
    def get_by_email(db, email):
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return User(row['id'], row['email'], row['role'], row['full_name'])
        return None
    
    @staticmethod
    def get_by_id(db, user_id):
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return User(row['id'], row['email'], row['role'], row['full_name'])
        return None
    
    @staticmethod
    def authenticate(db, email, password):
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        row = cursor.fetchone()
        conn.close()
        
        if row and User.check_password(password, row['password_hash']):
            return User(row['id'], row['email'], row['role'], row['full_name'])
        return None
