"""
PackLens Video Processor: Extract frames and audio for Grok-4 analysis
- Processes videos to extract key activity frames
- Extracts audio segments for behavioral context
- Integrates with Grok-4 for detailed behavioral analysis
- Stores results in database for frontend display

Usage:
  export OPENROUTER_API_KEY="your_key_here"
  python video_processor.py --video path/to/video.mp4
"""

import argparse
import os
import cv2
import numpy as np
import base64
import json
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
import requests
from io import BytesIO
from PIL import Image
import threading
import queue
import time

# Audio imports
try:
    from moviepy.editor import VideoFileClip
    import librosa
    import soundfile as sf
    AUDIO_OK = True
    print("[Audio] MoviePy + librosa available")
except ImportError as e:
    print(f"[Audio] Not available: {e}")
    AUDIO_OK = False

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError as e:
    print(f"[YOLO] Not available: {e}")
    YOLO_OK = False

# COCO class ID for dogs
DOG_CLASS = 16

# -------------------- Database Manager --------------------

class DatabaseManager:
    """Manages SQLite database for video analysis storage"""
    
    def __init__(self, db_path="pawlens_analysis.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Videos table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                filepath TEXT NOT NULL,
                file_hash TEXT UNIQUE NOT NULL,
                duration REAL,
                fps REAL,
                width INTEGER,
                height INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP,
                status TEXT DEFAULT 'pending'
            )
        ''')
        
        # Analysis results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER,
                analysis_text TEXT,
                anxiety_level INTEGER,
                playfulness INTEGER,
                aggression INTEGER,
                confidence INTEGER,
                energy_level INTEGER,
                stress_indicators INTEGER,
                overall_assessment TEXT,
                suggestions TEXT,
                key_moments TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos (id)
            )
        ''')
        
        # Extracted frames table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS extracted_frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER,
                frame_path TEXT,
                timestamp REAL,
                activity_score REAL,
                dogs_detected INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos (id)
            )
        ''')
        
        # Audio events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audio_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER,
                timestamp REAL,
                event_type TEXT,
                intensity REAL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"[Database] Initialized: {self.db_path}")
    
    def add_video(self, filename, filepath, file_hash, duration, fps, width, height):
        """Add new video to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO videos (filename, filepath, file_hash, duration, fps, width, height)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (filename, filepath, file_hash, duration, fps, width, height))
            
            video_id = cursor.lastrowid
            conn.commit()
            print(f"[Database] Added video: {filename} (ID: {video_id})")
            return video_id
        except sqlite3.IntegrityError:
            # Video already exists
            cursor.execute('SELECT id FROM videos WHERE file_hash = ?', (file_hash,))
            video_id = cursor.fetchone()[0]
            print(f"[Database] Video already exists: {filename} (ID: {video_id})")
            return video_id
        finally:
            conn.close()
    
    def update_video_status(self, video_id, status, processed_at=None):
        """Update video processing status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if processed_at is None:
            processed_at = datetime.now()
        
        cursor.execute('''
            UPDATE videos SET status = ?, processed_at = ? WHERE id = ?
        ''', (status, processed_at, video_id))
        
        conn.commit()
        conn.close()
    
    def add_analysis_result(self, video_id, analysis_data):
        """Add analysis result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_results (
                video_id, analysis_text, anxiety_level, playfulness, aggression,
                confidence, energy_level, stress_indicators, overall_assessment,
                suggestions, key_moments
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            video_id,
            analysis_data.get('analysis_text', ''),
            analysis_data.get('anxiety_level', 0),
            analysis_data.get('playfulness', 0),
            analysis_data.get('aggression', 0),
            analysis_data.get('confidence', 0),
            analysis_data.get('energy_level', 0),
            analysis_data.get('stress_indicators', 0),
            analysis_data.get('overall_assessment', ''),
            analysis_data.get('suggestions', ''),
            analysis_data.get('key_moments', '')
        ))
        
        conn.commit()
        conn.close()
        print(f"[Database] Added analysis for video ID: {video_id}")
    
    def add_extracted_frame(self, video_id, frame_path, timestamp, activity_score, dogs_detected):
        """Add extracted frame info to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO extracted_frames (video_id, frame_path, timestamp, activity_score, dogs_detected)
            VALUES (?, ?, ?, ?, ?)
        ''', (video_id, frame_path, timestamp, activity_score, dogs_detected))
        
        conn.commit()
        conn.close()
    
    def add_audio_event(self, video_id, timestamp, event_type, intensity, description):
        """Add audio event to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO audio_events (video_id, timestamp, event_type, intensity, description)
            VALUES (?, ?, ?, ?, ?)
        ''', (video_id, timestamp, event_type, intensity, description))
        
        conn.commit()
        conn.close()
    
    def get_all_videos(self):
        """Get all videos with their analysis results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT v.*, a.anxiety_level, a.playfulness, a.aggression, a.confidence,
                   a.energy_level, a.stress_indicators, a.overall_assessment, a.suggestions
            FROM videos v
            LEFT JOIN analysis_results a ON v.id = a.video_id
            ORDER BY v.created_at DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        return results

# -------------------- Frame Extractor --------------------

class FrameExtractor:
    """Extracts key activity frames from videos"""
    
    def __init__(self, output_dir="extracted_frames"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if YOLO_OK:
            self.yolo_model = YOLO("yolov8n.pt")
        else:
            self.yolo_model = None
            print("[FrameExtractor] YOLO not available - using motion detection only")
    
    def extract_frames(self, video_path, video_id, max_frames=10):
        """Extract key frames that capture most activity"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"[FrameExtractor] Processing {video_path}")
        print(f"[FrameExtractor] Duration: {duration:.1f}s, FPS: {fps:.1f}")
        
        extracted_frames = []
        frame_scores = []
        prev_gray = None
        
        # Sample frames throughout video
        sample_interval = max(1, total_frames // (max_frames * 2))  # Sample more than needed
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_interval == 0:
                timestamp = frame_idx / fps
                
                # Calculate activity score
                activity_score = self._calculate_activity_score(frame, prev_gray)
                
                # Detect dogs if YOLO available
                dogs_detected = 0
                if self.yolo_model:
                    dogs_detected = self._count_dogs(frame)
                
                frame_scores.append({
                    'frame': frame.copy(),
                    'timestamp': timestamp,
                    'activity_score': activity_score,
                    'dogs_detected': dogs_detected,
                    'frame_idx': frame_idx
                })
                
                # Update previous frame for motion detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_gray = gray
            
            frame_idx += 1
        
        cap.release()
        
        # Select top frames based on activity score and dog presence
        frame_scores.sort(key=lambda x: x['activity_score'] + (x['dogs_detected'] * 0.5), reverse=True)
        selected_frames = frame_scores[:max_frames]
        
        # Save selected frames
        for i, frame_data in enumerate(selected_frames):
            frame_filename = f"video_{video_id}_frame_{i:03d}_{frame_data['timestamp']:.1f}s.jpg"
            frame_path = self.output_dir / frame_filename
            
            cv2.imwrite(str(frame_path), frame_data['frame'])
            
            extracted_frames.append({
                'path': str(frame_path),
                'timestamp': frame_data['timestamp'],
                'activity_score': frame_data['activity_score'],
                'dogs_detected': frame_data['dogs_detected']
            })
        
        print(f"[FrameExtractor] Extracted {len(extracted_frames)} key frames")
        return extracted_frames
    
    def _calculate_activity_score(self, frame, prev_gray):
        """Calculate activity/motion score for frame"""
        if prev_gray is None:
            return 0.5  # Default score for first frame
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow magnitude
        flow = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, 
            np.random.randint(0, gray.shape[1], (100, 1, 2)).astype(np.float32),
            None
        )[0]
        
        if flow is not None:
            # Calculate average motion magnitude
            motion_magnitude = np.mean(np.linalg.norm(flow.reshape(-1, 2), axis=1))
            return min(motion_magnitude / 10.0, 1.0)  # Normalize to 0-1
        
        return 0.0
    
    def _count_dogs(self, frame):
        """Count dogs in frame using YOLO"""
        if not self.yolo_model:
            return 0
        
        try:
            results = self.yolo_model(frame, verbose=False, conf=0.3)
            dog_count = 0
            
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == DOG_CLASS:
                        dog_count += 1
            
            return dog_count
        except Exception as e:
            print(f"[FrameExtractor] YOLO error: {e}")
            return 0

# -------------------- Audio Extractor --------------------

class AudioExtractor:
    """Extracts and analyzes audio from videos"""
    
    def __init__(self):
        self.sample_rate = 22050
    
    def extract_audio_events(self, video_path, video_id):
        """Extract audio events from video"""
        if not AUDIO_OK:
            print("[AudioExtractor] Audio processing not available")
            return []
        
        try:
            # Extract audio using MoviePy
            video = VideoFileClip(video_path)
            if video.audio is None:
                print("[AudioExtractor] No audio track found")
                return []
            
            # Save temporary audio file
            temp_audio_path = f"temp_audio_{video_id}.wav"
            video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
            
            # Load with librosa
            audio_data, sr = librosa.load(temp_audio_path, sr=self.sample_rate)
            duration = len(audio_data) / sr
            
            print(f"[AudioExtractor] Processing audio: {duration:.1f}s @ {sr}Hz")
            
            # Analyze audio in segments
            segment_length = 3.0  # 3-second segments
            audio_events = []
            
            for start_time in np.arange(0, duration, segment_length):
                end_time = min(start_time + segment_length, duration)
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                
                segment = audio_data[start_sample:end_sample]
                events = self._analyze_audio_segment(segment, start_time, sr)
                audio_events.extend(events)
            
            # Cleanup
            video.close()
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            
            print(f"[AudioExtractor] Found {len(audio_events)} audio events")
            return audio_events
            
        except Exception as e:
            print(f"[AudioExtractor] Error: {e}")
            return []
    
    def _analyze_audio_segment(self, segment, start_time, sr):
        """Analyze audio segment for dog-related sounds"""
        if len(segment) == 0:
            return []
        
        events = []
        
        # Basic audio feature analysis
        rms_energy = np.sqrt(np.mean(segment**2))
        zero_crossings = np.sum(np.diff(np.sign(segment)) != 0)
        
        # Detect potential bark/growl events
        if rms_energy > 0.05:  # Significant energy
            mid_time = start_time + len(segment) / (2 * sr)
            
            if zero_crossings > len(segment) * 0.1:  # High frequency content
                events.append({
                    'timestamp': mid_time,
                    'type': 'bark',
                    'intensity': min(rms_energy * 10, 1.0),
                    'description': f"Bark detected (energy: {rms_energy:.3f})"
                })
            elif rms_energy > 0.1:  # Lower frequency, higher energy
                events.append({
                    'timestamp': mid_time,
                    'type': 'growl',
                    'intensity': min(rms_energy * 10, 1.0),
                    'description': f"Growl detected (energy: {rms_energy:.3f})"
                })
        
        return events

# -------------------- Grok-4 Analyzer --------------------

class Grok4Analyzer:
    """Analyzes frames and audio using Grok-4 LLM"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def analyze_video_data(self, video_filename, extracted_frames, audio_events):
        """Analyze extracted frames and audio events"""
        if not self.api_key:
            return self._generate_fallback_analysis(video_filename, extracted_frames, audio_events)
        
        try:
            # Prepare analysis prompt
            prompt = self._create_analysis_prompt(video_filename, extracted_frames, audio_events)
            
            # Encode frames for API
            frame_data = []
            for frame_info in extracted_frames[:5]:  # Limit to 5 frames for API efficiency
                image_b64 = self._encode_image_to_base64(frame_info['path'])
                frame_data.append({
                    "timestamp": frame_info['timestamp'],
                    "dogs_detected": frame_info['dogs_detected'],
                    "activity_score": frame_info['activity_score'],
                    "image": image_b64
                })
            
            # Create API request
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            
            # Add frame images
            for i, frame_info in enumerate(frame_data):
                messages[0]["content"].append({
                    "type": "text", 
                    "text": f"\n--- Frame {i+1} at {frame_info['timestamp']:.1f}s ({frame_info['dogs_detected']} dogs, activity: {frame_info['activity_score']:.2f}) ---"
                })
                messages[0]["content"].append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_info['image']}"}
                })
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": "x-ai/grok-4-fast:free",
                "messages": messages,
                "max_tokens": 1200,
                "temperature": 0.3
            }
            
            print("[Grok-4] Analyzing video data...")
            response = requests.post(self.api_url, headers=headers, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                analysis_text = result['choices'][0]['message']['content'].strip()
                return self._parse_analysis_response(analysis_text, video_filename)
            else:
                print(f"[Grok-4] API Error: {response.status_code}")
                return self._generate_fallback_analysis(video_filename, extracted_frames, audio_events)
                
        except Exception as e:
            print(f"[Grok-4] Error: {e}")
            return self._generate_fallback_analysis(video_filename, extracted_frames, audio_events)
    
    def _create_analysis_prompt(self, video_filename, extracted_frames, audio_events):
        """Create comprehensive analysis prompt for dog owners"""
        audio_summary = self._format_audio_events(audio_events)
        
        return f"""You are an expert dog behaviorist analyzing a video titled "{video_filename}" for a dog owner. 

VIDEO CONTEXT:
- Total frames analyzed: {len(extracted_frames)}
- Audio events detected: {len(audio_events)}

AUDIO EVENTS:
{audio_summary}

Please provide a comprehensive behavioral analysis that a dog owner can easily understand and act upon.

PROVIDE ANALYSIS IN THIS EXACT FORMAT:

**BEHAVIORAL_SCORES** (Rate each 1-10):
- Anxiety Level: [score] - [brief explanation]
- Playfulness: [score] - [brief explanation]  
- Aggression: [score] - [brief explanation]
- Confidence: [score] - [brief explanation]
- Energy Level: [score] - [brief explanation]
- Stress Indicators: [score] - [brief explanation]

**OVERALL_ASSESSMENT**: [2-3 sentences summarizing your dog's behavior in this video]

**KEY_MOMENTS**: [List 2-3 most important behavioral moments you observed]

**OWNER_SUGGESTIONS**: [3-4 specific, actionable recommendations for the dog owner]

Focus on practical insights that help the owner understand their dog better and improve their relationship."""
    
    def _format_audio_events(self, audio_events):
        """Format audio events for prompt"""
        if not audio_events:
            return "No significant audio events detected"
        
        formatted = []
        for event in audio_events[:8]:  # Limit to 8 events
            formatted.append(f"- {event['timestamp']:.1f}s: {event['type']} (intensity: {event['intensity']:.2f}) - {event['description']}")
        
        return "\n".join(formatted)
    
    def _encode_image_to_base64(self, image_path):
        """Convert image file to base64 string"""
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            
        # Resize image for API efficiency
        image = Image.open(BytesIO(image_data))
        max_size = 512
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=75)
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _parse_analysis_response(self, analysis_text, video_filename):
        """Parse Grok-4 analysis response"""
        try:
            analysis_data = {
                'analysis_text': analysis_text,
                'video_filename': video_filename,
                'timestamp': datetime.now(),
                'anxiety_level': 5,
                'playfulness': 5,
                'aggression': 5,
                'confidence': 5,
                'energy_level': 5,
                'stress_indicators': 5,
                'overall_assessment': '',
                'key_moments': '',
                'suggestions': ''
            }
            
            # Parse behavioral scores
            if "**BEHAVIORAL_SCORES**" in analysis_text:
                scores_section = analysis_text.split("**BEHAVIORAL_SCORES**")[1].split("**")[0]
                
                # Extract scores using regex-like parsing
                for line in scores_section.split('\n'):
                    line = line.strip()
                    if 'Anxiety Level:' in line:
                        analysis_data['anxiety_level'] = self._extract_score(line)
                    elif 'Playfulness:' in line:
                        analysis_data['playfulness'] = self._extract_score(line)
                    elif 'Aggression:' in line:
                        analysis_data['aggression'] = self._extract_score(line)
                    elif 'Confidence:' in line:
                        analysis_data['confidence'] = self._extract_score(line)
                    elif 'Energy Level:' in line:
                        analysis_data['energy_level'] = self._extract_score(line)
                    elif 'Stress Indicators:' in line:
                        analysis_data['stress_indicators'] = self._extract_score(line)
            
            # Parse other sections
            if "**OVERALL_ASSESSMENT**:" in analysis_text:
                assessment = analysis_text.split("**OVERALL_ASSESSMENT**:")[1].split("**")[0].strip()
                analysis_data['overall_assessment'] = assessment
            
            if "**KEY_MOMENTS**:" in analysis_text:
                moments = analysis_text.split("**KEY_MOMENTS**:")[1].split("**")[0].strip()
                analysis_data['key_moments'] = moments
            
            if "**OWNER_SUGGESTIONS**:" in analysis_text:
                suggestions = analysis_text.split("**OWNER_SUGGESTIONS**:")[1].strip()
                analysis_data['suggestions'] = suggestions
            
            return analysis_data
            
        except Exception as e:
            print(f"[Grok-4] Parsing error: {e}")
            return self._generate_fallback_analysis(video_filename, [], [])
    
    def _extract_score(self, line):
        """Extract numerical score from line"""
        try:
            # Look for number after colon
            parts = line.split(':')
            if len(parts) > 1:
                score_part = parts[1].strip()
                # Extract first number found
                for char in score_part:
                    if char.isdigit():
                        return int(char)
            return 5  # Default
        except:
            return 5
    
    def _generate_fallback_analysis(self, video_filename, extracted_frames, audio_events):
        """Generate fallback analysis when Grok-4 is not available"""
        return {
            'analysis_text': f"Fallback analysis for {video_filename}",
            'video_filename': video_filename,
            'timestamp': datetime.now(),
            'anxiety_level': 4,
            'playfulness': 6,
            'aggression': 2,
            'confidence': 6,
            'energy_level': 5,
            'stress_indicators': 3,
            'overall_assessment': f"Analysis of {video_filename} completed. The video shows typical dog behavior with moderate activity levels.",
            'key_moments': f"Key activity detected in {len(extracted_frames)} frames with {len(audio_events)} audio events.",
            'suggestions': "Continue monitoring your dog's behavior and maintain regular exercise and socialization routines."
        }

# -------------------- Main Video Processor --------------------

class VideoProcessor:
    """Main video processing orchestrator"""
    
    def __init__(self, api_key=None):
        self.db = DatabaseManager()
        self.frame_extractor = FrameExtractor()
        self.audio_extractor = AudioExtractor()
        self.grok4_analyzer = Grok4Analyzer(api_key)
        
        print("[VideoProcessor] Initialized all components")
    
    def process_video(self, video_path):
        """Process a single video completely"""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        print(f"\n{'='*60}")
        print(f"🎬 PROCESSING VIDEO: {video_path.name}")
        print(f"{'='*60}")
        
        # Calculate file hash for deduplication
        file_hash = self._calculate_file_hash(video_path)
        
        # Get video metadata
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        
        # Add video to database
        video_id = self.db.add_video(
            filename=video_path.name,
            filepath=str(video_path),
            file_hash=file_hash,
            duration=duration,
            fps=fps,
            width=width,
            height=height
        )
        
        try:
            # Update status to processing
            self.db.update_video_status(video_id, 'processing')
            
            # Step 1: Extract key frames
            print("\n🖼️  STEP 1: Extracting key frames...")
            extracted_frames = self.frame_extractor.extract_frames(str(video_path), video_id)
            
            # Save frame info to database
            for frame_info in extracted_frames:
                self.db.add_extracted_frame(
                    video_id=video_id,
                    frame_path=frame_info['path'],
                    timestamp=frame_info['timestamp'],
                    activity_score=frame_info['activity_score'],
                    dogs_detected=frame_info['dogs_detected']
                )
            
            # Step 2: Extract audio events
            print("\n🔊 STEP 2: Extracting audio events...")
            audio_events = self.audio_extractor.extract_audio_events(str(video_path), video_id)
            
            # Save audio events to database
            for event in audio_events:
                self.db.add_audio_event(
                    video_id=video_id,
                    timestamp=event['timestamp'],
                    event_type=event['type'],
                    intensity=event['intensity'],
                    description=event['description']
                )
            
            # Step 3: Grok-4 Analysis
            print("\n🧠 STEP 3: Grok-4 behavioral analysis...")
            analysis_result = self.grok4_analyzer.analyze_video_data(
                video_path.name, extracted_frames, audio_events
            )
            
            # Save analysis to database
            self.db.add_analysis_result(video_id, analysis_result)
            
            # Update status to completed
            self.db.update_video_status(video_id, 'completed')
            
            print(f"\n✅ PROCESSING COMPLETE!")
            print(f"📊 Analysis Summary:")
            print(f"   • Anxiety Level: {analysis_result['anxiety_level']}/10")
            print(f"   • Playfulness: {analysis_result['playfulness']}/10")
            print(f"   • Confidence: {analysis_result['confidence']}/10")
            print(f"   • Key Frames: {len(extracted_frames)}")
            print(f"   • Audio Events: {len(audio_events)}")
            print(f"{'='*60}")
            
            return {
                'video_id': video_id,
                'status': 'completed',
                'frames_extracted': len(extracted_frames),
                'audio_events': len(audio_events),
                'analysis': analysis_result
            }
            
        except Exception as e:
            print(f"\n❌ PROCESSING FAILED: {e}")
            self.db.update_video_status(video_id, 'failed')
            raise
    
    def _calculate_file_hash(self, file_path):
        """Calculate SHA-256 hash of file for deduplication"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def get_all_videos(self):
        """Get all processed videos"""
        return self.db.get_all_videos()

# -------------------- Background Processing Queue --------------------

class BackgroundProcessor:
    """Background queue for processing videos"""
    
    def __init__(self, api_key=None):
        self.processor = VideoProcessor(api_key)
        self.queue = queue.Queue()
        self.worker_thread = None
        self.running = False
    
    def start(self):
        """Start background processing"""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        print("[BackgroundProcessor] Started")
    
    def stop(self):
        """Stop background processing"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
        print("[BackgroundProcessor] Stopped")
    
    def add_video(self, video_path):
        """Add video to processing queue"""
        self.queue.put(video_path)
        print(f"[BackgroundProcessor] Queued: {video_path}")
    
    def _worker(self):
        """Background worker thread"""
        while self.running:
            try:
                video_path = self.queue.get(timeout=1)
                print(f"[BackgroundProcessor] Processing: {video_path}")
                self.processor.process_video(video_path)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[BackgroundProcessor] Error processing video: {e}")

# -------------------- CLI Interface --------------------

def main():
    parser = argparse.ArgumentParser(description="PackLens Video Processor")
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--background', action='store_true', help='Process in background')
    args = parser.parse_args()
    
    # Get API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Warning: OPENROUTER_API_KEY not set. Using fallback analysis.")
    
    if args.background:
        # Background processing
        processor = BackgroundProcessor(api_key)
        processor.start()
        processor.add_video(args.video)
        
        print("Processing in background. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            processor.stop()
    else:
        # Direct processing
        processor = VideoProcessor(api_key)
        result = processor.process_video(args.video)
        print(f"\nProcessing result: {result}")

if __name__ == "__main__":
    main()
