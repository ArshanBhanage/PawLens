"""
PackLens API Server: REST API for frontend communication
- Serves video analysis data to frontend
- Handles video upload and processing
- Provides real-time processing status
- Background video processing integration

Usage:
  export OPENROUTER_API_KEY="your_key_here"
  python api_server.py
"""

import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
import threading
import time
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import uuid
from dotenv import load_dotenv

from video_processor import VideoProcessor, BackgroundProcessor

# Load environment variables from .env file
load_dotenv()

# Custom JSON encoder to handle bytes and other non-serializable objects
class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            # Convert bytes to string, handling potential encoding issues
            try:
                return obj.decode('utf-8')
            except UnicodeDecodeError:
                return str(obj)[2:-1]  # Remove b' and ' from string representation
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# -------------------- Flask App Setup --------------------

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Custom jsonify function that handles bytes and other non-serializable objects
def safe_jsonify(data, status_code=200):
    """Custom jsonify that handles bytes and other non-serializable objects"""
    from flask import Response
    json_str = json.dumps(data, cls=SafeJSONEncoder, indent=2)
    return Response(json_str, status=status_code, mimetype='application/json')

# Configuration
UPLOAD_FOLDER = Path('uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Global background processor
background_processor = None

# -------------------- Helper Functions --------------------

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db_connection():
    """Get database connection"""
    return sqlite3.connect('pawlens_analysis.db')

def dict_factory(cursor, row):
    """Convert sqlite row to dictionary"""
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

# -------------------- API Endpoints --------------------

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/videos', methods=['GET'])
def get_all_videos():
    """Get all videos with their analysis results"""
    try:
        conn = get_db_connection()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                v.id, v.filename, v.filepath, v.duration, v.fps, v.width, v.height,
                v.created_at, v.processed_at, v.status,
                a.anxiety_level, a.playfulness, a.aggression, a.confidence,
                a.energy_level, a.stress_indicators, a.overall_assessment, 
                a.suggestions, a.key_moments,
                COUNT(f.id) as frame_count,
                COUNT(ae.id) as audio_event_count
            FROM videos v
            LEFT JOIN analysis_results a ON v.id = a.video_id
            LEFT JOIN extracted_frames f ON v.id = f.video_id
            LEFT JOIN audio_events ae ON v.id = ae.video_id
            GROUP BY v.id
            ORDER BY v.created_at DESC
        ''')
        
        videos = cursor.fetchall()
        conn.close()
        
        # Format response
        formatted_videos = []
        for video in videos:
            formatted_video = {
                'id': video['id'],
                'filename': video['filename'],
                'duration': video['duration'],
                'fps': video['fps'],
                'dimensions': f"{video['width']}x{video['height']}" if video['width'] else None,
                'created_at': video['created_at'],
                'processed_at': video['processed_at'],
                'status': video['status'],
                'frame_count': video['frame_count'] or 0,
                'audio_event_count': video['audio_event_count'] or 0,
                'analysis': {
                    'anxiety_level': video['anxiety_level'],
                    'playfulness': video['playfulness'],
                    'aggression': video['aggression'],
                    'confidence': video['confidence'],
                    'energy_level': video['energy_level'],
                    'stress_indicators': video['stress_indicators'],
                    'overall_assessment': video['overall_assessment'],
                    'suggestions': video['suggestions'],
                    'key_moments': video['key_moments']
                } if video['anxiety_level'] is not None else None
            }
            formatted_videos.append(formatted_video)
        
        return jsonify({
            'success': True,
            'videos': formatted_videos,
            'total': len(formatted_videos)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/videos/<int:video_id>', methods=['GET'])
def get_video_details(video_id):
    """Get detailed information for a specific video"""
    try:
        conn = get_db_connection()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        
        # Get video info
        cursor.execute('SELECT * FROM videos WHERE id = ?', (video_id,))
        video = cursor.fetchone()
        
        if not video:
            return safe_jsonify({
                'success': False,
                'error': 'Video not found'
            }, 404)
        
        # Get analysis results
        cursor.execute('SELECT * FROM analysis_results WHERE video_id = ?', (video_id,))
        analysis = cursor.fetchone()
        
        # Get extracted frames
        cursor.execute('SELECT * FROM extracted_frames WHERE video_id = ? ORDER BY timestamp', (video_id,))
        frames = cursor.fetchall()
        
        # Get audio events and clean any bytes data
        cursor.execute('SELECT * FROM audio_events WHERE video_id = ? ORDER BY timestamp', (video_id,))
        audio_events = cursor.fetchall()
        
        # Clean bytes data from audio_events
        for event in audio_events:
            for key, value in event.items():
                if isinstance(value, bytes):
                    try:
                        event[key] = value.decode('utf-8')
                    except UnicodeDecodeError:
                        event[key] = str(value)[2:-1]  # Remove b' and '
        
        conn.close()
        
        return safe_jsonify({
            'success': True,
            'video': video,
            'analysis': analysis,
            'frames': frames,
            'audio_events': audio_events
        })
        
    except Exception as e:
        return safe_jsonify({
            'success': False,
            'error': str(e)
        }, 500)

@app.route('/api/videos/<int:video_id>/frames/<int:frame_id>', methods=['GET'])
def get_frame_image(video_id, frame_id):
    """Serve extracted frame image"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT frame_path FROM extracted_frames 
            WHERE video_id = ? AND id = ?
        ''', (video_id, frame_id))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return jsonify({
                'success': False,
                'error': 'Frame not found'
            }), 404
        
        frame_path = result[0]
        if not os.path.exists(frame_path):
            return jsonify({
                'success': False,
                'error': 'Frame file not found'
            }), 404
        
        return send_file(frame_path, mimetype='image/jpeg')
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/videos/<int:video_id>/video', methods=['GET'])
def get_video_file(video_id):
    """Serve video file for playback"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT filepath FROM videos WHERE id = ?', (video_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return jsonify({
                'success': False,
                'error': 'Video not found'
            }), 404
        
        video_path = result[0]
        
        if not os.path.exists(video_path):
            return jsonify({
                'success': False,
                'error': 'Video file not found'
            }), 404
        
        return send_file(video_path, mimetype='video/mp4')
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Upload and process new video"""
    try:
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No video file provided'
            }), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'File type not allowed. Supported: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = app.config['UPLOAD_FOLDER'] / unique_filename
        file.save(file_path)
        
        # Add to background processing queue
        global background_processor
        if background_processor:
            background_processor.add_video(str(file_path))
        
        return jsonify({
            'success': True,
            'message': 'Video uploaded successfully and queued for processing',
            'filename': unique_filename
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/process/<path:video_path>', methods=['POST'])
def process_existing_video(video_path):
    """Process existing video file"""
    try:
        if not os.path.exists(video_path):
            return jsonify({
                'success': False,
                'error': 'Video file not found'
            }), 404
        
        # Add to background processing queue
        global background_processor
        if background_processor:
            background_processor.add_video(video_path)
            
            return jsonify({
                'success': True,
                'message': f'Video {video_path} queued for processing'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Background processor not available'
            }), 500
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get overall statistics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Total videos
        cursor.execute('SELECT COUNT(*) FROM videos')
        total_videos = cursor.fetchone()[0]
        
        # Processed videos
        cursor.execute('SELECT COUNT(*) FROM videos WHERE status = "completed"')
        processed_videos = cursor.fetchone()[0]
        
        # Processing videos
        cursor.execute('SELECT COUNT(*) FROM videos WHERE status = "processing"')
        processing_videos = cursor.fetchone()[0]
        
        # Failed videos
        cursor.execute('SELECT COUNT(*) FROM videos WHERE status = "failed"')
        failed_videos = cursor.fetchone()[0]
        
        # Total frames extracted
        cursor.execute('SELECT COUNT(*) FROM extracted_frames')
        total_frames = cursor.fetchone()[0]
        
        # Total audio events
        cursor.execute('SELECT COUNT(*) FROM audio_events')
        total_audio_events = cursor.fetchone()[0]
        
        # Average scores
        cursor.execute('''
            SELECT 
                AVG(anxiety_level) as avg_anxiety,
                AVG(playfulness) as avg_playfulness,
                AVG(aggression) as avg_aggression,
                AVG(confidence) as avg_confidence,
                AVG(energy_level) as avg_energy,
                AVG(stress_indicators) as avg_stress
            FROM analysis_results
        ''')
        avg_scores = cursor.fetchone()
        
        conn.close()
        
        return jsonify({
            'success': True,
            'stats': {
                'total_videos': total_videos,
                'processed_videos': processed_videos,
                'processing_videos': processing_videos,
                'failed_videos': failed_videos,
                'total_frames': total_frames,
                'total_audio_events': total_audio_events,
                'average_scores': {
                    'anxiety_level': round(avg_scores[0] or 0, 1),
                    'playfulness': round(avg_scores[1] or 0, 1),
                    'aggression': round(avg_scores[2] or 0, 1),
                    'confidence': round(avg_scores[3] or 0, 1),
                    'energy_level': round(avg_scores[4] or 0, 1),
                    'stress_indicators': round(avg_scores[5] or 0, 1)
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/dog-profile', methods=['GET'])
def get_dog_profile():
    """Get overall dog behavioral profile across all videos"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all behavioral scores from analysis_results table
        cursor.execute('''
            SELECT anxiety_level, playfulness, aggression, confidence, 
                   energy_level, stress_indicators, overall_assessment, analysis_text
            FROM analysis_results 
            WHERE anxiety_level IS NOT NULL
        ''')
        
        analyses = cursor.fetchall()
        
        if not analyses:
            return safe_jsonify({
                'success': False,
                'error': 'No behavioral analysis data found'
            }, 404)
        
        # Calculate averages
        total_analyses = len(analyses)
        avg_anxiety = sum(row[0] for row in analyses if row[0]) / total_analyses
        avg_playfulness = sum(row[1] for row in analyses if row[1]) / total_analyses
        avg_aggression = sum(row[2] for row in analyses if row[2]) / total_analyses
        avg_confidence = sum(row[3] for row in analyses if row[3]) / total_analyses
        avg_energy = sum(row[4] for row in analyses if row[4]) / total_analyses
        avg_stress = sum(row[5] for row in analyses if row[5]) / total_analyses
        
        # Get recent assessments for breed analysis
        recent_assessments = [row[6] for row in analyses if row[6]]
        analysis_texts = [row[7] for row in analyses if row[7]]
        
        # Simple breed detection based on "white dog" mention and behavioral patterns
        breed_info = analyze_breed_characteristics(avg_confidence, avg_energy, avg_playfulness, analysis_texts)
        
        # Get video count and date range
        cursor.execute('SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM videos')
        video_stats = cursor.fetchone()
        
        conn.close()
        
        return safe_jsonify({
            'success': True,
            'profile': {
                'behavioral_scores': {
                    'anxiety': round(avg_anxiety, 1),
                    'playfulness': round(avg_playfulness, 1),
                    'aggression': round(avg_aggression, 1),
                    'confidence': round(avg_confidence, 1),
                    'energy': round(avg_energy, 1),
                    'stress': round(avg_stress, 1)
                },
                'breed_info': breed_info,
                'statistics': {
                    'total_videos_analyzed': total_analyses,
                    'total_videos': video_stats[0],
                    'analysis_period': {
                        'start': video_stats[1],
                        'end': video_stats[2]
                    }
                },
                'recent_assessments': recent_assessments[-3:] if recent_assessments else []
            }
        })
        
    except Exception as e:
        return safe_jsonify({
            'success': False,
            'error': str(e)
        }, 500)

def analyze_breed_characteristics(confidence, energy, playfulness, analysis_texts):
    """Analyze breed characteristics based on behavioral patterns"""
    
    # Look for breed mentions in analysis texts
    breed_mentions = []
    for text in analysis_texts:
        if text:
            text_lower = text.lower()
            # Common white dog breeds
            if any(breed in text_lower for breed in ['golden', 'retriever', 'labrador', 'lab']):
                breed_mentions.append('Golden Retriever or Labrador')
            elif any(breed in text_lower for breed in ['husky', 'siberian']):
                breed_mentions.append('Siberian Husky')
            elif any(breed in text_lower for breed in ['shepherd', 'german']):
                breed_mentions.append('German Shepherd')
            elif any(breed in text_lower for breed in ['poodle']):
                breed_mentions.append('Poodle')
    
    # Behavioral-based breed analysis for white dogs
    if confidence >= 7 and energy >= 7 and playfulness >= 7:
        likely_breed = "Golden Retriever"
        characteristics = [
            "Highly confident and outgoing personality",
            "High energy levels requiring regular exercise",
            "Very playful and social with other dogs",
            "Excellent family companion breed",
            "Known for friendly, gentle temperament"
        ]
    elif confidence >= 6 and energy >= 8 and playfulness >= 6:
        likely_breed = "Labrador Retriever"
        characteristics = [
            "Confident and energetic working breed",
            "High exercise needs and stamina",
            "Playful and eager to please",
            "Excellent with children and families",
            "Strong retrieving and swimming instincts"
        ]
    elif energy >= 8 and playfulness >= 7:
        likely_breed = "Siberian Husky (White/Light colored)"
        characteristics = [
            "High energy sled dog breed",
            "Very playful and pack-oriented",
            "Requires significant daily exercise",
            "Independent but social personality",
            "Thick double coat, often white markings"
        ]
    elif confidence >= 8 and energy >= 6:
        likely_breed = "German Shepherd (White variety)"
        characteristics = [
            "Highly confident and alert breed",
            "Moderate to high energy levels",
            "Loyal and protective nature",
            "Intelligent working dog breed",
            "White German Shepherds are less common but exist"
        ]
    else:
        likely_breed = "Mixed Breed or White-coated breed"
        characteristics = [
            "Unique personality blend",
            "Behavioral traits suggest friendly temperament",
            "Well-socialized and adaptable",
            "Could be mix of retriever, shepherd, or other breeds",
            "Individual personality more important than breed"
        ]
    
    return {
        'likely_breed': likely_breed,
        'confidence_level': 'Medium' if breed_mentions else 'Low',
        'characteristics': characteristics,
        'breed_notes': f"Analysis based on behavioral patterns from {len(analysis_texts)} video sessions. White coat color noted.",
        'recommendations': [
            "Continue regular socialization with other dogs",
            "Maintain consistent exercise routine based on energy levels",
            "Monitor stress indicators during new situations",
            "Regular grooming for white coat maintenance"
        ]
    }

# -------------------- Background Processing Integration --------------------

def init_background_processor():
    """Initialize background processor"""
    global background_processor
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Warning: OPENROUTER_API_KEY not set. Using fallback analysis.")
    
    background_processor = BackgroundProcessor(api_key)
    background_processor.start()
    print("[API Server] Background processor started")

def cleanup_background_processor():
    """Cleanup background processor"""
    global background_processor
    if background_processor:
        background_processor.stop()
        print("[API Server] Background processor stopped")

# -------------------- Server Startup --------------------

if __name__ == '__main__':
    print("🚀 Starting PackLens API Server...")
    
    # Initialize database
    from video_processor import DatabaseManager
    db = DatabaseManager()
    print("✅ Database initialized")
    
    # Start background processor
    init_background_processor()
    
    try:
        # Start Flask server
        print("🌐 Starting web server on http://localhost:5000")
        print("📊 API endpoints:")
        print("   • GET  /api/health - Health check")
        print("   • GET  /api/videos - Get all videos")
        print("   • GET  /api/videos/<id> - Get video details")
        print("   • POST /api/upload - Upload new video")
        print("   • POST /api/process/<path> - Process existing video")
        print("   • GET  /api/stats - Get statistics")
        print("   • GET  /api/dog-profile - Get overall dog behavioral profile")
        print("=" * 60)
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
    finally:
        cleanup_background_processor()
        print("✅ Server shutdown complete")
