# 🐕 PackLens Complete System

**AI-Powered Dog Behavior Analysis with Web Dashboard**

A comprehensive system that processes dog videos, extracts behavioral insights using Grok-4 LLM, and presents results through an intuitive web dashboard.

## 🎯 System Overview

### **Complete Workflow:**
```
Video Upload → Frame/Audio Extraction → Grok-4 Analysis → Database Storage → Web Dashboard
```

### **Key Features:**
- 🎬 **Video Processing**: Automatic frame and audio extraction
- 🧠 **LLM Analysis**: Grok-4 powered behavioral assessment
- 📊 **Web Dashboard**: Intuitive React frontend
- 🔄 **Background Processing**: Immediate video processing
- 💾 **Database Storage**: Persistent analysis results
- 📈 **Analytics**: Behavioral trends and insights

## 🚀 Quick Start

### **1. Install Dependencies**
```bash
# Python dependencies
pip install -r requirements.txt

# Frontend dependencies (requires Node.js)
cd frontend
npm install
cd ..
```

### **2. Set API Key**
```bash
export OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

### **3. Start Complete System**
```bash
python start_system.py
```

### **4. Access Dashboard**
- **Frontend**: http://localhost:3000
- **API**: http://localhost:5000

## 📁 System Components

### **Backend Components:**

#### **1. Video Processor (`video_processor.py`)**
- Extracts key activity frames from videos
- Analyzes audio for barks, growls, and other sounds
- Integrates with Grok-4 for behavioral analysis
- Stores results in SQLite database

#### **2. API Server (`api_server.py`)**
- Flask REST API for frontend communication
- Handles video uploads and processing requests
- Serves analysis results and statistics
- Background processing integration

#### **3. Database Schema**
- **Videos**: Metadata and processing status
- **Analysis Results**: Behavioral scores and assessments
- **Extracted Frames**: Key activity frames with timestamps
- **Audio Events**: Detected sounds and their intensity

### **Frontend Components:**

#### **1. React Dashboard (`frontend/`)**
- **Video Grid**: Overview of all processed videos
- **Upload Area**: Drag & drop video upload
- **Video Details**: Detailed behavioral analysis
- **Statistics**: System-wide analytics and trends

## 🎨 Dashboard Features

### **Main Dashboard:**
- **Stats Overview**: Total videos, frames processed, audio events
- **Upload Area**: Drag & drop new videos for analysis
- **Video Grid**: All videos with analysis previews
- **Real-time Updates**: Processing status updates

### **Video Details Page:**
- **Behavioral Scores**: Anxiety, playfulness, confidence, etc.
- **Radar Chart**: Visual behavioral profile
- **Overall Assessment**: LLM-generated summary
- **Recommendations**: Actionable advice for dog owners
- **Technical Details**: Frames, audio events, processing info

## 🧠 LLM Analysis

### **Grok-4 Integration:**
- **Input**: Key video frames + audio events + metadata
- **Analysis**: Expert-level behavioral assessment
- **Output**: Structured scores and recommendations

### **Behavioral Traits Analyzed:**
1. **Anxiety Level** (0-10): Stress indicators, panting, trembling
2. **Playfulness** (0-10): Play behaviors, bouncy movements
3. **Aggression** (0-10): Stiff body, raised hackles, direct stare
4. **Confidence** (0-10): Approach behavior, posture
5. **Energy Level** (0-10): Movement intensity, alertness
6. **Stress Indicators** (0-10): Displacement behaviors, lip licking

### **Owner-Focused Output:**
- **Overall Assessment**: Plain-language behavior summary
- **Key Moments**: Most important behavioral observations
- **Recommendations**: Specific, actionable advice
- **Suggestions**: Training and socialization guidance

## 🔧 API Endpoints

### **Core Endpoints:**
- `GET /api/health` - Health check
- `GET /api/videos` - Get all videos with analysis
- `GET /api/videos/<id>` - Get detailed video information
- `POST /api/upload` - Upload new video for processing
- `GET /api/stats` - System statistics and averages

### **Example API Response:**
```json
{
  "success": true,
  "videos": [
    {
      "id": 1,
      "filename": "dog_park_session.mp4",
      "duration": 120.5,
      "status": "completed",
      "analysis": {
        "anxiety_level": 3,
        "playfulness": 8,
        "confidence": 7,
        "overall_assessment": "Dog shows confident, playful behavior...",
        "suggestions": "Continue park socialization sessions..."
      }
    }
  ]
}
```

## 📊 Database Schema

### **Videos Table:**
```sql
CREATE TABLE videos (
    id INTEGER PRIMARY KEY,
    filename TEXT UNIQUE,
    filepath TEXT,
    file_hash TEXT UNIQUE,
    duration REAL,
    fps REAL,
    width INTEGER,
    height INTEGER,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP,
    processed_at TIMESTAMP
);
```

### **Analysis Results Table:**
```sql
CREATE TABLE analysis_results (
    id INTEGER PRIMARY KEY,
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
    FOREIGN KEY (video_id) REFERENCES videos (id)
);
```

## 🎯 Usage Examples

### **1. Process Single Video:**
```bash
python video_processor.py --video "dog_video.mp4"
```

### **2. Start API Server Only:**
```bash
python api_server.py
```

### **3. Upload Video via API:**
```bash
curl -X POST -F "video=@dog_video.mp4" http://localhost:5000/api/upload
```

### **4. Get Analysis Results:**
```bash
curl http://localhost:5000/api/videos
```

## 🔄 Processing Flow

### **1. Video Upload:**
- User uploads video via web interface
- File saved to uploads directory
- Added to background processing queue

### **2. Frame Extraction:**
- YOLO detects dogs in frames
- Motion analysis identifies key activity moments
- Top 10 frames selected based on activity score

### **3. Audio Analysis:**
- Audio extracted using MoviePy + librosa
- Segments analyzed for barks, growls, vocalizations
- Events timestamped and intensity measured

### **4. LLM Analysis:**
- Key frames + audio events sent to Grok-4
- Expert behavioral analysis generated
- Structured response parsed and stored

### **5. Dashboard Update:**
- Results immediately available in web interface
- Real-time status updates during processing
- Detailed analysis accessible via video details page

## 🛠️ Technical Requirements

### **Python Dependencies:**
- `ultralytics` - YOLO dog detection
- `opencv-python` - Video processing
- `moviepy` - Audio extraction
- `librosa` - Audio analysis
- `flask` - API server
- `requests` - Grok-4 API calls
- `sqlite3` - Database (built-in)

### **Frontend Dependencies:**
- `react` - UI framework
- `axios` - API communication
- `recharts` - Data visualization
- `tailwindcss` - Styling
- `react-dropzone` - File upload

### **System Requirements:**
- Python 3.8+
- Node.js 16+ (for frontend)
- 4GB+ RAM (for video processing)
- GPU optional (improves YOLO performance)

## 🎨 UI Screenshots

### **Main Dashboard:**
- Clean, modern interface with video grid
- Upload area with drag & drop functionality
- Real-time statistics and processing status

### **Video Details:**
- Comprehensive behavioral analysis display
- Interactive radar chart for behavioral profile
- Detailed recommendations and suggestions

## 🔮 Advanced Features

### **Background Processing:**
- Videos processed immediately upon upload
- Queue system handles multiple videos
- Real-time status updates in dashboard

### **Smart Frame Selection:**
- Motion detection identifies activity peaks
- YOLO ensures dog presence in selected frames
- Activity scoring prioritizes interesting moments

### **Comprehensive Audio Analysis:**
- Multi-segment audio processing
- Bark/growl detection with intensity measurement
- Context integration with visual analysis

### **Owner-Focused Design:**
- Plain-language behavioral assessments
- Actionable recommendations and suggestions
- Visual charts for easy understanding

## 🚀 Deployment

### **Development:**
```bash
python start_system.py
```

### **Production:**
- Use production WSGI server (gunicorn)
- Configure reverse proxy (nginx)
- Set up proper database (PostgreSQL)
- Implement authentication if needed

## 🎉 Success Metrics

The system successfully:
- ✅ Processes videos with frame and audio extraction
- ✅ Integrates Grok-4 LLM for behavioral analysis
- ✅ Stores results in structured database
- ✅ Provides intuitive web dashboard
- ✅ Handles background processing
- ✅ Delivers owner-focused insights

**Perfect for dog owners, trainers, and researchers who want professional-grade behavioral analysis with an easy-to-use interface!** 🐕📊✨
