"""
PackLens + Grok-4 Fast: Advanced AI demo for dog behavior analysis
- Enhanced with Grok-4 Fast (free) via OpenRouter API for detailed behavioral insights
- Analyzes individual frames for comprehensive dog behavior understanding
- Combines YOLO detection + motion analysis + Grok-4 behavioral assessment

Usage:
  export OPENROUTER_API_KEY="your_openrouter_api_key_here"
  python packlens_grok4_demo.py --source video.mp4 --save enhanced_demo.mp4
"""

import argparse
import time
import cv2
import numpy as np
import base64
import os
import requests
import json
from collections import deque
from io import BytesIO
from PIL import Image

# Audio imports
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_OK = True
except ImportError as e:
    print(f"Warning: MoviePy not available: {e}")
    MOVIEPY_OK = False
    VideoFileClip = None

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError as e:
    print(f"Error: Ultralytics YOLO not available: {e}")
    YOLO = None
    YOLO_OK = False

# COCO class ID for dogs
DOG_CLASS = 16

# -------------------- Utils --------------------

def put_text(img, text, org, scale=0.6, color=(255,255,255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_label(frame, bbox, text):
    x1,y1,x2,y2 = bbox
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    label = text
    tw = max(120, int(8 * len(label)))
    cv2.rectangle(frame, (x1, max(0, y1-22)), (min(frame.shape[1]-1, x1+tw), y1), (0,255,0), -1)
    put_text(frame, label, (x1+5, y1-6), 0.5, (0,0,0), 1)

def encode_image_to_base64(image):
    """Convert OpenCV image to base64 string for API"""
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    
    # Resize if too large to save on API costs
    max_size = 800
    if max(pil_image.size) > max_size:
        ratio = max_size / max(pil_image.size)
        new_size = tuple(int(dim * ratio) for dim in pil_image.size)
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Save to bytes
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=80)
    buffer.seek(0)
    
    # Encode to base64
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# -------------------- Grok-4 Fast Analyzer --------------------

class Grok4Analyzer:
    """Uses Grok-4 Fast via OpenRouter to analyze dog behavior in images"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.last_analysis_time = 0
        self.analysis_cooldown = 5.0  # Analyze every 5 seconds
        self.last_analysis = ""
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        if api_key:
            print("[Grok-4] Initialized successfully with OpenRouter API")
        else:
            print("[Grok-4] No API key provided. Set OPENROUTER_API_KEY environment variable")
    
    def analyze_frame(self, frame, dog_boxes, current_time):
        """Analyze a frame with detected dogs using Grok-4 Fast"""
        if not self.api_key or not dog_boxes:
            return ""
        
        # Rate limiting
        if current_time - self.last_analysis_time < self.analysis_cooldown:
            return self.last_analysis
        
        try:
            # Encode frame to base64
            image_b64 = encode_image_to_base64(frame)
            
            # Create prompt for concise behavioral assessment
            prompt = f"""URGENT: Analyze this image with {len(dog_boxes)} dog(s) for SAFETY ALERTS.

Focus ONLY on:
1. **Emotional State**: Calm/Alert/Anxious/Aggressive/Playful
2. **Warning Signs**: Stiff body, raised hackles, direct stare, bared teeth, tense posture
3. **Safety Alert**: Is immediate intervention needed?

Respond in this format:
**State**: [emotional state]
**Alert**: [SAFE/CAUTION/DANGER] 
**Action**: [what owner should do]

Keep under 50 words. Be direct and actionable."""

            # Prepare API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": "x-ai/grok-4-fast:free",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 200,
                "temperature": 0.3
            }
            
            # Call Grok-4 Fast API
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                analysis = result['choices'][0]['message']['content'].strip()
                self.last_analysis = analysis
                self.last_analysis_time = current_time
                
                print(f"[Grok-4] Full Analysis: {analysis}")
                return analysis
            else:
                print(f"[Grok-4] API Error: {response.status_code} - {response.text}")
                return ""
            
        except Exception as e:
            print(f"[Grok-4] Analysis error: {e}")
            return ""

# -------------------- Simple Tracker --------------------

class SimpleTracker:
    def __init__(self, max_miss=15, dist_thresh=100):
        self.tracks = {}   # id -> bbox
        self.miss = {}
        self.next_id = 1
        self.max_miss = max_miss
        self.dist_thresh = dist_thresh

    @staticmethod
    def center(b):
        x1,y1,x2,y2 = b
        return ((x1+x2)//2, (y1+y2)//2)

    def update(self, boxes):
        used = set()
        # assign existing tracks
        for tid, bb in list(self.tracks.items()):
            c = self.center(bb)
            candidates = [(i, np.linalg.norm(np.array(self.center(b))-np.array(c)))
                          for i,b in enumerate(boxes) if i not in used]
            if candidates:
                i, d = min(candidates, key=lambda x: x[1])
                if d < self.dist_thresh:
                    self.tracks[tid] = boxes[i]
                    self.miss[tid] = 0
                    used.add(i)
                else:
                    self.miss[tid] = self.miss.get(tid, 0) + 1
            else:
                self.miss[tid] = self.miss.get(tid, 0) + 1
            if self.miss[tid] > self.max_miss:
                del self.tracks[tid]
                del self.miss[tid]
        # new tracks
        for i,b in enumerate(boxes):
            if i in used: continue
            tid = self.next_id; self.next_id += 1
            self.tracks[tid] = b
            self.miss[tid] = 0
        return self.tracks

# -------------------- Audio Analyzer --------------------

class AudioAnalyzer:
    """Analyze audio for growling, barking, whining"""
    def __init__(self, path):
        self.path = path
        self.events = []
        if MOVIEPY_OK:
            self._process()

    def _process(self):
        try:
            clip = VideoFileClip(self.path)
            if clip.audio is None:
                print("[Audio] No audio track found")
                return
            
            # Extract audio features
            sr = 16000
            audio = clip.audio.to_soundarray(fps=sr)
            clip.close()
            
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            
            # Simple audio event detection
            frame_len = int(0.05 * sr)  # 50ms frames
            hop = int(0.025 * sr)       # 25ms hop
            
            events = []
            for i in range(0, len(audio) - frame_len, hop):
                frame = audio[i:i+frame_len]
                rms = np.sqrt(np.mean(frame**2))
                
                # Simple heuristics for dog sounds
                if rms > 0.1:  # Loud sound
                    # Frequency analysis
                    fft = np.fft.rfft(frame)
                    freqs = np.fft.rfftfreq(len(frame), 1/sr)
                    dominant_freq = freqs[np.argmax(np.abs(fft))]
                    
                    t = i / sr
                    if dominant_freq < 500:  # Low frequency
                        events.append((t, "growl"))
                    elif 500 <= dominant_freq <= 2000:  # Mid frequency
                        events.append((t, "bark"))
                    elif dominant_freq > 2000:  # High frequency
                        events.append((t, "whine"))
            
            # Deduplicate events
            self.events = []
            last_t = -1
            for t, label in events:
                if t - last_t > 0.5:  # 500ms gap
                    self.events.append((t, label))
                    last_t = t
                    
        except Exception as e:
            print(f"[Audio] Error: {e}")
            self.events = []

# -------------------- Enhanced Agent --------------------

class EnhancedAgent:
    def __init__(self, cooldown=3.0, max_msgs=12):
        self.cooldown = cooldown
        self.last_spoken = 0.0
        self.msgs = deque(maxlen=max_msgs)
        self.buffer = deque()
        self.audio_alerts = deque(maxlen=10)  # Recent audio events
        self.last_danger_alert = 0

    def observe_audio(self, t, label):
        """Process audio events for alerts"""
        self.audio_alerts.append((t, label))
        if label == "growl":
            self.buffer.append("🚨 AUDIO ALERT: Growling detected - potential aggression!")
        elif label == "bark" and len([x for x in self.audio_alerts if x[1] == "bark" and t - x[0] < 2]) > 2:
            self.buffer.append("⚠️ AUDIO: Rapid barking - high excitement/stress")

    def observe_grok_analysis(self, t, analysis, audio_context=""):
        """Add Grok-4 behavioral analysis with audio context"""
        if not analysis:
            return
            
        # Check for danger signals
        danger_words = ["danger", "aggressive", "attack", "bite", "fight"]
        caution_words = ["caution", "alert", "tense", "anxious", "warning"]
        
        # Recent audio events
        recent_audio = [x[1] for x in self.audio_alerts if t - x[0] < 3]
        audio_warning = "growl" in recent_audio or recent_audio.count("bark") > 2
        
        if any(word in analysis.lower() for word in danger_words) or audio_warning:
            if t - self.last_danger_alert > 5:  # Don't spam alerts
                if audio_warning:
                    self.buffer.append(f"🚨 DANGER: {analysis} + AUDIO WARNING!")
                else:
                    self.buffer.append(f"🚨 DANGER: {analysis}")
                self.last_danger_alert = t
        elif any(word in analysis.lower() for word in caution_words):
            self.buffer.append(f"⚠️ CAUTION: {analysis}")
        else:
            self.buffer.append(f"✅ SAFE: {analysis}")

    def observe(self, t, dog_id, state, context, confidence=0.7):
        # Basic observation (simplified for focus on alerts)
        pass

    def observe_interaction(self, t, a, b, label, tip):
        self.buffer.append(self._interaction_template(a, b, label, tip))

    @staticmethod
    def _template(tid, state, context):
        if state == "Active" and context == "Social":
            return f"Dog {tid} looks playful near another dog."
        if state == "Active" and context == "Neutral":
            return f"Dog {tid} is energetic."
        if state == "Calm" and context == "Social":
            return f"Dog {tid} is calmly greeting."
        return f"Dog {tid} is calm."

    @staticmethod
    def _interaction_template(a, b, label, tip):
        icon = {"POSITIVE":"✅", "NEUTRAL":"ℹ️", "ANXIOUS":"⚠️"}.get(label, "ℹ️")
        return f"{icon} Interaction {a} ↔ {b}: {label.title()} — {tip}"

    def tick(self, now):
        if self.buffer and (now - self.last_spoken) > self.cooldown:
            msg = self.buffer.popleft()
            self.msgs.appendleft((now, msg))
            self.last_spoken = now
            return msg
        return ""

# -------------------- Interaction Analyzer --------------------

class InteractionAnalyzer:
    """Simple interaction analysis based on proximity and motion"""
    def __init__(self, w, h, fps, near_ratio=0.35):
        self.w = w; self.h = h; self.fps = fps
        self.near_thresh = near_ratio * min(w, h)
        self.pairs = {}

    def update(self, now, centers, motion):
        tids = list(centers.keys())
        results = []
        for i in range(len(tids)):
            for j in range(i+1, len(tids)):
                a, b = sorted((tids[i], tids[j]))
                ca, cb = centers[a], centers[b]
                d = float(np.linalg.norm(np.array(ca)-np.array(cb)))
                
                if d < self.near_thresh:
                    m_mean = 0.5 * (motion.get(a, 0) + motion.get(b, 0))
                    
                    if m_mean > 0.12:
                        label = 'POSITIVE'
                        tip = 'Active play - monitor and allow space'
                    elif m_mean < 0.06:
                        label = 'NEUTRAL'
                        tip = 'Calm interaction - let them sniff'
                    else:
                        label = 'NEUTRAL'
                        tip = 'Normal interaction'
                    
                    results.append((a, b, label, tip))
        return results

# -------------------- Main Demo --------------------

def run(source, conf=0.35, cooldown=4.0, save_path=None):
    if not YOLO_OK:
        raise RuntimeError("Ultralytics YOLO not available. Install with `pip install ultralytics`.")

    # Initialize Grok-4 and Audio Analysis
    api_key = os.getenv('OPENROUTER_API_KEY')
    grok4 = Grok4Analyzer(api_key)
    audio_analyzer = AudioAnalyzer(source)
    audio_events = deque(audio_analyzer.events)

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    # Get video dimensions
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Empty video.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    H, W = first.shape[:2]

    # Initialize components
    writer = None
    agent = EnhancedAgent(cooldown=cooldown)
    tracker = SimpleTracker(max_miss=10)
    interact = InteractionAnalyzer(W, H, fps)
    prev_gray = None

    # Layout
    panel_w = 520  # Wide panel for Grok-4 insights
    font_h = 18

    frame_idx = 0

    print(f"[Info] Starting Grok-4 enhanced video processing: {source}")
    print(f"[Info] Video dimensions: {W}x{H}, FPS: {fps:.1f}")
    print(f"[Info] Audio events detected: {len(audio_events)}")
    print(f"[Info] Grok-4 Fast: {'Enabled' if grok4.api_key else 'Disabled'}")
    print("[Info] Press ESC to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        t_sec = frame_idx / fps
        frame_idx += 1

        # Detect dogs
        res = model(frame, verbose=False, conf=conf)[0]
        boxes = []
        for b in res.boxes:
            cls = int(b.cls[0])
            if cls != DOG_CLASS: continue
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(w-1,x2), min(h-1,y2)
            boxes.append((x1,y1,x2,y2))

        tracks = tracker.update(boxes)

        # Motion analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = {}
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            for tid, (x1,y1,x2,y2) in tracks.items():
                roi = diff[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                m = float(np.mean(roi)/255.0) if roi.size else 0.0
                motion[tid] = m
        prev_gray = gray

        # Process audio events in sync with video
        current_time = time.time()
        while audio_events and audio_events[0][0] <= t_sec:
            _, audio_label = audio_events.popleft()
            agent.observe_audio(current_time, audio_label)

        # Grok-4 analysis (periodic)
        if tracks:  # Only analyze when dogs are detected
            grok_analysis = grok4.analyze_frame(frame, list(tracks.values()), current_time)
            if grok_analysis:
                agent.observe_grok_analysis(current_time, grok_analysis)

        # Basic behavioral classification
        centers = {tid: ((bb[0]+bb[2])//2, (bb[1]+bb[3])//2) for tid, bb in tracks.items()}
        
        # Agent observations
        now = time.time()
        for tid, bb in tracks.items():
            state = "Active" if motion.get(tid, 0.0) > 0.08 else "Calm"
            context = "Social" if len(tracks) > 1 else "Neutral"
            agent.observe(now, tid, state, context, 0.7)

        # Interaction analysis
        for (a,b,label,tip) in interact.update(now, centers, motion):
            agent.observe_interaction(now, a, b, label, tip)

        agent.tick(now)

        # Draw video with overlays
        vis = frame.copy()
        for tid, bb in tracks.items():
            label = f"Dog {tid} • {'Active' if motion.get(tid,0)>0.08 else 'Calm'}"
            draw_label(vis, bb, label)

        # Enhanced side panel
        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        put_text(panel, "PackLens + Grok-4 Fast AI", (12, 28), 0.8, (0,255,255))
        put_text(panel, "Advanced Behavioral Analysis", (12, 50), 0.5, (200,200,200))
        
        y = 80
        if agent.msgs:
            for ts, msg in list(agent.msgs)[: int((h-80)//font_h) ]:
                # Color code messages
                color = (255,255,255)  # Default white
                if "🚨 DANGER" in msg:
                    color = (0,0,255)    # Red for danger
                elif "⚠️ CAUTION" in msg:
                    color = (0,165,255)  # Orange for caution
                elif "✅ SAFE" in msg:
                    color = (0,255,0)    # Green for safe
                elif "🚨 AUDIO" in msg:
                    color = (255,0,255)  # Magenta for audio alerts
                
                # Smart text wrapping for full visibility
                max_chars = 60  # Characters per line
                if len(msg) > max_chars:
                    # Split into multiple lines
                    words = msg.split()
                    lines = []
                    current_line = ""
                    
                    for word in words:
                        if len(current_line + " " + word) <= max_chars:
                            current_line += (" " + word) if current_line else word
                        else:
                            if current_line:
                                lines.append(current_line)
                            current_line = word
                    
                    if current_line:
                        lines.append(current_line)
                    
                    # Display all lines
                    for i, line in enumerate(lines[:4]):  # Max 4 lines per message
                        prefix = "• " if i == 0 else "  "
                        put_text(panel, f"{prefix}{line}", (12, y), 0.4, color)
                        y += 14
                    y += 4  # Extra space between messages
                else:
                    put_text(panel, f"• {msg}", (12, y), 0.4, color)
                    y += font_h
        else:
            put_text(panel, "(analyzing with Grok-4...)", (12, y), 0.5, (200,200,200))

        # Concat side-by-side
        out = np.zeros((h, w+panel_w, 3), dtype=np.uint8)
        out[:, :w] = vis
        out[:, w:] = panel

        # Writer setup
        if save_path and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_path, fourcc, fps, (w+panel_w, h))
        if writer is not None:
            writer.write(out)

        cv2.imshow("PackLens + Grok-4 Fast - AI Dog Behavior Analysis", out)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[Info] Saved Grok-4 enhanced output video: {save_path}")
    cv2.destroyAllWindows()
    print("[Info] Grok-4 enhanced demo completed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=str, required=True, help='Path to a video file')
    ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--cooldown', type=float, default=4.0)
    ap.add_argument('--save', type=str, default=None, help='Optional path to save the enhanced output (mp4)')
    args = ap.parse_args()
    
    try:
        run(args.source, conf=args.conf, cooldown=args.cooldown, save_path=args.save)
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Set OPENROUTER_API_KEY environment variable for Grok-4 analysis")
        print("2. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("3. Check that the video file exists and is readable")
        exit(1)
