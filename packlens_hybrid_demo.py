"""
PackLens Hybrid: Local + Cloud AI for dog behavior analysis
- Fast local vision models (CLIP, BLIP) for real-time analysis
- Optional cloud AI (Grok-4) for detailed assessment
- Easy switching between local and cloud modes
- Optimized for speed and accuracy

Usage:
  # Local mode (fast, no API key needed)
  python packlens_hybrid_demo.py --source video.mp4 --mode local --save demo.mp4
  
  # Cloud mode (detailed analysis, requires API key)
  export OPENROUTER_API_KEY="your_key"
  python packlens_hybrid_demo.py --source video.mp4 --mode cloud --save demo.mp4
  
  # Hybrid mode (local + cloud validation)
  python packlens_hybrid_demo.py --source video.mp4 --mode hybrid --save demo.mp4
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

# Local Vision Model imports
try:
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import CLIPProcessor, CLIPModel
    TRANSFORMERS_OK = True
except ImportError as e:
    print(f"Warning: Transformers not available: {e}")
    print("Install with: pip install transformers torch torchvision")
    TRANSFORMERS_OK = False

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
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    
    # Resize if too large
    max_size = 800
    if max(pil_image.size) > max_size:
        ratio = max_size / max(pil_image.size)
        new_size = tuple(int(dim * ratio) for dim in pil_image.size)
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
    
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=80)
    buffer.seek(0)
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# -------------------- Local Vision Models --------------------

class LocalVisionAnalyzer:
    """Fast local vision models for dog behavior analysis"""
    
    def __init__(self, device="auto"):
        self.device = self._get_device(device)
        self.models_loaded = False
        self.last_analysis_time = 0
        self.analysis_cooldown = 0.5  # 500ms for local models
        self.last_analysis = ""
        
        if TRANSFORMERS_OK:
            self._load_models()
        else:
            print("[Local Vision] Transformers not available - using fallback analysis")
    
    def _get_device(self, device):
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            else:
                return "cpu"
        return device
    
    def _load_models(self):
        """Load local vision models"""
        try:
            print(f"[Local Vision] Loading models on {self.device}...")
            
            # BLIP for image captioning and understanding
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model.to(self.device)
            
            # CLIP for image-text matching
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
            
            self.models_loaded = True
            print("[Local Vision] Models loaded successfully")
            
        except Exception as e:
            print(f"[Local Vision] Error loading models: {e}")
            self.models_loaded = False
    
    def analyze_frame(self, frame, dog_boxes, current_time):
        """Analyze frame using local vision models"""
        if not self.models_loaded or not dog_boxes:
            return ""
        
        # Rate limiting for performance
        if current_time - self.last_analysis_time < self.analysis_cooldown:
            return self.last_analysis
        
        try:
            # Convert frame to PIL
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Resize for faster processing
            pil_image = pil_image.resize((512, 384), Image.Resampling.LANCZOS)
            
            # BLIP analysis for scene understanding
            scene_description = self._get_scene_description(pil_image)
            
            # CLIP analysis for behavioral classification
            behavioral_state = self._classify_behavior(pil_image)
            
            # Combine analyses
            analysis = self._generate_analysis(scene_description, behavioral_state, len(dog_boxes))
            
            self.last_analysis = analysis
            self.last_analysis_time = current_time
            
            return analysis
            
        except Exception as e:
            print(f"[Local Vision] Analysis error: {e}")
            return ""
    
    def _get_scene_description(self, image):
        """Get scene description using BLIP"""
        try:
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50, num_beams=3)
            
            description = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return description.lower()
            
        except Exception as e:
            print(f"[Local Vision] BLIP error: {e}")
            return "dogs in scene"
    
    def _classify_behavior(self, image):
        """Classify dog behavior using CLIP"""
        try:
            # Behavioral categories
            behavior_texts = [
                "calm relaxed dog sitting peacefully",
                "playful dog with happy expression",
                "alert dog with raised ears and focused attention", 
                "aggressive dog with tense body and bared teeth",
                "anxious stressed dog with lowered posture",
                "excited energetic dog with active movement"
            ]
            
            inputs = self.clip_processor(
                text=behavior_texts, 
                images=image, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Get top prediction
            top_idx = probs.argmax().item()
            confidence = probs[0][top_idx].item()
            
            behavior_map = {
                0: ("calm", "SAFE"),
                1: ("playful", "SAFE"), 
                2: ("alert", "CAUTION"),
                3: ("aggressive", "DANGER"),
                4: ("anxious", "CAUTION"),
                5: ("excited", "SAFE")
            }
            
            behavior, alert_level = behavior_map.get(top_idx, ("neutral", "SAFE"))
            
            return {
                "behavior": behavior,
                "alert_level": alert_level,
                "confidence": confidence
            }
            
        except Exception as e:
            print(f"[Local Vision] CLIP error: {e}")
            return {"behavior": "neutral", "alert_level": "SAFE", "confidence": 0.5}
    
    def _generate_analysis(self, scene_description, behavioral_state, num_dogs):
        """Generate structured analysis"""
        behavior = behavioral_state["behavior"]
        alert_level = behavioral_state["alert_level"]
        confidence = behavioral_state["confidence"]
        
        # Generate appropriate action based on behavior
        actions = {
            "aggressive": "Separate dogs immediately; create distance to prevent escalation",
            "anxious": "Monitor closely; provide calm environment and space",
            "alert": "Stay attentive; ready to intervene if situation changes",
            "excited": "Supervise play; ensure safe interaction",
            "playful": "Allow supervised interaction; monitor for overstimulation", 
            "calm": "Continue normal supervision; dogs appear relaxed"
        }
        
        action = actions.get(behavior, "Monitor situation and be ready to intervene")
        
        # Format analysis
        analysis = f"**State**: {behavior.title()} ({confidence:.2f})\n**Alert**: {alert_level}\n**Action**: {action}"
        
        return analysis

# -------------------- Cloud Vision Analyzer (Grok-4) --------------------

class CloudVisionAnalyzer:
    """Cloud-based vision analysis using Grok-4 Fast"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.last_analysis_time = 0
        self.analysis_cooldown = 5.0  # 5 seconds for cloud API
        self.last_analysis = ""
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        if api_key:
            print("[Cloud Vision] Grok-4 initialized successfully")
        else:
            print("[Cloud Vision] No API key provided")
    
    def analyze_frame(self, frame, dog_boxes, current_time):
        """Analyze frame using Grok-4 Fast"""
        if not self.api_key or not dog_boxes:
            return ""
        
        if current_time - self.last_analysis_time < self.analysis_cooldown:
            return self.last_analysis
        
        try:
            image_b64 = encode_image_to_base64(frame)
            
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
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                        ]
                    }
                ],
                "max_tokens": 200,
                "temperature": 0.3
            }
            
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                analysis = result['choices'][0]['message']['content'].strip()
                self.last_analysis = analysis
                self.last_analysis_time = current_time
                return analysis
            else:
                print(f"[Cloud Vision] API Error: {response.status_code}")
                return ""
            
        except Exception as e:
            print(f"[Cloud Vision] Analysis error: {e}")
            return ""

# -------------------- Hybrid Analyzer --------------------

class HybridAnalyzer:
    """Combines local and cloud analysis for optimal performance"""
    
    def __init__(self, mode="local", api_key=None):
        self.mode = mode
        self.local_analyzer = LocalVisionAnalyzer()
        self.cloud_analyzer = CloudVisionAnalyzer(api_key) if api_key else None
        
        print(f"[Hybrid] Mode: {mode}")
        if mode == "cloud" and not api_key:
            print("[Hybrid] Warning: Cloud mode requested but no API key provided, falling back to local")
            self.mode = "local"
    
    def analyze_frame(self, frame, dog_boxes, current_time):
        """Analyze frame based on selected mode"""
        if self.mode == "local":
            return self.local_analyzer.analyze_frame(frame, dog_boxes, current_time)
        
        elif self.mode == "cloud" and self.cloud_analyzer:
            return self.cloud_analyzer.analyze_frame(frame, dog_boxes, current_time)
        
        elif self.mode == "hybrid":
            # Fast local analysis first
            local_result = self.local_analyzer.analyze_frame(frame, dog_boxes, current_time)
            
            # Cloud validation for high-risk situations
            if "DANGER" in local_result or "CAUTION" in local_result:
                if self.cloud_analyzer:
                    cloud_result = self.cloud_analyzer.analyze_frame(frame, dog_boxes, current_time)
                    if cloud_result:
                        return f"LOCAL: {local_result}\n\nCLOUD: {cloud_result}"
            
            return local_result
        
        else:
            return self.local_analyzer.analyze_frame(frame, dog_boxes, current_time)

# -------------------- Audio Analyzer --------------------

class AudioAnalyzer:
    """Simple audio analysis for growling/barking detection"""
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
            
            sr = 16000
            audio = clip.audio.to_soundarray(fps=sr)
            clip.close()
            
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            
            # Simple audio event detection
            frame_len = int(0.05 * sr)
            hop = int(0.025 * sr)
            
            events = []
            for i in range(0, len(audio) - frame_len, hop):
                frame = audio[i:i+frame_len]
                rms = np.sqrt(np.mean(frame**2))
                
                if rms > 0.1:
                    fft = np.fft.rfft(frame)
                    freqs = np.fft.rfftfreq(len(frame), 1/sr)
                    dominant_freq = freqs[np.argmax(np.abs(fft))]
                    
                    t = i / sr
                    if dominant_freq < 500:
                        events.append((t, "growl"))
                    elif 500 <= dominant_freq <= 2000:
                        events.append((t, "bark"))
                    elif dominant_freq > 2000:
                        events.append((t, "whine"))
            
            # Deduplicate
            self.events = []
            last_t = -1
            for t, label in events:
                if t - last_t > 0.5:
                    self.events.append((t, label))
                    last_t = t
                    
        except Exception as e:
            print(f"[Audio] Error: {e}")
            self.events = []

# -------------------- Enhanced Agent --------------------

class EnhancedAgent:
    def __init__(self, cooldown=2.0, max_msgs=12):
        self.cooldown = cooldown
        self.last_spoken = 0.0
        self.msgs = deque(maxlen=max_msgs)
        self.buffer = deque()
        self.audio_alerts = deque(maxlen=10)
        self.last_danger_alert = 0

    def observe_audio(self, t, label):
        self.audio_alerts.append((t, label))
        if label == "growl":
            self.buffer.append("🚨 AUDIO ALERT: Growling detected - potential aggression!")
        elif label == "bark" and len([x for x in self.audio_alerts if x[1] == "bark" and t - x[0] < 2]) > 2:
            self.buffer.append("⚠️ AUDIO: Rapid barking - high excitement/stress")

    def observe_vision_analysis(self, t, analysis, source="AI"):
        if not analysis:
            return
            
        danger_words = ["danger", "aggressive", "attack", "bite", "fight"]
        caution_words = ["caution", "alert", "tense", "anxious", "warning"]
        
        recent_audio = [x[1] for x in self.audio_alerts if t - x[0] < 3]
        audio_warning = "growl" in recent_audio or recent_audio.count("bark") > 2
        
        if any(word in analysis.lower() for word in danger_words) or audio_warning:
            if t - self.last_danger_alert > 3:
                prefix = "🚨 DANGER" if not audio_warning else "🚨 DANGER + AUDIO"
                self.buffer.append(f"{prefix}: {analysis}")
                self.last_danger_alert = t
        elif any(word in analysis.lower() for word in caution_words):
            self.buffer.append(f"⚠️ CAUTION: {analysis}")
        else:
            self.buffer.append(f"✅ SAFE: {analysis}")

    def tick(self, now):
        if self.buffer and (now - self.last_spoken) > self.cooldown:
            msg = self.buffer.popleft()
            self.msgs.appendleft((now, msg))
            self.last_spoken = now
            return msg
        return ""

# -------------------- Simple Tracker --------------------

class SimpleTracker:
    def __init__(self, max_miss=15, dist_thresh=100):
        self.tracks = {}
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
        
        for i,b in enumerate(boxes):
            if i in used: continue
            tid = self.next_id; self.next_id += 1
            self.tracks[tid] = b
            self.miss[tid] = 0
        return self.tracks

# -------------------- Main Demo --------------------

def run(source, conf=0.35, cooldown=2.0, save_path=None, mode="local"):
    if not YOLO_OK:
        raise RuntimeError("Ultralytics YOLO not available. Install with `pip install ultralytics`.")

    # Initialize analyzers
    api_key = os.getenv('OPENROUTER_API_KEY')
    vision_analyzer = HybridAnalyzer(mode=mode, api_key=api_key)
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
    prev_gray = None

    # Layout
    panel_w = 550
    font_h = 16

    frame_idx = 0

    print(f"[Info] Starting PackLens Hybrid processing: {source}")
    print(f"[Info] Video dimensions: {W}x{H}, FPS: {fps:.1f}")
    print(f"[Info] Audio events detected: {len(audio_events)}")
    print(f"[Info] Vision mode: {mode.upper()}")
    print("[Info] Press ESC to quit")

    start_time = time.time()

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

        # Process audio events
        current_time = time.time()
        while audio_events and audio_events[0][0] <= t_sec:
            _, audio_label = audio_events.popleft()
            agent.observe_audio(current_time, audio_label)

        # Vision analysis
        if tracks:
            analysis = vision_analyzer.analyze_frame(frame, list(tracks.values()), current_time)
            if analysis:
                agent.observe_vision_analysis(current_time, analysis, mode.upper())

        agent.tick(current_time)

        # Draw video with overlays
        vis = frame.copy()
        for tid, bb in tracks.items():
            motion_state = "Active" if motion.get(tid, 0) > 0.08 else "Calm"
            label = f"Dog {tid} • {motion_state}"
            draw_label(vis, bb, label)

        # Enhanced side panel
        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        put_text(panel, f"PackLens Hybrid - {mode.upper()} Mode", (12, 28), 0.7, (0,255,255))
        
        # Performance info
        elapsed = time.time() - start_time
        fps_actual = frame_idx / elapsed if elapsed > 0 else 0
        put_text(panel, f"FPS: {fps_actual:.1f} | Dogs: {len(tracks)}", (12, 50), 0.5, (200,200,200))
        
        y = 75
        if agent.msgs:
            for ts, msg in list(agent.msgs)[:int((h-75)//font_h)]:
                # Color coding
                color = (255,255,255)
                if "🚨 DANGER" in msg:
                    color = (0,0,255)    # Red
                elif "⚠️ CAUTION" in msg:
                    color = (0,165,255)  # Orange
                elif "✅ SAFE" in msg:
                    color = (0,255,0)    # Green
                elif "🚨 AUDIO" in msg:
                    color = (255,0,255)  # Magenta

                # Smart text wrapping
                max_chars = 65
                if len(msg) > max_chars:
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
                    
                    for i, line in enumerate(lines[:4]):
                        prefix = "• " if i == 0 else "  "
                        put_text(panel, f"{prefix}{line}", (12, y), 0.38, color)
                        y += 13
                    y += 3
                else:
                    put_text(panel, f"• {msg}", (12, y), 0.38, color)
                    y += font_h
        else:
            put_text(panel, f"(analyzing with {mode} vision...)", (12, y), 0.5, (200,200,200))

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

        cv2.imshow(f"PackLens Hybrid - {mode.upper()} Mode", out)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[Info] Saved output video: {save_path}")
    cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0
    print(f"[Info] Demo completed - Average FPS: {avg_fps:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=str, required=True, help='Path to a video file')
    ap.add_argument('--mode', type=str, choices=['local', 'cloud', 'hybrid'], default='local', 
                    help='Analysis mode: local (fast), cloud (detailed), hybrid (best of both)')
    ap.add_argument('--conf', type=float, default=0.35, help='YOLO confidence threshold')
    ap.add_argument('--cooldown', type=float, default=2.0, help='Seconds between analysis updates')
    ap.add_argument('--save', type=str, default=None, help='Optional path to save output video')
    args = ap.parse_args()
    
    try:
        run(args.source, conf=args.conf, cooldown=args.cooldown, save_path=args.save, mode=args.mode)
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. For cloud mode: Set OPENROUTER_API_KEY environment variable")
        print("2. For local mode: Install transformers and torch")
        print("3. Ensure video file exists and is readable")
        exit(1)
