# PackLens

🐕 **Agentic multimodal dog-interaction assistant: video+audio → Positive/Neutral/Anxious + tip**

## 🚀 Hybrid AI System: Local + Cloud

**Fast Local Vision Models** + **Optional Cloud AI** for optimal performance:
- **Local Mode**: CLIP + BLIP models (500ms response, no API needed)
- **Cloud Mode**: Grok-4 Fast via OpenRouter (detailed analysis, free)
- **Hybrid Mode**: Local speed + Cloud validation for critical situations

## Quickstart

1. Create virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Set up free Grok-4 API access:
```bash
export OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

## Run Examples

### ⚡ Local Mode (FASTEST - No API Key Needed)
```bash
# Fast local analysis with CLIP + BLIP models
python packlens_hybrid_demo.py --source walk.mp4 --mode local --save local_demo.mp4
```

### 🔥 Cloud Mode (Most Detailed)
```bash
# Set up OpenRouter API key first
export OPENROUTER_API_KEY="sk-or-v1-your_key_here"

# Detailed cloud analysis with Grok-4 Fast
python packlens_hybrid_demo.py --source walk.mp4 --mode cloud --save cloud_demo.mp4
```

### 🚀 Hybrid Mode (Best of Both)
```bash
# Local speed + cloud validation for critical situations
export OPENROUTER_API_KEY="sk-or-v1-your_key_here"
python packlens_hybrid_demo.py --source walk.mp4 --mode hybrid --save hybrid_demo.mp4
```

### 📊 Legacy Grok-4 Only Demo
```bash
python packlens_grok4_demo.py --source walk.mp4 --save grok4_demo.mp4
```

## Features

### ⚡ Hybrid Demo (`packlens_hybrid_demo.py`) - **RECOMMENDED**
- ✅ **YOLO dog detection and tracking**
- ✅ **Motion and proximity analysis**  
- ✅ **Real-time behavioral classification**
- 🚀 **Local Vision Models (CLIP + BLIP)**
  - 500ms response time
  - No API key required
  - Runs offline
- 🔥 **Optional Cloud Enhancement (Grok-4)**
  - Detailed behavioral analysis
  - Free tier available
- 🎯 **Smart Mode Switching**
  - Local: Fast everyday analysis
  - Cloud: Detailed assessment when needed
  - Hybrid: Best of both worlds

### 📊 Legacy Grok-4 Demo (`packlens_grok4_demo.py`)
- ✅ **Cloud-only analysis**
- 🔥 **Grok-4 Fast vision analysis**
- 🔥 **Detailed behavioral insights**

### 🤖 What Grok-4 Analyzes:
1. **Body Language & Posture**: Ears, tail, stance analysis
2. **Emotional State**: Calm, excited, anxious, aggressive, playful, alert
3. **Social Dynamics**: Multi-dog interaction assessment
4. **Warning Signs**: Early detection of concerning behaviors
5. **Safety Recommendations**: Actionable advice for dog owners

## API Setup

1. **Get Free API Key**: Visit [OpenRouter.ai](https://openrouter.ai/)
2. **Set Environment Variable**: `export OPENROUTER_API_KEY="your_key"`
3. **Run Enhanced Demo**: Uses Grok-4 Fast (completely free tier)

## Troubleshooting

- **For Grok-4**: Get free API key at https://openrouter.ai/
- **Without API key**: Demo falls back to basic behavioral analysis
- **Rate Limits**: Grok-4 analysis runs every 5 seconds to optimize API usage
- **Image Optimization**: Frames are resized to 800px max to reduce API costs

## Example Output

The enhanced demo displays:
- **Left Panel**: Video with dog detection and tracking
- **Right Panel**: Live AI insights including:
  - 🤖 **AI Alert**: Warning behaviors detected by Grok-4
  - 🤖 **AI Insight**: Positive behaviors and recommendations
  - ✅ **Positive Interactions**: Safe play behaviors
  - ⚠️ **Warnings**: Concerning behavioral patterns

Perfect for dog owners, trainers, and anyone wanting to understand dog behavior in real-time!
