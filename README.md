# PackLens

🐕 **Agentic multimodal dog-interaction assistant: video+audio → Positive/Neutral/Anxious + tip**

## 🚀 Enhanced with Grok-4 Fast (FREE)

Advanced AI-powered behavioral analysis using Grok-4 Fast via OpenRouter API for detailed dog behavior insights - completely free!

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

### 🔥 Grok-4 Enhanced Demo (FREE)
```bash
# Set up OpenRouter API key first
export OPENROUTER_API_KEY="sk-or-v1-your_key_here"

# Run enhanced analysis with Grok-4 Fast
python packlens_grok4_demo.py --source walk.mp4 --save grok4_demo.mp4
```

## Features

### 🚀 Grok-4 Enhanced Demo (`packlens_grok4_demo.py`)
- ✅ **YOLO dog detection**
- ✅ **Motion and proximity analysis**  
- ✅ **Real-time behavioral classification**
- 🔥 **Grok-4 Fast vision analysis (FREE)**
- 🔥 **Advanced behavioral insights**
- 🔥 **Detailed body language assessment**
- 🔥 **AI-powered safety recommendations**
- 🔥 **Enhanced visual interface**

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
