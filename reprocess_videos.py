#!/usr/bin/env python3
"""
Reprocess Videos Script
Re-analyzes existing videos with real AI analysis (Grok-4) instead of fallback analysis
"""

import sqlite3
import os
from video_processor import VideoProcessor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def reprocess_all_videos():
    """Reprocess all videos with real AI analysis"""
    
    print("🔄 Starting video reprocessing with real AI analysis...")
    
    # Check API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in .env file!")
        return
    
    print(f"✅ API Key loaded: {api_key[:20]}...")
    
    # Connect to database with timeout
    db_path = "pawlens_analysis.db"
    if not os.path.exists(db_path):
        print("❌ Database not found!")
        return
    
    # Use timeout and WAL mode to handle concurrent access
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    cursor = conn.cursor()
    
    try:
        # Get all videos
        cursor.execute("SELECT id, filename, filepath FROM videos WHERE status = 'completed'")
        videos = cursor.fetchall()
        
        if not videos:
            print("📭 No completed videos found to reprocess")
            return
        
        print(f"📹 Found {len(videos)} videos to reprocess")
        
        # Initialize video processor with API key
        processor = VideoProcessor(api_key)
        
        for video_id, filename, filepath in videos:
            print(f"\n🎬 Reprocessing: {filename}")
            
            if not os.path.exists(filepath):
                print(f"   ⚠️ Video file not found: {filepath}")
                continue
            
            try:
                # Delete old analysis
                cursor.execute("DELETE FROM analysis_results WHERE video_id = ?", (video_id,))
                cursor.execute("DELETE FROM extracted_frames WHERE video_id = ?", (video_id,))
                cursor.execute("DELETE FROM audio_events WHERE video_id = ?", (video_id,))
                
                # Reprocess with real AI
                result = processor.process_video(filepath)
                
                if result['success']:
                    print(f"   ✅ Reprocessed successfully!")
                    print(f"   📊 New scores: Anxiety: {result.get('anxiety_level', 'N/A')}, "
                          f"Playfulness: {result.get('playfulness', 'N/A')}, "
                          f"Confidence: {result.get('confidence', 'N/A')}")
                else:
                    print(f"   ❌ Reprocessing failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"   ❌ Error reprocessing {filename}: {e}")
        
        # Commit changes
        conn.commit()
        print(f"\n✅ Reprocessing complete!")
        print("🎯 Check your dashboard to see the updated behavioral scores!")
        
    except Exception as e:
        print(f"❌ Error during reprocessing: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    print("⚠️  This will reprocess all videos with real AI analysis")
    print("🔄 Existing fallback analysis will be replaced with Grok-4 analysis")
    print("💰 This will use API credits for each video")
    
    confirm = input("\n❓ Do you want to proceed? (yes/no): ").lower().strip()
    
    if confirm in ['yes', 'y']:
        reprocess_all_videos()
    else:
        print("❌ Reprocessing cancelled")
        print("💡 New videos uploaded will automatically use real AI analysis!")
