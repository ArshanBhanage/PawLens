#!/usr/bin/env python3
"""
Reprocess Videos via API
Re-analyzes existing videos by calling the API endpoints
"""

import requests
import json
import time

def reprocess_via_api():
    """Reprocess videos by calling API endpoints"""
    
    print("🔄 Reprocessing videos via API...")
    
    base_url = "http://localhost:5000"
    
    try:
        # Get all videos
        response = requests.get(f"{base_url}/api/videos")
        if not response.ok:
            print(f"❌ Failed to get videos: {response.status_code}")
            return
        
        data = response.json()
        if not data['success']:
            print(f"❌ API error: {data.get('error', 'Unknown error')}")
            return
        
        videos = data['videos']
        print(f"📹 Found {len(videos)} videos to reprocess")
        
        for video in videos:
            video_id = video['id']
            filename = video['filename']
            filepath = video['filepath']
            
            print(f"\n🎬 Reprocessing: {filename[:50]}...")
            
            # Call the process endpoint
            try:
                process_response = requests.post(f"{base_url}/api/process/{filepath}")
                
                if process_response.ok:
                    result = process_response.json()
                    if result['success']:
                        print(f"   ✅ Queued for reprocessing!")
                        print(f"   ⏳ Processing in background...")
                    else:
                        print(f"   ❌ Failed to queue: {result.get('error', 'Unknown error')}")
                else:
                    print(f"   ❌ API error: {process_response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print(f"\n✅ All videos queued for reprocessing!")
        print("⏳ Videos are being processed in the background with real AI analysis")
        print("🎯 Check your dashboard in a few minutes to see updated scores!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🔄 This will reprocess all videos with real AI analysis via API")
    print("💰 This will use API credits for each video")
    
    confirm = input("\n❓ Do you want to proceed? (yes/no): ").lower().strip()
    
    if confirm in ['yes', 'y']:
        reprocess_via_api()
    else:
        print("❌ Reprocessing cancelled")
