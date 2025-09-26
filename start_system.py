#!/usr/bin/env python3
"""
PackLens System Startup Script
Starts the complete PackLens system with API server and frontend

Usage:
  export OPENROUTER_API_KEY="your_key_here"
  python start_system.py
"""

import os
import sys
import subprocess
import threading
import time
import signal
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import flask_cors
        print("✅ Flask dependencies available")
    except ImportError:
        print("❌ Flask dependencies missing. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "flask", "flask-cors", "werkzeug"])
    
    # Check if frontend dependencies are installed
    frontend_dir = Path("frontend")
    if frontend_dir.exists():
        node_modules = frontend_dir / "node_modules"
        if not node_modules.exists():
            print("📦 Installing frontend dependencies...")
            subprocess.run(["npm", "install"], cwd=frontend_dir)
        else:
            print("✅ Frontend dependencies available")
    else:
        print("⚠️ Frontend directory not found")

def start_api_server():
    """Start the Flask API server"""
    print("🚀 Starting API server...")
    try:
        subprocess.run([sys.executable, "api_server.py"])
    except KeyboardInterrupt:
        print("\n🛑 API server stopped")

def start_frontend():
    """Start the React frontend"""
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return
    
    print("🌐 Starting frontend...")
    try:
        subprocess.run(["npm", "start"], cwd=frontend_dir)
    except KeyboardInterrupt:
        print("\n🛑 Frontend stopped")

def main():
    print("🐕 PackLens System Startup")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("⚠️ Warning: OPENROUTER_API_KEY not set")
        print("   Set your API key: export OPENROUTER_API_KEY='your_key_here'")
        print("   System will use fallback analysis without API key")
    else:
        print("✅ API key configured")
    
    # Check dependencies
    check_dependencies()
    
    print("\n🎯 Starting system components...")
    print("   • API Server: http://localhost:5000")
    print("   • Frontend: http://localhost:3000")
    print("   • Press Ctrl+C to stop all services")
    print("=" * 50)
    
    # Start API server in background thread
    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()
    
    # Give API server time to start
    time.sleep(3)
    
    # Start frontend (this will block)
    try:
        start_frontend()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down system...")
        print("✅ System stopped")

if __name__ == "__main__":
    main()
