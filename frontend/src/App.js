import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import axios from 'axios';
import './index.css';

// Components
import Header from './components/Header';
import VideoGrid from './components/VideoGrid';
import VideoDetails from './components/VideoDetails';
import UploadArea from './components/UploadArea';
import StatsOverview from './components/StatsOverview';
import LoadingSpinner from './components/LoadingSpinner';
import DogProfile from './components/DogProfile';

function App() {
  const [videos, setVideos] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch videos and stats
  const fetchData = async () => {
    try {
      setLoading(true);
      const [videosResponse, statsResponse] = await Promise.all([
        axios.get('/api/videos'),
        axios.get('/api/stats')
      ]);

      if (videosResponse.data.success) {
        setVideos(videosResponse.data.videos);
      }

      if (statsResponse.data.success) {
        setStats(statsResponse.data.stats);
      }

      setError(null);
    } catch (err) {
      console.error('Error fetching data:', err);
      setError('Failed to load data. Please check if the API server is running.');
    } finally {
      setLoading(false);
    }
  };

  // Initial data fetch
  useEffect(() => {
    fetchData();
    
    // Auto-refresh every 10 seconds to show processing updates
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  // Handle video upload
  const handleVideoUpload = async (file) => {
    try {
      const formData = new FormData();
      formData.append('video', file);

      const response = await axios.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.success) {
        // Refresh data to show new video
        setTimeout(fetchData, 1000);
        return { success: true, message: response.data.message };
      } else {
        return { success: false, error: response.data.error };
      }
    } catch (err) {
      console.error('Upload error:', err);
      return { 
        success: false, 
        error: err.response?.data?.error || 'Upload failed' 
      };
    }
  };

  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Header stats={stats} />
        
        <main className="container mx-auto px-4 py-8">
          {loading && videos.length === 0 ? (
            <div className="flex items-center justify-center h-96">
              <LoadingSpinner />
            </div>
          ) : error ? (
            <div className="card p-8 text-center">
              <div className="text-red-500 text-6xl mb-4">⚠️</div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Connection Error</h2>
              <p className="text-gray-600 mb-4">{error}</p>
              <button 
                onClick={fetchData}
                className="btn-primary"
              >
                Try Again
              </button>
            </div>
          ) : (
            <Routes>
              <Route 
                path="/" 
                element={
                  <div className="space-y-8">
                    <DogProfile />
                    <StatsOverview stats={stats} />
                    <UploadArea onUpload={handleVideoUpload} />
                    <VideoGrid videos={videos} />
                  </div>
                } 
              />
              <Route 
                path="/video/:id" 
                element={<VideoDetails />} 
              />
              <Route path="/profile" element={<DogProfile />} />
            </Routes>
          )}
        </main>
      </div>
    </Router>
  );
}

export default App;
