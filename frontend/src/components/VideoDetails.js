import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import axios from 'axios';
import { ArrowLeft, Clock, Volume2, Image, AlertTriangle, CheckCircle } from 'lucide-react';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer } from 'recharts';
import LoadingSpinner from './LoadingSpinner';

const VideoDetails = () => {
  const { id } = useParams();
  const [videoData, setVideoData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchVideoDetails = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`/api/videos/${id}`);
        
        if (response.data.success) {
          setVideoData(response.data);
        } else {
          setError(response.data.error || 'Failed to load video details');
        }
      } catch (err) {
        console.error('Error fetching video details:', err);
        setError('Failed to load video details');
      } finally {
        setLoading(false);
      }
    };

    fetchVideoDetails();
  }, [id]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <LoadingSpinner />
      </div>
    );
  }

  if (error) {
    return (
      <div className="card p-8 text-center">
        <AlertTriangle className="h-12 w-12 text-danger-500 mx-auto mb-4" />
        <h2 className="text-xl font-semibold text-gray-900 mb-2">Error Loading Video</h2>
        <p className="text-gray-600 mb-4">{error}</p>
            <Link to="/" className="btn-primary">
              Back to Videos
            </Link>
      </div>
    );
  }

  const { video, analysis, frames, audio_events } = videoData;

  // Prepare radar chart data
  const radarData = analysis ? [
    { trait: 'Anxiety', score: 10 - analysis.anxiety_level }, // Invert anxiety (lower is better)
    { trait: 'Playfulness', score: analysis.playfulness },
    { trait: 'Confidence', score: analysis.confidence },
    { trait: 'Energy', score: analysis.energy_level },
    { trait: 'Calmness', score: 10 - analysis.stress_indicators }, // Invert stress
  ] : [];

  const formatDuration = (seconds) => {
    if (!seconds) return 'Unknown';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getScoreColor = (score, isInverted = false) => {
    const effectiveScore = isInverted ? 10 - score : score;
    if (effectiveScore <= 3) return 'text-red-600';
    if (effectiveScore <= 6) return 'text-yellow-600';
    return 'text-green-600';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center space-x-4">
        <Link to="/" className="btn-secondary">
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Videos
        </Link>
        <div>
          <h1 className="text-2xl font-bold text-gray-900">{video.filename}</h1>
          <p className="text-gray-600">
            {formatDuration(video.duration)} • {video.width}x{video.height} • {video.fps} FPS
          </p>
        </div>
      </div>

      {/* Status */}
      <div className="flex items-center space-x-2">
        {video.status === 'completed' && (
          <>
            <CheckCircle className="h-5 w-5 text-green-600" />
            <span className="text-green-600 font-medium">Analysis Complete</span>
          </>
        )}
        {video.status === 'processing' && (
          <>
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
            <span className="text-blue-600 font-medium">Processing...</span>
          </>
        )}
        {video.status === 'failed' && (
          <>
            <AlertTriangle className="h-5 w-5 text-red-600" />
            <span className="text-red-600 font-medium">Analysis Failed</span>
          </>
        )}
      </div>

      {analysis && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Behavioral Scores */}
          <div className="card p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Behavioral Assessment</h2>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-gray-700">Anxiety Level</span>
                <span className={`font-semibold ${getScoreColor(analysis.anxiety_level, true)}`}>
                  {analysis.anxiety_level}/10
                </span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-gray-700">Playfulness</span>
                <span className={`font-semibold ${getScoreColor(analysis.playfulness)}`}>
                  {analysis.playfulness}/10
                </span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-gray-700">Aggression</span>
                <span className={`font-semibold ${getScoreColor(analysis.aggression, true)}`}>
                  {analysis.aggression}/10
                </span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-gray-700">Confidence</span>
                <span className={`font-semibold ${getScoreColor(analysis.confidence)}`}>
                  {analysis.confidence}/10
                </span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-gray-700">Energy Level</span>
                <span className={`font-semibold ${getScoreColor(analysis.energy_level)}`}>
                  {analysis.energy_level}/10
                </span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-gray-700">Stress Indicators</span>
                <span className={`font-semibold ${getScoreColor(analysis.stress_indicators, true)}`}>
                  {analysis.stress_indicators}/10
                </span>
              </div>
            </div>
          </div>

          {/* Radar Chart */}
          <div className="card p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Behavioral Profile</h2>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="trait" tick={{ fontSize: 12 }} />
                  <PolarRadiusAxis 
                    angle={90} 
                    domain={[0, 10]} 
                    tick={{ fontSize: 10 }}
                  />
                  <Radar
                    name="Score"
                    dataKey="score"
                    stroke="#3b82f6"
                    fill="#3b82f6"
                    fillOpacity={0.2}
                    strokeWidth={2}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {analysis && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Overall Assessment */}
          <div className="card p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Overall Assessment</h2>
            <p className="text-gray-700 leading-relaxed">
              {analysis.overall_assessment || 'No assessment available.'}
            </p>
          </div>

          {/* Suggestions */}
          <div className="card p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Recommendations</h2>
            <div className="text-gray-700 leading-relaxed">
              {analysis.suggestions ? (
                <div dangerouslySetInnerHTML={{ 
                  __html: analysis.suggestions.replace(/\n/g, '<br />') 
                }} />
              ) : (
                'No specific recommendations available.'
              )}
            </div>
          </div>
        </div>
      )}

      {analysis && analysis.key_moments && (
        <div className="card p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Key Moments</h2>
          <p className="text-gray-700 leading-relaxed">
            {analysis.key_moments}
          </p>
        </div>
      )}

      {/* Technical Details */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Extracted Frames */}
        <div className="card p-6">
          <div className="flex items-center space-x-2 mb-4">
            <Image className="h-5 w-5 text-gray-600" />
            <h2 className="text-lg font-semibold text-gray-900">Extracted Frames</h2>
          </div>
          
          {frames && frames.length > 0 ? (
            <div className="space-y-3">
              <p className="text-sm text-gray-600 mb-3">
                {frames.length} key frames analyzed
              </p>
              <div className="grid grid-cols-2 gap-2">
                {frames.slice(0, 4).map((frame, index) => (
                  <div key={frame.id} className="bg-gray-100 rounded p-2 text-xs">
                    <div className="font-medium">Frame {index + 1}</div>
                    <div className="text-gray-600">
                      {frame.timestamp.toFixed(1)}s • {frame.dogs_detected} dogs
                    </div>
                    <div className="text-gray-500">
                      Activity: {(frame.activity_score * 100).toFixed(0)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-gray-500">No frames extracted</p>
          )}
        </div>

        {/* Audio Events */}
        <div className="card p-6">
          <div className="flex items-center space-x-2 mb-4">
            <Volume2 className="h-5 w-5 text-gray-600" />
            <h2 className="text-lg font-semibold text-gray-900">Audio Events</h2>
          </div>
          
          {audio_events && audio_events.length > 0 ? (
            <div className="space-y-3">
              <p className="text-sm text-gray-600 mb-3">
                {audio_events.length} audio events detected
              </p>
              <div className="space-y-2">
                {audio_events.slice(0, 5).map((event, index) => (
                  <div key={event.id} className="bg-gray-50 rounded p-2 text-sm">
                    <div className="flex items-center justify-between">
                      <span className="font-medium capitalize">{event.event_type}</span>
                      <span className="text-gray-500">{event.timestamp.toFixed(1)}s</span>
                    </div>
                    <div className="text-gray-600 text-xs mt-1">
                      Intensity: {(event.intensity * 100).toFixed(0)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-gray-500">No audio events detected</p>
          )}
        </div>
      </div>

      {/* Processing Info */}
      <div className="card p-6">
        <div className="flex items-center space-x-2 mb-4">
          <Clock className="h-5 w-5 text-gray-600" />
          <h2 className="text-lg font-semibold text-gray-900">Processing Information</h2>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div>
            <span className="text-gray-600">Created:</span>
            <div className="font-medium">
              {new Date(video.created_at).toLocaleString()}
            </div>
          </div>
          
          {video.processed_at && (
            <div>
              <span className="text-gray-600">Processed:</span>
              <div className="font-medium">
                {new Date(video.processed_at).toLocaleString()}
              </div>
            </div>
          )}
          
          <div>
            <span className="text-gray-600">Status:</span>
            <div className="font-medium capitalize">{video.status}</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoDetails;
