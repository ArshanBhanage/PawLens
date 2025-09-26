import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import axios from 'axios';
import { ArrowLeft, Clock, Volume2, Image, AlertTriangle, CheckCircle, Play } from 'lucide-react';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer } from 'recharts';
import LoadingSpinner from './LoadingSpinner';

// Helper function to parse key moments and extract timestamps
const parseKeyMoments = (keyMomentsText) => {
  if (!keyMomentsText) return [];
  
  const moments = [];
  const lines = keyMomentsText.split('\n').filter(line => line.trim());
  
  lines.forEach(line => {
    // Extract timestamp ranges like "1.5s-4.5s" or "16.5s"
    const timeMatch = line.match(/(\d+\.?\d*)s(?:-(\d+\.?\d*)s)?/);
    const frameMatch = line.match(/\(Frames? (\d+(?:,?\s*\d+)*(?:\s*and\s*\d+)?)\)/);
    
    if (timeMatch) {
      const startTime = parseFloat(timeMatch[1]);
      const endTime = timeMatch[2] ? parseFloat(timeMatch[2]) : startTime;
      const frameNumbers = frameMatch ? frameMatch[1].split(/[,\s]+and\s+|[,\s]+/).map(f => parseInt(f.trim())) : [];
      
      moments.push({
        startTime,
        endTime,
        frameNumbers,
        description: line.replace(/^\d+\.\s*/, '').trim()
      });
    }
  });
  
  return moments;
};

// Video Player Component for Key Moments
const VideoPlayer = ({ videoId, videoFilename, currentSegment, onTimeUpdate }) => {
  const videoRef = React.useRef(null);
  const [isPlaying, setIsPlaying] = React.useState(false);
  const [currentTime, setCurrentTime] = React.useState(0);

  // Handle segment playback and general time updates
  React.useEffect(() => {
    if (videoRef.current) {
      const video = videoRef.current;
      let segmentTimeoutId = null;
      
      const handleTimeUpdate = () => {
        const time = video.currentTime;
        setCurrentTime(time);
        onTimeUpdate && onTimeUpdate(time);
        
        // If we're playing a segment and reached the end time, pause
        if (currentSegment && time >= currentSegment.endTime - 0.1) {
          video.pause();
          setIsPlaying(false);
        }
      };
      
      const handlePlay = () => setIsPlaying(true);
      const handlePause = () => setIsPlaying(false);
      
      // Add event listeners
      video.addEventListener('timeupdate', handleTimeUpdate);
      video.addEventListener('play', handlePlay);
      video.addEventListener('pause', handlePause);
      
      // If we have a current segment, start playing it
      if (currentSegment) {
        const startSegmentPlayback = async () => {
          try {
            video.currentTime = currentSegment.startTime;
            await video.play();
            setIsPlaying(true);
          } catch (error) {
            console.log('Playback failed:', error);
            setIsPlaying(false);
          }
        };
        
        // Small delay to ensure video is ready
        segmentTimeoutId = setTimeout(startSegmentPlayback, 100);
      }
      
      return () => {
        video.removeEventListener('timeupdate', handleTimeUpdate);
        video.removeEventListener('play', handlePlay);
        video.removeEventListener('pause', handlePause);
        if (segmentTimeoutId) {
          clearTimeout(segmentTimeoutId);
        }
      };
    }
  }, [currentSegment, onTimeUpdate]);

  return (
    <div className="card p-6 mb-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
        <Play className="h-5 w-5 mr-2 text-green-600" />
        Video Player
        {currentSegment && (
          <span className="ml-2 text-sm text-gray-600">
            Playing: {currentSegment.startTime}s - {currentSegment.endTime}s
          </span>
        )}
      </h2>
      
      <div className="relative bg-black rounded-lg overflow-hidden">
        <video
          ref={videoRef}
          className="w-full h-64 object-contain"
          controls
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          onTimeUpdate={(e) => setCurrentTime(e.target.currentTime)}
        >
          <source src={`/api/videos/${videoId}/video`} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
        
        {currentSegment && (
          <div className="absolute top-2 left-2 bg-blue-600 text-white text-xs px-2 py-1 rounded">
            Key Moment: {currentSegment.startTime}s - {currentSegment.endTime}s
          </div>
        )}
      </div>
      
      <div className="mt-4 flex items-center justify-between">
        <div className="text-sm text-gray-600">
          Current time: {currentTime.toFixed(1)}s
        </div>
        {currentSegment && (
          <div className="text-sm text-blue-600 font-medium">
            {currentSegment.description.substring(0, 100)}...
          </div>
        )}
      </div>
    </div>
  );
};

// Interactive Key Moments Component
const InteractiveKeyMoments = ({ keyMoments, videoId, onMomentClick }) => {
  const moments = parseKeyMoments(keyMoments);

  if (moments.length === 0) return null;

  return (
    <div className="card p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
        <Image className="h-5 w-5 mr-2 text-blue-600" />
        Interactive Key Moments
      </h2>
      
      <div className="space-y-4">
        {moments.map((moment, index) => (
          <div 
            key={index} 
            className="border-l-4 border-blue-500 pl-4 cursor-pointer hover:bg-blue-50 p-3 rounded-r-lg transition-colors"
            onClick={() => onMomentClick(moment)}
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <span className="bg-blue-100 text-blue-800 text-xs font-medium px-2 py-1 rounded">
                  Moment {index + 1}
                </span>
                <span className="text-sm text-gray-600">
                  {moment.startTime === moment.endTime 
                    ? `${moment.startTime}s` 
                    : `${moment.startTime}s - ${moment.endTime}s`}
                </span>
              </div>
              <button className="flex items-center space-x-1 text-blue-600 hover:text-blue-800 text-sm font-medium">
                <Play className="h-4 w-4" />
                <span>Play Segment</span>
              </button>
            </div>
            
            <p className="text-sm text-gray-700 leading-relaxed">
              {moment.description}
            </p>
          </div>
        ))}
      </div>
      
      <div className="mt-4 p-3 bg-blue-50 rounded-lg">
        <p className="text-sm text-blue-800">
          💡 <strong>Tip:</strong> Click on any key moment above to play that specific video segment and see the exact behavioral context.
        </p>
      </div>
    </div>
  );
};

// Visual Timeline Component
const KeyMomentsTimeline = ({ keyMoments, audioEvents, videoDuration, currentTime, onTimelineClick }) => {
  const moments = parseKeyMoments(keyMoments);
  
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getPositionPercent = (time) => (time / videoDuration) * 100;

  return (
    <div className="card p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
        <Clock className="h-5 w-5 mr-2 text-green-600" />
        Behavioral Timeline
      </h2>
      
      <div className="relative">
        {/* Timeline bar */}
        <div 
          className="relative h-12 bg-gray-100 rounded-lg mb-6 cursor-pointer"
          onClick={(e) => {
            if (onTimelineClick) {
              const rect = e.currentTarget.getBoundingClientRect();
              const clickX = e.clientX - rect.left;
              const timelineWidth = rect.width;
              const clickedTime = (clickX / timelineWidth) * videoDuration;
              onTimelineClick(clickedTime);
            }
          }}
        >
          {/* Time markers */}
          <div className="absolute inset-0 flex items-center justify-between px-2 text-xs text-gray-500 pointer-events-none">
            <span>0:00</span>
            <span>{formatTime(videoDuration)}</span>
          </div>
          
          {/* Current time indicator */}
          {currentTime > 0 && (
            <div
              className="absolute top-0 w-1 h-full bg-green-500 rounded-full z-10"
              style={{ left: `${getPositionPercent(currentTime)}%` }}
            >
              <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 bg-green-600 text-white text-xs px-1 py-0.5 rounded whitespace-nowrap">
                {formatTime(currentTime)}
              </div>
            </div>
          )}
          
          {/* Audio events */}
          {audioEvents.map((event, index) => (
            <div
              key={index}
              className={`absolute top-1 w-2 h-2 rounded-full ${
                event.event_type === 'growl' ? 'bg-red-500' : 
                event.event_type === 'bark' ? 'bg-orange-500' : 'bg-yellow-500'
              }`}
              style={{ left: `${getPositionPercent(event.timestamp)}%` }}
              title={`${event.event_type} at ${event.timestamp}s`}
            />
          ))}
          
          {/* Key moments */}
          {moments.map((moment, index) => (
            <div
              key={index}
              className="absolute top-0 h-full"
              style={{
                left: `${getPositionPercent(moment.startTime)}%`,
                width: `${getPositionPercent(moment.endTime - moment.startTime) || 2}%`
              }}
            >
              <div className="h-full bg-blue-500 bg-opacity-30 border-l-2 border-blue-500 rounded-r">
                <div className="absolute -top-8 left-0 bg-blue-600 text-white text-xs px-2 py-1 rounded whitespace-nowrap">
                  Moment {index + 1}
                </div>
              </div>
            </div>
          ))}
        </div>
        
        {/* Legend */}
        <div className="flex flex-wrap items-center space-x-6 text-sm">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-blue-500 bg-opacity-30 border-l-2 border-blue-500"></div>
            <span className="text-gray-600">Key Moments</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-red-500 rounded-full"></div>
            <span className="text-gray-600">Growls</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
            <span className="text-gray-600">Barks</span>
          </div>
        </div>
        
        {/* Key moments descriptions */}
        <div className="mt-6 space-y-4">
          {moments.map((moment, index) => (
            <div key={index} className="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg">
              <div className="flex-shrink-0 w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center text-sm font-medium">
                {index + 1}
              </div>
              <div className="flex-1">
                <div className="flex items-center space-x-2 mb-1">
                  <span className="font-medium text-gray-900">
                    {moment.startTime === moment.endTime 
                      ? `${moment.startTime}s` 
                      : `${moment.startTime}s - ${moment.endTime}s`}
                  </span>
                  <Play className="h-4 w-4 text-gray-400" />
                </div>
                <p className="text-gray-700 text-sm leading-relaxed">
                  {moment.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const VideoDetails = () => {
  const { id } = useParams();
  const [videoData, setVideoData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentSegment, setCurrentSegment] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);

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

  const handleMomentClick = (moment) => {
    setCurrentSegment(moment);
    // Scroll to video player
    const videoPlayer = document.querySelector('[data-video-player]');
    if (videoPlayer) {
      videoPlayer.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  const handleTimeUpdate = (time) => {
    setCurrentTime(time);
  };

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
        <div className="space-y-6">
          {/* Video Player */}
          <div data-video-player>
            <VideoPlayer 
              videoId={id}
              videoFilename={video.filename}
              currentSegment={currentSegment}
              onTimeUpdate={handleTimeUpdate}
            />
          </div>
          
          {/* Interactive Key Moments */}
          <InteractiveKeyMoments 
            keyMoments={analysis.key_moments}
            videoId={id}
            onMomentClick={handleMomentClick}
          />
          
          {/* Visual Timeline */}
          <KeyMomentsTimeline 
            keyMoments={analysis.key_moments}
            audioEvents={audio_events}
            videoDuration={video.duration}
            currentTime={currentTime}
            onTimelineClick={(time) => {
              // Find if clicked time is within any moment
              const moments = parseKeyMoments(analysis.key_moments);
              const clickedMoment = moments.find(m => time >= m.startTime && time <= m.endTime);
              if (clickedMoment) {
                handleMomentClick(clickedMoment);
              }
            }}
          />
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
