import React from 'react';
import { Link } from 'react-router-dom';
import { Play, Clock, AlertTriangle, CheckCircle, Loader } from 'lucide-react';

const VideoGrid = ({ videos }) => {
  if (!videos || videos.length === 0) {
    return (
      <div className="card p-8 text-center">
        <div className="text-gray-400 text-6xl mb-4">🎬</div>
        <h3 className="text-xl font-semibold text-gray-900 mb-2">No Videos Yet</h3>
        <p className="text-gray-600">Upload your first dog video to get started with behavioral analysis!</p>
      </div>
    );
  }

  const getStatusBadge = (status) => {
    switch (status) {
      case 'completed':
        return <span className="badge-success"><CheckCircle className="h-3 w-3 mr-1" />Analyzed</span>;
      case 'processing':
        return <span className="badge-warning"><Loader className="h-3 w-3 mr-1 animate-spin" />Processing</span>;
      case 'failed':
        return <span className="badge-danger"><AlertTriangle className="h-3 w-3 mr-1" />Failed</span>;
      default:
        return <span className="badge-info"><Clock className="h-3 w-3 mr-1" />Pending</span>;
    }
  };

  const getAnxietyColor = (level) => {
    if (level <= 3) return 'text-green-600';
    if (level <= 6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const formatDuration = (seconds) => {
    if (!seconds) return 'Unknown';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Video Analysis</h2>
        <p className="text-gray-600">{videos.length} video{videos.length !== 1 ? 's' : ''}</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {videos.map((video) => (
          <Link
            key={video.id}
            to={`/video/${video.id}`}
            className="card hover:shadow-lg transition-shadow duration-200 overflow-hidden group"
          >
            {/* Video Thumbnail Placeholder */}
            <div className="bg-gray-200 h-48 flex items-center justify-center group-hover:bg-gray-300 transition-colors">
              <Play className="h-12 w-12 text-gray-400 group-hover:text-gray-500" />
            </div>

            <div className="p-4">
              {/* Header */}
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1 min-w-0">
                  <h3 className="font-semibold text-gray-900 truncate" title={video.filename}>
                    {video.filename}
                  </h3>
                  <p className="text-sm text-gray-600">
                    {formatDuration(video.duration)} • {video.dimensions}
                  </p>
                </div>
                <div className="ml-2">
                  {getStatusBadge(video.status)}
                </div>
              </div>

              {/* Analysis Results */}
              {video.analysis && (
                <div className="space-y-3">
                  {/* Key Metrics */}
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <span className="text-gray-600">Anxiety:</span>
                      <span className={`ml-1 font-medium ${getAnxietyColor(video.analysis.anxiety_level)}`}>
                        {video.analysis.anxiety_level}/10
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">Playfulness:</span>
                      <span className="ml-1 font-medium text-blue-600">
                        {video.analysis.playfulness}/10
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">Confidence:</span>
                      <span className="ml-1 font-medium text-green-600">
                        {video.analysis.confidence}/10
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">Energy:</span>
                      <span className="ml-1 font-medium text-yellow-600">
                        {video.analysis.energy_level}/10
                      </span>
                    </div>
                  </div>

                  {/* Overall Assessment Preview */}
                  {video.analysis.overall_assessment && (
                    <div className="border-t pt-3">
                      <p className="text-sm text-gray-700 line-clamp-2">
                        {video.analysis.overall_assessment.length > 100
                          ? `${video.analysis.overall_assessment.substring(0, 100)}...`
                          : video.analysis.overall_assessment
                        }
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* Processing Info */}
              {video.status === 'processing' && (
                <div className="text-center py-4">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto mb-2"></div>
                  <p className="text-sm text-gray-600">Analyzing behavioral patterns...</p>
                </div>
              )}

              {/* Failed State */}
              {video.status === 'failed' && (
                <div className="text-center py-4">
                  <AlertTriangle className="h-8 w-8 text-red-500 mx-auto mb-2" />
                  <p className="text-sm text-gray-600">Analysis failed. Click to retry.</p>
                </div>
              )}

              {/* Footer */}
              <div className="flex items-center justify-between mt-4 pt-3 border-t text-xs text-gray-500">
                <span>{new Date(video.created_at).toLocaleDateString()}</span>
                <div className="flex items-center space-x-3">
                  <span>{video.frame_count} frames</span>
                  <span>{video.audio_event_count} audio events</span>
                </div>
              </div>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
};

export default VideoGrid;
