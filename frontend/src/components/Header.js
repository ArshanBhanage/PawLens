import React from 'react';
import { Link } from 'react-router-dom';
import { Heart, Activity, Brain } from 'lucide-react';

const Header = ({ stats }) => {
  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          {/* Logo and Title */}
          <Link to="/" className="flex items-center space-x-3">
            <div className="bg-blue-600 p-2 rounded-lg">
              <Heart className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">PackLens</h1>
              <p className="text-sm text-gray-600">Dog Behavior Analysis</p>
            </div>
          </Link>

          {/* Quick Stats */}
          {stats && (
            <div className="hidden md:flex items-center space-x-6">
              <div className="flex items-center space-x-2">
                <Activity className="h-5 w-5 text-blue-600" />
                <div>
                  <p className="text-sm font-medium text-gray-900">
                    {stats.processed_videos}
                  </p>
                  <p className="text-xs text-gray-600">Videos Analyzed</p>
                </div>
              </div>
              
              <div className="flex items-center space-x-2">
                <Brain className="h-5 w-5 text-blue-600" />
                <div>
                  <p className="text-sm font-medium text-gray-900">
                    {stats.total_frames}
                  </p>
                  <p className="text-xs text-gray-600">Frames Processed</p>
                </div>
              </div>

              {stats.processing_videos > 0 && (
                <div className="flex items-center space-x-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                  <div>
                    <p className="text-sm font-medium text-blue-600">
                      {stats.processing_videos}
                    </p>
                    <p className="text-xs text-gray-600">Processing</p>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;
