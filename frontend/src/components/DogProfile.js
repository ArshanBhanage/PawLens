import React, { useState, useEffect } from 'react';
import { Heart, Activity, Shield, Zap, Brain, AlertTriangle, Dog, Star, Calendar, TrendingUp } from 'lucide-react';
import LoadingSpinner from './LoadingSpinner';

const DogProfile = () => {
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchDogProfile();
  }, []);

  const fetchDogProfile = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/dog-profile');
      const data = await response.json();
      
      if (data.success) {
        setProfile(data.profile);
      } else {
        setError(data.error);
      }
    } catch (err) {
      setError('Failed to load dog profile');
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score) => {
    if (score <= 3) return 'text-green-600 bg-green-100';
    if (score <= 6) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getScoreIcon = (trait) => {
    const iconProps = { className: "h-5 w-5" };
    switch (trait) {
      case 'anxiety': return <AlertTriangle {...iconProps} />;
      case 'playfulness': return <Heart {...iconProps} />;
      case 'aggression': return <Shield {...iconProps} />;
      case 'confidence': return <Star {...iconProps} />;
      case 'energy': return <Zap {...iconProps} />;
      case 'stress': return <Activity {...iconProps} />;
      default: return <Brain {...iconProps} />;
    }
  };

  const ScoreBar = ({ score, maxScore = 10 }) => {
    const percentage = (score / maxScore) * 100;
    const colorClass = score <= 3 ? 'bg-green-500' : score <= 6 ? 'bg-yellow-500' : 'bg-red-500';
    
    return (
      <div className="w-full bg-gray-200 rounded-full h-3">
        <div 
          className={`h-3 rounded-full transition-all duration-500 ${colorClass}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    );
  };

  if (loading) return <LoadingSpinner />;
  if (error) return <div className="text-red-600 p-4">Error: {error}</div>;
  if (!profile) return <div className="text-gray-600 p-4">No profile data available</div>;

  const { behavioral_scores, breed_info, statistics, recent_assessments } = profile;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card p-6">
        <div className="flex items-center space-x-4 mb-6">
          <div className="p-3 bg-blue-100 rounded-full">
            <Dog className="h-8 w-8 text-blue-600" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Your White Dog's Profile</h1>
            <p className="text-gray-600">Comprehensive behavioral analysis across all videos</p>
          </div>
        </div>

        {/* Statistics */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="bg-blue-50 p-4 rounded-lg">
            <div className="flex items-center space-x-2">
              <Calendar className="h-5 w-5 text-blue-600" />
              <span className="text-sm font-medium text-blue-900">Videos Analyzed</span>
            </div>
            <p className="text-2xl font-bold text-blue-600 mt-1">{statistics.total_videos_analyzed}</p>
          </div>
          <div className="bg-green-50 p-4 rounded-lg">
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-5 w-5 text-green-600" />
              <span className="text-sm font-medium text-green-900">Total Videos</span>
            </div>
            <p className="text-2xl font-bold text-green-600 mt-1">{statistics.total_videos}</p>
          </div>
          <div className="bg-purple-50 p-4 rounded-lg">
            <div className="flex items-center space-x-2">
              <Brain className="h-5 w-5 text-purple-600" />
              <span className="text-sm font-medium text-purple-900">Analysis Period</span>
            </div>
            <p className="text-sm font-bold text-purple-600 mt-1">
              {statistics.analysis_period.start ? 
                new Date(statistics.analysis_period.start).toLocaleDateString() : 'N/A'}
            </p>
          </div>
        </div>
      </div>

      {/* Behavioral Scores */}
      <div className="card p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-6 flex items-center">
          <Activity className="h-6 w-6 mr-2 text-blue-600" />
          Average Behavioral Scores
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {Object.entries(behavioral_scores).map(([trait, score]) => (
            <div key={trait} className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  {getScoreIcon(trait)}
                  <span className="font-medium text-gray-900 capitalize">{trait}</span>
                </div>
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${getScoreColor(score)}`}>
                  {score}/10
                </span>
              </div>
              <ScoreBar score={score} />
              <div className="text-xs text-gray-500">
                {trait === 'anxiety' && score <= 3 && "😌 Low anxiety - very calm and relaxed"}
                {trait === 'anxiety' && score > 3 && score <= 6 && "😐 Moderate anxiety - some stress indicators"}
                {trait === 'anxiety' && score > 6 && "😰 High anxiety - needs attention and support"}
                
                {trait === 'playfulness' && score <= 3 && "😴 Low playfulness - prefers calm activities"}
                {trait === 'playfulness' && score > 3 && score <= 6 && "🙂 Moderate playfulness - enjoys some play"}
                {trait === 'playfulness' && score > 6 && "🎾 High playfulness - loves to play and interact"}
                
                {trait === 'aggression' && score <= 3 && "😇 Low aggression - very gentle and peaceful"}
                {trait === 'aggression' && score > 3 && score <= 6 && "⚠️ Moderate aggression - monitor interactions"}
                {trait === 'aggression' && score > 6 && "🚨 High aggression - requires careful management"}
                
                {trait === 'confidence' && score <= 3 && "😟 Low confidence - may need encouragement"}
                {trait === 'confidence' && score > 3 && score <= 6 && "😊 Moderate confidence - generally comfortable"}
                {trait === 'confidence' && score > 6 && "😎 High confidence - very self-assured"}
                
                {trait === 'energy' && score <= 3 && "😴 Low energy - prefers rest and calm activities"}
                {trait === 'energy' && score > 3 && score <= 6 && "🚶 Moderate energy - balanced activity needs"}
                {trait === 'energy' && score > 6 && "🏃 High energy - needs lots of exercise and stimulation"}
                
                {trait === 'stress' && score <= 3 && "😌 Low stress - handles situations well"}
                {trait === 'stress' && score > 3 && score <= 6 && "😐 Moderate stress - some situational stress"}
                {trait === 'stress' && score > 6 && "😰 High stress - needs stress management support"}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Breed Information */}
      <div className="card p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-6 flex items-center">
          <Dog className="h-6 w-6 mr-2 text-green-600" />
          Breed Analysis
        </h2>
        
        <div className="space-y-4">
          <div className="bg-green-50 p-4 rounded-lg">
            <h3 className="font-semibold text-green-900 mb-2">Likely Breed</h3>
            <p className="text-lg font-bold text-green-700">{breed_info.likely_breed}</p>
            <p className="text-sm text-green-600 mt-1">
              Confidence: {breed_info.confidence_level} • {breed_info.breed_notes}
            </p>
          </div>

          <div>
            <h3 className="font-semibold text-gray-900 mb-3">Breed Characteristics</h3>
            <ul className="space-y-2">
              {breed_info.characteristics.map((characteristic, index) => (
                <li key={index} className="flex items-start space-x-2">
                  <Star className="h-4 w-4 text-yellow-500 mt-0.5 flex-shrink-0" />
                  <span className="text-gray-700">{characteristic}</span>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h3 className="font-semibold text-gray-900 mb-3">Care Recommendations</h3>
            <ul className="space-y-2">
              {breed_info.recommendations.map((recommendation, index) => (
                <li key={index} className="flex items-start space-x-2">
                  <Heart className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
                  <span className="text-gray-700">{recommendation}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* Recent Assessments */}
      {recent_assessments.length > 0 && (
        <div className="card p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-6 flex items-center">
            <TrendingUp className="h-6 w-6 mr-2 text-purple-600" />
            Recent Overall Assessments
          </h2>
          
          <div className="space-y-4">
            {recent_assessments.map((assessment, index) => (
              <div key={index} className="bg-purple-50 p-4 rounded-lg">
                <p className="text-gray-700 leading-relaxed">{assessment}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default DogProfile;
