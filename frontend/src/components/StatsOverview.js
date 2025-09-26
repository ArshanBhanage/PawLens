import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Activity, Brain, AlertTriangle, Heart } from 'lucide-react';

const StatsOverview = ({ stats }) => {
  if (!stats) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="card p-6 animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
            <div className="h-8 bg-gray-200 rounded w-1/2"></div>
          </div>
        ))}
      </div>
    );
  }

  const behaviorData = [
    { name: 'Anxiety', value: stats.average_scores.anxiety_level, color: '#ef4444' },
    { name: 'Playfulness', value: stats.average_scores.playfulness, color: '#3b82f6' },
    { name: 'Aggression', value: stats.average_scores.aggression, color: '#f59e0b' },
    { name: 'Confidence', value: stats.average_scores.confidence, color: '#22c55e' },
    { name: 'Energy', value: stats.average_scores.energy_level, color: '#8b5cf6' },
    { name: 'Stress', value: stats.average_scores.stress_indicators, color: '#f97316' },
  ];

  const StatCard = ({ icon: Icon, title, value, subtitle, color = 'text-primary-600' }) => (
    <div className="card p-6">
      <div className="flex items-center">
        <div className={`p-2 rounded-lg bg-opacity-10 ${color.replace('text-', 'bg-').replace('-600', '-100')}`}>
          <Icon className={`h-6 w-6 ${color}`} />
        </div>
        <div className="ml-4">
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
          {subtitle && <p className="text-xs text-gray-500">{subtitle}</p>}
        </div>
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <StatCard
          icon={Activity}
          title="Videos Analyzed"
          value={stats.processed_videos}
          subtitle={`${stats.total_videos} total`}
          color="text-primary-600"
        />
        <StatCard
          icon={Brain}
          title="Frames Processed"
          value={stats.total_frames.toLocaleString()}
          subtitle="AI analyzed"
          color="text-success-600"
        />
        <StatCard
          icon={Heart}
          title="Audio Events"
          value={stats.total_audio_events}
          subtitle="Detected sounds"
          color="text-warning-600"
        />
        <StatCard
          icon={AlertTriangle}
          title="Processing"
          value={stats.processing_videos}
          subtitle={stats.failed_videos > 0 ? `${stats.failed_videos} failed` : 'In queue'}
          color={stats.processing_videos > 0 ? "text-warning-600" : "text-gray-600"}
        />
      </div>

      {/* Average Behavior Scores Chart */}
      {stats.processed_videos > 0 && (
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Average Behavioral Scores
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={behaviorData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="name" 
                  tick={{ fontSize: 12 }}
                  angle={-45}
                  textAnchor="end"
                  height={60}
                />
                <YAxis 
                  domain={[0, 10]}
                  tick={{ fontSize: 12 }}
                />
                <Tooltip 
                  formatter={(value) => [`${value.toFixed(1)}/10`, 'Score']}
                  labelStyle={{ color: '#374151' }}
                />
                <Bar 
                  dataKey="value" 
                  fill="#3b82f6"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-sm text-gray-600 mt-2">
            Based on {stats.processed_videos} analyzed video{stats.processed_videos !== 1 ? 's' : ''}
          </p>
        </div>
      )}
    </div>
  );
};

export default StatsOverview;
