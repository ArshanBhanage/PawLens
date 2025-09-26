import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, CheckCircle, AlertCircle, Loader } from 'lucide-react';

const UploadArea = ({ onUpload }) => {
  const [uploadStatus, setUploadStatus] = useState(null); // null, 'uploading', 'success', 'error'
  const [uploadMessage, setUploadMessage] = useState('');

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    },
    multiple: false,
    maxSize: 500 * 1024 * 1024, // 500MB
    onDrop: async (acceptedFiles, rejectedFiles) => {
      if (rejectedFiles.length > 0) {
        const rejection = rejectedFiles[0];
        if (rejection.file.size > 500 * 1024 * 1024) {
          setUploadStatus('error');
          setUploadMessage('File too large. Maximum size is 500MB.');
        } else {
          setUploadStatus('error');
          setUploadMessage('Invalid file type. Please upload a video file.');
        }
        return;
      }

      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0];
        setUploadStatus('uploading');
        setUploadMessage(`Uploading ${file.name}...`);

        try {
          const result = await onUpload(file);
          if (result.success) {
            setUploadStatus('success');
            setUploadMessage(result.message || 'Video uploaded successfully!');
            
            // Clear success message after 5 seconds
            setTimeout(() => {
              setUploadStatus(null);
              setUploadMessage('');
            }, 5000);
          } else {
            setUploadStatus('error');
            setUploadMessage(result.error || 'Upload failed');
          }
        } catch (error) {
          setUploadStatus('error');
          setUploadMessage('Upload failed. Please try again.');
        }
      }
    }
  });

  const getStatusIcon = () => {
    switch (uploadStatus) {
      case 'uploading':
        return <Loader className="h-8 w-8 text-blue-600 animate-spin" />;
      case 'success':
        return <CheckCircle className="h-8 w-8 text-green-600" />;
      case 'error':
        return <AlertCircle className="h-8 w-8 text-red-600" />;
      default:
        return <Upload className="h-8 w-8 text-gray-400" />;
    }
  };

  const getStatusColor = () => {
    switch (uploadStatus) {
      case 'uploading':
        return 'border-blue-300 bg-blue-50';
      case 'success':
        return 'border-green-300 bg-green-50';
      case 'error':
        return 'border-red-300 bg-red-50';
      default:
        return isDragActive ? 'border-blue-400 bg-blue-50' : 'border-gray-300 bg-white';
    }
  };

  const getMessageColor = () => {
    switch (uploadStatus) {
      case 'uploading':
        return 'text-blue-700';
      case 'success':
        return 'text-green-700';
      case 'error':
        return 'text-red-700';
      default:
        return 'text-gray-600';
    }
  };

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold text-gray-900">Upload New Video</h2>
      
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-200
          ${getStatusColor()}
          ${uploadStatus === 'uploading' ? 'cursor-not-allowed' : 'hover:border-blue-400 hover:bg-blue-50'}
        `}
      >
        <input {...getInputProps()} disabled={uploadStatus === 'uploading'} />
        
        <div className="space-y-4">
          {getStatusIcon()}
          
          <div>
            {uploadStatus ? (
              <div>
                <p className={`text-lg font-medium ${getMessageColor()}`}>
                  {uploadMessage}
                </p>
                {uploadStatus === 'uploading' && (
                  <p className="text-sm text-gray-500 mt-1">
                    This may take a few moments...
                  </p>
                )}
                {uploadStatus === 'success' && (
                  <p className="text-sm text-gray-500 mt-1">
                    Your video will be processed in the background and appear in the grid below.
                  </p>
                )}
              </div>
            ) : (
              <div>
                <p className="text-lg font-medium text-gray-900">
                  {isDragActive ? 'Drop your video here' : 'Drag & drop a video file here'}
                </p>
                <p className="text-gray-600 mt-1">
                  or click to browse files
                </p>
                <p className="text-sm text-gray-500 mt-2">
                  Supports MP4, AVI, MOV, MKV, WMV (max 500MB)
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {uploadStatus === 'error' && (
        <div className="text-center">
          <button
            onClick={() => {
              setUploadStatus(null);
              setUploadMessage('');
            }}
            className="btn-secondary"
          >
            Try Again
          </button>
        </div>
      )}
    </div>
  );
};

export default UploadArea;
