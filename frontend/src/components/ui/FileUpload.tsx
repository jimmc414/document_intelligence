import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import type { FileRejection } from 'react-dropzone';
import { Upload, File, X, AlertCircle } from 'lucide-react';
import { cn } from '../../lib/utils/cn';
import { Progress } from './Progress';

export interface FileUploadProps {
  onFilesSelected: (files: File[]) => void;
  accept?: Record<string, string[]>;
  maxSize?: number;
  maxFiles?: number;
  multiple?: boolean;
  disabled?: boolean;
}

interface FileWithProgress {
  file: File;
  progress: number;
  error?: string;
}

export function FileUpload({
  onFilesSelected,
  accept = {
    'application/pdf': ['.pdf'],
    'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'],
    'audio/*': ['.mp3', '.wav', '.m4a', '.ogg'],
  },
  maxSize = 10 * 1024 * 1024, // 10MB
  maxFiles = 10,
  multiple = true,
  disabled = false,
}: FileUploadProps) {
  const [files, setFiles] = useState<FileWithProgress[]>([]);
  const [errors, setErrors] = useState<string[]>([]);

  const onDrop = useCallback(
    (acceptedFiles: File[], rejectedFiles: FileRejection[]) => {
      // Handle rejected files
      if (rejectedFiles.length > 0) {
        const errorMessages = rejectedFiles.map((rejection) => {
          const errors = rejection.errors.map((e) => e.message).join(', ');
          return `${rejection.file.name}: ${errors}`;
        });
        setErrors(errorMessages);
      }

      // Handle accepted files
      if (acceptedFiles.length > 0) {
        const newFiles: FileWithProgress[] = acceptedFiles.map((file) => ({
          file,
          progress: 0,
        }));
        setFiles((prev) => [...prev, ...newFiles]);
        onFilesSelected(acceptedFiles);
        setErrors([]);
      }
    },
    [onFilesSelected]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept,
    maxSize,
    maxFiles,
    multiple,
    disabled,
  });

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className="w-full space-y-4">
      {/* Drop zone */}
      <div
        {...getRootProps()}
        className={cn(
          'relative border-2 border-dashed rounded-lg p-8',
          'transition-all duration-fast cursor-pointer',
          'flex flex-col items-center justify-center gap-4',
          'min-h-[200px]',
          isDragActive
            ? 'border-primary-500 bg-primary-50 dark:bg-primary-950'
            : 'border-gray-300 dark:border-gray-700 hover:border-primary-400 dark:hover:border-primary-600',
          disabled && 'opacity-50 cursor-not-allowed'
        )}
      >
        <input {...getInputProps()} />

        <Upload
          size={48}
          className={cn(
            'transition-colors',
            isDragActive
              ? 'text-primary-600 dark:text-primary-400'
              : 'text-gray-400 dark:text-gray-600'
          )}
        />

        <div className="text-center">
          <p className="text-lg font-medium text-gray-900 dark:text-gray-100">
            {isDragActive ? 'Drop files here' : 'Drag & drop files here'}
          </p>
          <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
            or click to browse
          </p>
        </div>

        <div className="text-xs text-gray-500 dark:text-gray-500 text-center space-y-1">
          <p>Supported: PDF, Images (PNG, JPG, etc.), Audio (MP3, WAV, etc.)</p>
          <p>Max file size: {formatFileSize(maxSize)}</p>
          {multiple && <p>Max files: {maxFiles}</p>}
        </div>
      </div>

      {/* Error messages */}
      {errors.length > 0 && (
        <div className="rounded-lg bg-error-50 dark:bg-error-950 border border-error-200 dark:border-error-800 p-4">
          <div className="flex items-start gap-3">
            <AlertCircle className="text-error-600 dark:text-error-400 flex-shrink-0 mt-0.5" size={20} />
            <div className="flex-1 space-y-1">
              {errors.map((error, index) => (
                <p key={index} className="text-sm text-error-900 dark:text-error-100">
                  {error}
                </p>
              ))}
            </div>
            <button
              onClick={() => setErrors([])}
              className="text-error-600 dark:text-error-400 hover:text-error-800 dark:hover:text-error-200"
            >
              <X size={16} />
            </button>
          </div>
        </div>
      )}

      {/* File list */}
      {files.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100">
            Uploaded Files ({files.length})
          </h3>
          <div className="space-y-2">
            {files.map((fileItem, index) => (
              <div
                key={index}
                className="flex items-center gap-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700"
              >
                <File className="text-gray-400 flex-shrink-0" size={20} />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                    {fileItem.file.name}
                  </p>
                  <p className="text-xs text-gray-600 dark:text-gray-400">
                    {formatFileSize(fileItem.file.size)}
                  </p>
                  {fileItem.progress > 0 && fileItem.progress < 100 && (
                    <Progress value={fileItem.progress} size="sm" className="mt-2" />
                  )}
                  {fileItem.error && (
                    <p className="text-xs text-error-600 dark:text-error-400 mt-1">
                      {fileItem.error}
                    </p>
                  )}
                </div>
                <button
                  onClick={() => removeFile(index)}
                  className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 flex-shrink-0"
                >
                  <X size={16} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
