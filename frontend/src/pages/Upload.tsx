import { useState } from 'react';
import { FileUpload } from '../components/ui/FileUpload';
import { Card, CardHeader, CardBody, CardFooter } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { CheckCircle, FileText, Image, Music } from 'lucide-react';

export function Upload() {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [processing, setProcessing] = useState(false);

  const handleFilesSelected = (files: File[]) => {
    setSelectedFiles((prev) => [...prev, ...files]);
  };

  const handleProcess = async () => {
    setProcessing(true);
    // Simulate processing
    await new Promise((resolve) => setTimeout(resolve, 2000));
    setProcessing(false);
    // Here you would make API calls to process the files
  };

  const features = [
    {
      icon: FileText,
      title: 'PDF Processing',
      description: 'Extract text, analyze structure, and classify documents automatically',
      supported: ['Text extraction', 'OCR', 'Classification', 'Sentiment analysis'],
    },
    {
      icon: Image,
      title: 'Image Analysis',
      description: 'Process scanned documents and images with advanced OCR',
      supported: ['OCR', 'Entity extraction', 'Image enhancement', 'Text recognition'],
    },
    {
      icon: Music,
      title: 'Audio Transcription',
      description: 'Convert speech to text with high accuracy',
      supported: ['Speech-to-text', 'Speaker detection', 'Sentiment analysis', 'Summarization'],
    },
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
          Upload Documents
        </h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          Upload PDF files, images, or audio recordings for AI-powered analysis
        </p>
      </div>

      {/* Upload Card */}
      <Card variant="elevated">
        <CardHeader>
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
            Select Files
          </h2>
        </CardHeader>
        <CardBody>
          <FileUpload
            onFilesSelected={handleFilesSelected}
            multiple={true}
            maxFiles={20}
          />
        </CardBody>
        {selectedFiles.length > 0 && (
          <CardFooter>
            <Button
              variant="primary"
              size="lg"
              loading={processing}
              onClick={handleProcess}
            >
              Process {selectedFiles.length} {selectedFiles.length === 1 ? 'File' : 'Files'}
            </Button>
          </CardFooter>
        )}
      </Card>

      {/* Features */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-6">
          What We Can Do
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {features.map((feature) => (
            <Card key={feature.title} variant="outline">
              <CardBody>
                <div className="flex items-center gap-3 mb-4">
                  <div className="p-3 rounded-lg bg-primary-100 dark:bg-primary-950">
                    <feature.icon className="text-primary-600 dark:text-primary-400" size={24} />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                    {feature.title}
                  </h3>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                  {feature.description}
                </p>
                <div className="space-y-2">
                  {feature.supported.map((item) => (
                    <div key={item} className="flex items-center gap-2 text-sm">
                      <CheckCircle className="text-success-600 dark:text-success-400 flex-shrink-0" size={16} />
                      <span className="text-gray-700 dark:text-gray-300">{item}</span>
                    </div>
                  ))}
                </div>
              </CardBody>
            </Card>
          ))}
        </div>
      </div>

      {/* Processing Pipeline */}
      <Card variant="elevated">
        <CardHeader>
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
            Processing Pipeline
          </h2>
        </CardHeader>
        <CardBody>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {[
              { step: 1, title: 'Upload', description: 'Files uploaded to secure storage' },
              { step: 2, title: 'Extract', description: 'Text and data extraction' },
              { step: 3, title: 'Analyze', description: 'AI-powered analysis' },
              { step: 4, title: 'Results', description: 'View insights and metadata' },
            ].map((stage, index) => (
              <div key={stage.step} className="relative">
                <div className="flex flex-col items-center text-center">
                  <div className="w-12 h-12 rounded-full bg-primary-600 dark:bg-primary-500 text-white flex items-center justify-center font-bold text-lg mb-3">
                    {stage.step}
                  </div>
                  <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-1">
                    {stage.title}
                  </h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {stage.description}
                  </p>
                </div>
                {index < 3 && (
                  <div className="hidden md:block absolute top-6 left-full w-full h-0.5 bg-gray-300 dark:bg-gray-700 -ml-6" />
                )}
              </div>
            ))}
          </div>
        </CardBody>
      </Card>
    </div>
  );
}
