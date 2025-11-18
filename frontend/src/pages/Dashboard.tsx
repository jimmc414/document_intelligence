import { FileText, Zap, TrendingUp, Clock } from 'lucide-react';
import { Card, CardHeader, CardBody } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { Progress } from '../components/ui/Progress';

export function Dashboard() {
  const stats = [
    {
      label: 'Total Documents',
      value: '1,234',
      change: '+12.5%',
      icon: FileText,
      color: 'text-primary-600 dark:text-primary-400',
      bg: 'bg-primary-100 dark:bg-primary-950',
    },
    {
      label: 'Processed Today',
      value: '89',
      change: '+8.2%',
      icon: Zap,
      color: 'text-success-600 dark:text-success-400',
      bg: 'bg-success-100 dark:bg-success-950',
    },
    {
      label: 'Accuracy Rate',
      value: '98.5%',
      change: '+0.3%',
      icon: TrendingUp,
      color: 'text-info-600 dark:text-info-400',
      bg: 'bg-info-100 dark:bg-info-950',
    },
    {
      label: 'Avg Processing Time',
      value: '2.3s',
      change: '-15.8%',
      icon: Clock,
      color: 'text-warning-600 dark:text-warning-400',
      bg: 'bg-warning-100 dark:bg-warning-950',
    },
  ];

  const recentDocuments = [
    { id: 1, name: 'Invoice_Q4_2024.pdf', status: 'completed', progress: 100, type: 'Invoice' },
    { id: 2, name: 'Contract_ABC_Corp.pdf', status: 'processing', progress: 65, type: 'Contract' },
    { id: 3, name: 'Receipt_12345.jpg', status: 'completed', progress: 100, type: 'Receipt' },
    { id: 4, name: 'Meeting_Recording.mp3', status: 'queued', progress: 0, type: 'Audio' },
  ];

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return <Badge variant="success" size="sm">Completed</Badge>;
      case 'processing':
        return <Badge variant="info" size="sm" dot>Processing</Badge>;
      case 'queued':
        return <Badge variant="default" size="sm">Queued</Badge>;
      default:
        return null;
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
          Dashboard
        </h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          Welcome back! Here's what's happening with your documents.
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat) => (
          <Card key={stat.label} variant="elevated">
            <CardBody>
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                    {stat.label}
                  </p>
                  <p className="mt-2 text-3xl font-bold text-gray-900 dark:text-gray-100">
                    {stat.value}
                  </p>
                  <p className="mt-2 text-sm text-success-600 dark:text-success-400">
                    {stat.change} from last month
                  </p>
                </div>
                <div className={`p-3 rounded-lg ${stat.bg}`}>
                  <stat.icon className={`${stat.color}`} size={24} />
                </div>
              </div>
            </CardBody>
          </Card>
        ))}
      </div>

      {/* Recent Documents */}
      <Card variant="elevated">
        <CardHeader>
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
            Recent Documents
          </h2>
        </CardHeader>
        <CardBody>
          <div className="space-y-4">
            {recentDocuments.map((doc) => (
              <div
                key={doc.id}
                className="flex items-center gap-4 p-4 rounded-lg bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              >
                <FileText className="text-gray-400 flex-shrink-0" size={24} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                      {doc.name}
                    </p>
                    <Badge variant="default" size="sm">{doc.type}</Badge>
                  </div>
                  {doc.progress > 0 && doc.progress < 100 ? (
                    <Progress value={doc.progress} size="sm" showLabel />
                  ) : (
                    <p className="text-xs text-gray-600 dark:text-gray-400">
                      {doc.status === 'completed' ? 'Processed successfully' : 'Waiting in queue'}
                    </p>
                  )}
                </div>
                <div className="flex items-center gap-3">
                  {getStatusBadge(doc.status)}
                  <Button variant="ghost" size="sm">
                    View
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </CardBody>
      </Card>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card variant="elevated" className="cursor-pointer hover:shadow-lg transition-shadow">
          <CardBody className="text-center py-8">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary-100 dark:bg-primary-950 mb-4">
              <FileText className="text-primary-600 dark:text-primary-400" size={32} />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
              Upload New Documents
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
              Process PDFs, images, and audio files with AI
            </p>
            <Button variant="primary">
              Get Started
            </Button>
          </CardBody>
        </Card>

        <Card variant="elevated" className="cursor-pointer hover:shadow-lg transition-shadow">
          <CardBody className="text-center py-8">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-secondary-100 dark:bg-secondary-950 mb-4">
              <TrendingUp className="text-secondary-600 dark:text-secondary-400" size={32} />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
              View Analytics
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
              Insights from document analysis
            </p>
            <Button variant="secondary">
              View Reports
            </Button>
          </CardBody>
        </Card>
      </div>
    </div>
  );
}
