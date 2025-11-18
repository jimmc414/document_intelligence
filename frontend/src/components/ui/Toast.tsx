import { useEffect } from 'react';
import type { ReactNode } from 'react';
import { AlertCircle, CheckCircle, Info, X, AlertTriangle } from 'lucide-react';
import { cn } from '../../lib/utils/cn';

export type ToastType = 'success' | 'error' | 'warning' | 'info';

export interface ToastProps {
  id: string;
  type: ToastType;
  message: string;
  description?: string;
  duration?: number;
  action?: { label: string; onClick: () => void };
  onClose: (id: string) => void;
}

const toastIcons: Record<ToastType, ReactNode> = {
  success: <CheckCircle size={20} />,
  error: <AlertCircle size={20} />,
  warning: <AlertTriangle size={20} />,
  info: <Info size={20} />,
};

const toastStyles: Record<ToastType, string> = {
  success: 'bg-success-50 dark:bg-success-950 border-success-200 dark:border-success-800 text-success-900 dark:text-success-100',
  error: 'bg-error-50 dark:bg-error-950 border-error-200 dark:border-error-800 text-error-900 dark:text-error-100',
  warning: 'bg-warning-50 dark:bg-warning-950 border-warning-200 dark:border-warning-800 text-warning-900 dark:text-warning-100',
  info: 'bg-info-50 dark:bg-info-950 border-info-200 dark:border-info-800 text-info-900 dark:text-info-100',
};

const iconStyles: Record<ToastType, string> = {
  success: 'text-success-600 dark:text-success-400',
  error: 'text-error-600 dark:text-error-400',
  warning: 'text-warning-600 dark:text-warning-400',
  info: 'text-info-600 dark:text-info-400',
};

export function Toast({
  id,
  type,
  message,
  description,
  duration = 4000,
  action,
  onClose,
}: ToastProps) {
  useEffect(() => {
    if (duration && duration > 0) {
      const timer = setTimeout(() => {
        onClose(id);
      }, duration);

      return () => clearTimeout(timer);
    }
  }, [id, duration, onClose]);

  return (
    <div
      role="alert"
      className={cn(
        'max-w-md w-full shadow-lg rounded-lg border p-4',
        'animate-slide-in-right',
        toastStyles[type]
      )}
    >
      <div className="flex items-start gap-3">
        <div className={cn('flex-shrink-0', iconStyles[type])}>
          {toastIcons[type]}
        </div>

        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium">{message}</p>
          {description && (
            <p className="mt-1 text-sm opacity-90">{description}</p>
          )}
          {action && (
            <button
              onClick={action.onClick}
              className={cn(
                'mt-2 text-sm font-medium underline hover:no-underline',
                iconStyles[type]
              )}
            >
              {action.label}
            </button>
          )}
        </div>

        <button
          onClick={() => onClose(id)}
          className="flex-shrink-0 text-current opacity-50 hover:opacity-100 transition-opacity"
          aria-label="Close notification"
        >
          <X size={16} />
        </button>
      </div>
    </div>
  );
}

// Toast container component
export function ToastContainer({ children }: { children: ReactNode }) {
  return (
    <div
      className="fixed top-4 right-4 z-toast flex flex-col gap-2 pointer-events-none"
      aria-live="polite"
      aria-atomic="true"
    >
      <div className="pointer-events-auto">{children}</div>
    </div>
  );
}
