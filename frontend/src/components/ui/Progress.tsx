import { forwardRef } from 'react';
import type { HTMLAttributes } from 'react';
import { cn } from '../../lib/utils/cn';

export type ProgressSize = 'sm' | 'md' | 'lg';
export type ProgressVariant = 'default' | 'success' | 'warning' | 'error';

export interface ProgressProps extends HTMLAttributes<HTMLDivElement> {
  value?: number;
  max?: number;
  size?: ProgressSize;
  variant?: ProgressVariant;
  indeterminate?: boolean;
  showLabel?: boolean;
}

const progressSizes: Record<ProgressSize, string> = {
  sm: 'h-1',
  md: 'h-2',
  lg: 'h-3',
};

const progressVariants: Record<ProgressVariant, string> = {
  default: 'bg-primary-600 dark:bg-primary-500',
  success: 'bg-success-600 dark:bg-success-500',
  warning: 'bg-warning-600 dark:bg-warning-500',
  error: 'bg-error-600 dark:bg-error-500',
};

export const Progress = forwardRef<HTMLDivElement, ProgressProps>(
  (
    {
      className,
      value = 0,
      max = 100,
      size = 'md',
      variant = 'default',
      indeterminate = false,
      showLabel = false,
      ...props
    },
    ref
  ) => {
    const percentage = Math.min(Math.max((value / max) * 100, 0), 100);

    return (
      <div className="w-full">
        <div
          ref={ref}
          role="progressbar"
          aria-valuenow={indeterminate ? undefined : value}
          aria-valuemin={0}
          aria-valuemax={max}
          className={cn(
            'w-full bg-gray-200 dark:bg-gray-800 rounded-full overflow-hidden',
            progressSizes[size],
            className
          )}
          {...props}
        >
          {indeterminate ? (
            <div
              className={cn(
                'h-full rounded-full animate-pulse',
                progressVariants[variant]
              )}
              style={{
                width: '30%',
                animation: 'progress-indeterminate 1.5s ease-in-out infinite',
              }}
            />
          ) : (
            <div
              className={cn(
                'h-full rounded-full transition-all duration-300 ease-out',
                progressVariants[variant]
              )}
              style={{ width: `${percentage}%` }}
            />
          )}
        </div>
        {showLabel && !indeterminate && (
          <div className="mt-1 text-sm text-gray-600 dark:text-gray-400 text-right">
            {Math.round(percentage)}%
          </div>
        )}
      </div>
    );
  }
);

Progress.displayName = 'Progress';
