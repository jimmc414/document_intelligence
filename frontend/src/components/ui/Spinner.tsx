import { forwardRef } from 'react';
import type { HTMLAttributes } from 'react';
import { Loader2 } from 'lucide-react';
import { cn } from '../../lib/utils/cn';

export type SpinnerSize = 'sm' | 'md' | 'lg';

export interface SpinnerProps extends HTMLAttributes<HTMLDivElement> {
  size?: SpinnerSize;
  label?: string;
}

const spinnerSizes: Record<SpinnerSize, number> = {
  sm: 16,
  md: 24,
  lg: 32,
};

export const Spinner = forwardRef<HTMLDivElement, SpinnerProps>(
  ({ className, size = 'md', label, ...props }, ref) => {
    return (
      <div
        ref={ref}
        role="status"
        aria-label={label || 'Loading'}
        className={cn('inline-flex items-center justify-center', className)}
        {...props}
      >
        <Loader2
          className="animate-spin text-primary-600 dark:text-primary-400"
          size={spinnerSizes[size]}
        />
        {label && <span className="sr-only">{label}</span>}
      </div>
    );
  }
);

Spinner.displayName = 'Spinner';
