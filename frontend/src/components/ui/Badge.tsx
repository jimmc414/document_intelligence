import { forwardRef } from 'react';
import type { HTMLAttributes } from 'react';
import { cn } from '../../lib/utils/cn';

export type BadgeVariant = 'default' | 'primary' | 'success' | 'warning' | 'error' | 'info';
export type BadgeSize = 'sm' | 'md' | 'lg';

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
  size?: BadgeSize;
  dot?: boolean;
}

const badgeVariants: Record<BadgeVariant, string> = {
  default: 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300',
  primary: 'bg-primary-100 dark:bg-primary-950 text-primary-700 dark:text-primary-300',
  success: 'bg-success-100 dark:bg-success-950 text-success-700 dark:text-success-300',
  warning: 'bg-warning-100 dark:bg-warning-950 text-warning-700 dark:text-warning-300',
  error: 'bg-error-100 dark:bg-error-950 text-error-700 dark:text-error-300',
  info: 'bg-info-100 dark:bg-info-950 text-info-700 dark:text-info-300',
};

const badgeSizes: Record<BadgeSize, string> = {
  sm: 'text-xs px-2 py-0.5',
  md: 'text-sm px-2.5 py-1',
  lg: 'text-base px-3 py-1.5',
};

const dotColors: Record<BadgeVariant, string> = {
  default: 'bg-gray-400',
  primary: 'bg-primary-500',
  success: 'bg-success-500',
  warning: 'bg-warning-500',
  error: 'bg-error-500',
  info: 'bg-info-500',
};

export const Badge = forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className, variant = 'default', size = 'md', dot = false, children, ...props }, ref) => {
    return (
      <span
        ref={ref}
        className={cn(
          'inline-flex items-center gap-1.5 font-medium rounded-full',
          badgeVariants[variant],
          badgeSizes[size],
          className
        )}
        {...props}
      >
        {dot && <span className={cn('w-1.5 h-1.5 rounded-full', dotColors[variant])} />}
        {children}
      </span>
    );
  }
);

Badge.displayName = 'Badge';
