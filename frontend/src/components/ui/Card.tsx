import { forwardRef } from 'react';
import type { HTMLAttributes, ReactNode } from 'react';
import { cn } from '../../lib/utils/cn';

export type CardVariant = 'default' | 'elevated' | 'outline' | 'interactive';

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  variant?: CardVariant;
}

export interface CardHeaderProps extends HTMLAttributes<HTMLDivElement> {
  actions?: ReactNode;
}

export interface CardBodyProps extends HTMLAttributes<HTMLDivElement> {}

export interface CardFooterProps extends HTMLAttributes<HTMLDivElement> {}

const cardVariants: Record<CardVariant, string> = {
  default: 'bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800',
  elevated: 'bg-white dark:bg-gray-900 shadow-md',
  outline: 'bg-transparent border-2 border-gray-300 dark:border-gray-700',
  interactive: cn(
    'bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800',
    'transition-all duration-fast',
    'hover:shadow-lg hover:scale-[1.02]',
    'cursor-pointer'
  ),
};

/**
 * Card component for grouping related content
 *
 * @param variant - Visual style variant (default, elevated, outline, interactive)
 *
 * @example
 * <Card variant="elevated">
 *   <CardHeader>
 *     <h3>Title</h3>
 *   </CardHeader>
 *   <CardBody>
 *     Content here
 *   </CardBody>
 * </Card>
 */
export const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ className, variant = 'default', children, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn('rounded-lg overflow-hidden', cardVariants[variant], className)}
        {...props}
      >
        {children}
      </div>
    );
  }
);

Card.displayName = 'Card';

export const CardHeader = forwardRef<HTMLDivElement, CardHeaderProps>(
  ({ className, actions, children, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          'px-6 py-4 border-b border-gray-200 dark:border-gray-800',
          'flex items-center justify-between',
          className
        )}
        {...props}
      >
        <div className="flex-1">{children}</div>
        {actions && <div className="ml-4 flex items-center gap-2">{actions}</div>}
      </div>
    );
  }
);

CardHeader.displayName = 'CardHeader';

export const CardBody = forwardRef<HTMLDivElement, CardBodyProps>(
  ({ className, children, ...props }, ref) => {
    return (
      <div ref={ref} className={cn('px-6 py-4', className)} {...props}>
        {children}
      </div>
    );
  }
);

CardBody.displayName = 'CardBody';

export const CardFooter = forwardRef<HTMLDivElement, CardFooterProps>(
  ({ className, children, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          'px-6 py-4 border-t border-gray-200 dark:border-gray-800',
          'bg-gray-50 dark:bg-gray-800/50',
          'flex items-center justify-end gap-3',
          className
        )}
        {...props}
      >
        {children}
      </div>
    );
  }
);

CardFooter.displayName = 'CardFooter';
