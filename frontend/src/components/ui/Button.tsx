import { forwardRef } from 'react';
import type { ButtonHTMLAttributes, ReactNode } from 'react';
import { Loader2 } from 'lucide-react';
import { cn } from '../../lib/utils/cn';

export type ButtonVariant = 'primary' | 'secondary' | 'tertiary' | 'ghost' | 'danger';
export type ButtonSize = 'sm' | 'md' | 'lg';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
  fullWidth?: boolean;
}

const buttonVariants: Record<ButtonVariant, string> = {
  primary: cn(
    'bg-primary-600 text-white shadow-sm',
    'hover:bg-primary-700',
    'active:bg-primary-800',
    'focus:ring-primary-500',
    'disabled:bg-primary-400'
  ),
  secondary: cn(
    'bg-transparent text-primary-700 dark:text-primary-400 border-2 border-primary-600',
    'hover:bg-primary-50 dark:hover:bg-primary-950',
    'active:bg-primary-100 dark:active:bg-primary-900',
    'focus:ring-primary-500'
  ),
  tertiary: cn(
    'bg-transparent text-primary-700 dark:text-primary-400',
    'hover:bg-primary-50 dark:hover:bg-primary-950',
    'active:bg-primary-100 dark:active:bg-primary-900',
    'focus:ring-primary-500'
  ),
  ghost: cn(
    'bg-transparent text-gray-700 dark:text-gray-300',
    'hover:bg-gray-100 dark:hover:bg-gray-800',
    'active:bg-gray-200 dark:active:bg-gray-700',
    'focus:ring-gray-400'
  ),
  danger: cn(
    'bg-error-600 text-white shadow-sm',
    'hover:bg-error-700',
    'active:bg-error-800',
    'focus:ring-error-500',
    'disabled:bg-error-400'
  ),
};

const buttonSizes: Record<ButtonSize, string> = {
  sm: 'h-8 px-3 text-sm gap-1.5',
  md: 'h-10 px-4 text-base gap-2',
  lg: 'h-12 px-6 text-lg gap-2.5',
};

/**
 * Button component for primary user actions
 *
 * @param variant - Visual style variant (primary, secondary, tertiary, ghost, danger)
 * @param size - Size of the button (sm, md, lg)
 * @param disabled - Whether button is disabled
 * @param loading - Show loading spinner
 * @param leftIcon - Icon to display on the left
 * @param rightIcon - Icon to display on the right
 * @param fullWidth - Whether button should take full width
 * @param onClick - Click handler
 *
 * @example
 * <Button variant="primary" size="md" onClick={handleClick}>
 *   Click me
 * </Button>
 */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant = 'primary',
      size = 'md',
      loading = false,
      disabled = false,
      leftIcon,
      rightIcon,
      fullWidth = false,
      children,
      ...props
    },
    ref
  ) => {
    const isDisabled = disabled || loading;

    return (
      <button
        ref={ref}
        disabled={isDisabled}
        className={cn(
          // Base styles
          'inline-flex items-center justify-center',
          'font-medium rounded-md',
          'transition-all duration-fast',
          'focus:outline-none focus:ring-2 focus:ring-offset-2',
          'disabled:opacity-40 disabled:cursor-not-allowed',
          'active:scale-98',
          // Minimum touch target
          'min-w-[44px] min-h-[44px]',
          // Variant styles
          buttonVariants[variant],
          // Size styles
          buttonSizes[size],
          // Full width
          fullWidth && 'w-full',
          className
        )}
        {...props}
      >
        {loading && <Loader2 className="animate-spin" size={size === 'sm' ? 14 : size === 'lg' ? 20 : 16} />}
        {!loading && leftIcon && <span className="inline-flex">{leftIcon}</span>}
        {children && <span>{children}</span>}
        {!loading && rightIcon && <span className="inline-flex">{rightIcon}</span>}
      </button>
    );
  }
);

Button.displayName = 'Button';
