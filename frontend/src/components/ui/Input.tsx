import { forwardRef, useState } from 'react';
import type { InputHTMLAttributes, ReactNode } from 'react';
import { AlertCircle, Eye, EyeOff, X } from 'lucide-react';
import { cn } from '../../lib/utils/cn';

export type InputSize = 'sm' | 'md' | 'lg';

export interface InputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'size'> {
  label?: string;
  helper?: string;
  error?: string;
  leftAddon?: ReactNode;
  rightAddon?: ReactNode;
  size?: InputSize;
  clearable?: boolean;
  onClear?: () => void;
}

const inputSizes: Record<InputSize, string> = {
  sm: 'h-8 px-3 text-sm',
  md: 'h-10 px-4 text-base',
  lg: 'h-12 px-5 text-lg',
};

/**
 * Input component for form fields
 *
 * @param label - Label text displayed above input
 * @param helper - Helper text displayed below input
 * @param error - Error message (shows red styling)
 * @param leftAddon - Content to display on the left inside input
 * @param rightAddon - Content to display on the right inside input
 * @param size - Size of the input (sm, md, lg)
 * @param clearable - Show clear button when input has value
 * @param onClear - Callback when clear button is clicked
 *
 * @example
 * <Input
 *   label="Email"
 *   type="email"
 *   placeholder="Enter your email"
 *   helper="We'll never share your email"
 * />
 */
export const Input = forwardRef<HTMLInputElement, InputProps>(
  (
    {
      className,
      label,
      helper,
      error,
      leftAddon,
      rightAddon,
      size = 'md',
      type = 'text',
      disabled = false,
      required = false,
      clearable = false,
      onClear,
      id,
      value,
      ...props
    },
    ref
  ) => {
    const [showPassword, setShowPassword] = useState(false);
    const [isFocused, setIsFocused] = useState(false);
    const inputId = id || `input-${Math.random().toString(36).substr(2, 9)}`;
    const helperId = helper ? `${inputId}-helper` : undefined;
    const errorId = error ? `${inputId}-error` : undefined;

    const handleClear = () => {
      if (onClear) {
        onClear();
      }
    };

    const showClearButton = clearable && value && !disabled;
    const isPasswordField = type === 'password';
    const actualType = isPasswordField && showPassword ? 'text' : type;

    return (
      <div className="w-full">
        {label && (
          <label
            htmlFor={inputId}
            className={cn(
              'block text-sm font-medium mb-1.5',
              error ? 'text-error-700 dark:text-error-400' : 'text-gray-700 dark:text-gray-300',
              disabled && 'opacity-60'
            )}
          >
            {label}
            {required && <span className="text-error-500 ml-0.5">*</span>}
          </label>
        )}

        <div className="relative">
          {leftAddon && (
            <div className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500 dark:text-gray-400 pointer-events-none">
              {leftAddon}
            </div>
          )}

          <input
            ref={ref}
            id={inputId}
            type={actualType}
            disabled={disabled}
            value={value}
            required={required}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            aria-describedby={cn(helperId, errorId)}
            aria-invalid={error ? 'true' : undefined}
            aria-required={required ? 'true' : undefined}
            className={cn(
              // Base styles
              'w-full rounded-md border transition-all duration-fast',
              'bg-white dark:bg-gray-900',
              'text-gray-900 dark:text-gray-100',
              'placeholder:text-gray-400 dark:placeholder:text-gray-600',
              // Size
              inputSizes[size],
              // Padding adjustments for addons
              leftAddon && 'pl-10',
              (rightAddon || showClearButton || isPasswordField) && 'pr-10',
              // Border and focus states
              error
                ? 'border-error-300 dark:border-error-800 focus:border-error-500 focus:ring-error-500'
                : isFocused || value
                ? 'border-primary-500 dark:border-primary-600 ring-2 ring-primary-500/20'
                : 'border-gray-300 dark:border-gray-700 hover:border-gray-400 dark:hover:border-gray-600',
              // Disabled
              disabled && 'bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-600 cursor-not-allowed',
              // Focus ring
              'focus:outline-none focus:ring-2',
              className
            )}
            {...props}
          />

          <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-1">
            {showClearButton && (
              <button
                type="button"
                onClick={handleClear}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors p-0.5 rounded hover:bg-gray-100 dark:hover:bg-gray-800"
                aria-label="Clear input"
              >
                <X size={16} />
              </button>
            )}

            {isPasswordField && (
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors p-0.5 rounded hover:bg-gray-100 dark:hover:bg-gray-800"
                aria-label={showPassword ? 'Hide password' : 'Show password'}
              >
                {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            )}

            {rightAddon && !showClearButton && !isPasswordField && (
              <div className="text-gray-500 dark:text-gray-400">{rightAddon}</div>
            )}
          </div>

          {error && (
            <div className="absolute right-3 top-1/2 -translate-y-1/2 text-error-500">
              <AlertCircle size={16} />
            </div>
          )}
        </div>

        {(helper || error) && (
          <div className={cn('mt-1.5 text-sm', error ? 'text-error-600 dark:text-error-400' : 'text-gray-600 dark:text-gray-400')} id={errorId || helperId}>
            {error && (
              <div className="flex items-start gap-1">
                <AlertCircle size={14} className="mt-0.5 flex-shrink-0" />
                <span>{error}</span>
              </div>
            )}
            {!error && helper && <span>{helper}</span>}
          </div>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';
