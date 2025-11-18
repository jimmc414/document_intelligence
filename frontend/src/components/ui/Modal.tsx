import { useEffect } from 'react';
import type { ReactNode } from 'react';
import { X } from 'lucide-react';
import { cn } from '../../lib/utils/cn';

export type ModalSize = 'sm' | 'md' | 'lg' | 'xl' | 'full';

export interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  children: ReactNode;
  size?: ModalSize;
  closeOnOverlayClick?: boolean;
  closeOnEscape?: boolean;
  showCloseButton?: boolean;
  footer?: ReactNode;
}

const modalSizes: Record<ModalSize, string> = {
  sm: 'max-w-md',
  md: 'max-w-lg',
  lg: 'max-w-2xl',
  xl: 'max-w-4xl',
  full: 'max-w-7xl',
};

/**
 * Modal component for dialogs and overlays
 *
 * @param isOpen - Whether modal is visible
 * @param onClose - Callback when modal should close
 * @param title - Modal title
 * @param size - Modal width (sm, md, lg, xl, full)
 * @param closeOnOverlayClick - Close when clicking overlay
 * @param closeOnEscape - Close when pressing Escape
 * @param showCloseButton - Show X button in header
 *
 * @example
 * <Modal isOpen={isOpen} onClose={() => setIsOpen(false)} title="Edit Document">
 *   <p>Modal content</p>
 * </Modal>
 */
export function Modal({
  isOpen,
  onClose,
  title,
  children,
  size = 'md',
  closeOnOverlayClick = true,
  closeOnEscape = true,
  showCloseButton = true,
  footer,
}: ModalProps) {
  useEffect(() => {
    if (!isOpen) return;

    // Lock body scroll
    document.body.style.overflow = 'hidden';

    // Handle escape key
    const handleEscape = (e: KeyboardEvent) => {
      if (closeOnEscape && e.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('keydown', handleEscape);

    return () => {
      document.body.style.overflow = 'unset';
      document.removeEventListener('keydown', handleEscape);
    };
  }, [isOpen, closeOnEscape, onClose]);

  useEffect(() => {
    if (!isOpen) return;

    // Focus trap
    const modal = document.querySelector('[role="dialog"]');
    if (!modal) return;

    const focusableElements = modal.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    const firstElement = focusableElements[0] as HTMLElement;
    const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

    const handleTab = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;

      if (e.shiftKey) {
        if (document.activeElement === firstElement) {
          e.preventDefault();
          lastElement?.focus();
        }
      } else {
        if (document.activeElement === lastElement) {
          e.preventDefault();
          firstElement?.focus();
        }
      }
    };

    modal.addEventListener('keydown', handleTab as any);
    firstElement?.focus();

    return () => {
      modal.removeEventListener('keydown', handleTab as any);
    };
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-modal flex items-center justify-center p-4">
      {/* Overlay */}
      <div
        className="fixed inset-0 bg-black/50 backdrop-blur-sm animate-in fade-in duration-base"
        onClick={closeOnOverlayClick ? onClose : undefined}
        aria-hidden="true"
      />

      {/* Modal */}
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={title ? 'modal-title' : undefined}
        className={cn(
          'relative bg-white dark:bg-gray-900 rounded-lg shadow-xl',
          'w-full animate-in fade-in zoom-in-95 duration-base',
          'max-h-[calc(100vh-2rem)] flex flex-col',
          modalSizes[size]
        )}
      >
        {/* Header */}
        {(title || showCloseButton) && (
          <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-800">
            {title && (
              <h2 id="modal-title" className="text-xl font-semibold text-gray-900 dark:text-gray-100">
                {title}
              </h2>
            )}
            {showCloseButton && (
              <button
                onClick={onClose}
                className={cn(
                  'text-gray-400 hover:text-gray-600 dark:hover:text-gray-300',
                  'transition-colors rounded-md p-1 hover:bg-gray-100 dark:hover:bg-gray-800',
                  !title && 'absolute top-4 right-4'
                )}
                aria-label="Close modal"
              >
                <X size={20} />
              </button>
            )}
          </div>
        )}

        {/* Body */}
        <div className="px-6 py-4 overflow-y-auto scrollbar-thin flex-1">{children}</div>

        {/* Footer */}
        {footer && (
          <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-800/50">
            {footer}
          </div>
        )}
      </div>
    </div>
  );
}
