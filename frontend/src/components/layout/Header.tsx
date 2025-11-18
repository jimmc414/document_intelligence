import { Moon, Sun, Menu, FileText } from 'lucide-react';
import { useTheme } from '../../lib/hooks/useTheme';
import { Button } from '../ui/Button';

export interface HeaderProps {
  onMenuClick?: () => void;
  showMenuButton?: boolean;
}

export function Header({ onMenuClick, showMenuButton = true }: HeaderProps) {
  const { theme, setTheme } = useTheme();

  const toggleTheme = () => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
  };

  return (
    <header className="sticky top-0 z-sticky bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 shadow-sm">
      <div className="h-16 px-4 sm:px-6 lg:px-8 flex items-center justify-between gap-4">
        {/* Left section */}
        <div className="flex items-center gap-4">
          {showMenuButton && (
            <button
              onClick={onMenuClick}
              className="lg:hidden text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 transition-colors p-2 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800"
              aria-label="Toggle menu"
            >
              <Menu size={24} />
            </button>
          )}

          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-gradient-to-br from-primary-600 to-secondary-600">
              <FileText size={24} className="text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900 dark:text-gray-100">
                Document Intelligence
              </h1>
              <p className="hidden sm:block text-xs text-gray-600 dark:text-gray-400">
                AI-Powered Document Analysis
              </p>
            </div>
          </div>
        </div>

        {/* Right section */}
        <div className="flex items-center gap-2">
          <button
            onClick={toggleTheme}
            className="p-2 rounded-md text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            aria-label="Toggle theme"
          >
            {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
          </button>

          <div className="hidden sm:flex items-center gap-2 ml-4">
            <Button variant="ghost" size="sm">
              Documentation
            </Button>
            <Button variant="primary" size="sm">
              Upload Documents
            </Button>
          </div>
        </div>
      </div>
    </header>
  );
}
