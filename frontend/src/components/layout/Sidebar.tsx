import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Upload,
  FileText,
  BarChart3,
  Settings,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { cn } from '../../lib/utils/cn';

export interface SidebarProps {
  collapsed?: boolean;
  onToggleCollapse?: () => void;
  className?: string;
}

const navigationItems = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Upload', href: '/upload', icon: Upload },
  { name: 'Documents', href: '/documents', icon: FileText },
  { name: 'Analytics', href: '/analytics', icon: BarChart3 },
  { name: 'Settings', href: '/settings', icon: Settings },
];

export function Sidebar({ collapsed = false, onToggleCollapse, className }: SidebarProps) {
  return (
    <aside
      className={cn(
        'fixed left-0 top-16 bottom-0 z-sticky',
        'bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800',
        'transition-all duration-base',
        'flex flex-col',
        collapsed ? 'w-16' : 'w-64',
        className
      )}
    >
      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-1 overflow-y-auto scrollbar-thin">
        {navigationItems.map((item) => (
          <NavLink
            key={item.name}
            to={item.href}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 px-3 py-2.5 rounded-md',
                'text-sm font-medium transition-all duration-fast',
                'hover:bg-gray-100 dark:hover:bg-gray-800',
                isActive
                  ? 'bg-primary-50 dark:bg-primary-950 text-primary-700 dark:text-primary-300 border-l-4 border-primary-600'
                  : 'text-gray-700 dark:text-gray-300',
                collapsed && 'justify-center'
              )
            }
            title={collapsed ? item.name : undefined}
          >
            <item.icon size={20} className="flex-shrink-0" />
            {!collapsed && <span>{item.name}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Toggle button */}
      {onToggleCollapse && (
        <div className="p-4 border-t border-gray-200 dark:border-gray-800">
          <button
            onClick={onToggleCollapse}
            className={cn(
              'w-full flex items-center gap-3 px-3 py-2 rounded-md',
              'text-sm font-medium text-gray-700 dark:text-gray-300',
              'hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors',
              collapsed && 'justify-center'
            )}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
            {!collapsed && <span>Collapse</span>}
          </button>
        </div>
      )}
    </aside>
  );
}
