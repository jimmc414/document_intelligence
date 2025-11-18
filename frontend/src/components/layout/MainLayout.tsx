import { useState } from 'react';
import type { ReactNode } from 'react';
import { Header } from './Header';
import { Sidebar } from './Sidebar';
import { cn } from '../../lib/utils/cn';

export interface MainLayoutProps {
  children: ReactNode;
}

export function MainLayout({ children }: MainLayoutProps) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
      <Header onMenuClick={() => setMobileMenuOpen(!mobileMenuOpen)} />

      <div className="flex">
        {/* Desktop sidebar */}
        <div className="hidden lg:block">
          <Sidebar
            collapsed={sidebarCollapsed}
            onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
          />
        </div>

        {/* Mobile sidebar */}
        {mobileMenuOpen && (
          <>
            {/* Overlay */}
            <div
              className="fixed inset-0 bg-black/50 z-modal lg:hidden"
              onClick={() => setMobileMenuOpen(false)}
            />

            {/* Sidebar */}
            <Sidebar
              className="lg:hidden z-modal animate-slide-in-right"
              collapsed={false}
            />
          </>
        )}

        {/* Main content */}
        <main
          className={cn(
            'flex-1 min-h-[calc(100vh-4rem)]',
            'transition-all duration-base',
            'lg:ml-0',
            !sidebarCollapsed ? 'lg:pl-64' : 'lg:pl-16'
          )}
        >
          <div className="container-custom py-6 sm:py-8">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
