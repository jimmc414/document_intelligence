# Document Intelligence - Frontend

A modern, accessible, and beautiful React + TypeScript frontend for the Document Intelligence AI platform.

## Features

### Design System
- **Custom Design Tokens**: Comprehensive color palette with semantic colors
- **Responsive Design**: Mobile-first approach with breakpoints for all devices
- **Dark Mode**: Full dark mode support with smooth transitions
- **Accessibility**: WCAG AA compliant with keyboard navigation and screen reader support

### UI Components
- ✅ **Button** - Multiple variants (primary, secondary, tertiary, ghost, danger)
- ✅ **Input** - With validation, password toggle, and clear functionality
- ✅ **Card** - Flexible card layouts for content organization
- ✅ **Modal** - Accessible dialogs with focus management
- ✅ **Toast** - Non-intrusive notifications
- ✅ **Badge** - Status indicators
- ✅ **Spinner** - Loading states
- ✅ **Progress** - Progress indicators for uploads and processing
- ✅ **FileUpload** - Drag-and-drop file upload with progress tracking

### Pages
- **Dashboard**: Overview of document processing metrics and recent activity
- **Upload**: Drag-and-drop interface for uploading PDFs, images, and audio files
- **Documents**: Browse and manage processed documents (coming soon)
- **Analytics**: Insights and visualizations (coming soon)
- **Settings**: Application configuration (coming soon)

## Tech Stack

- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS with custom design tokens
- **State Management**: Zustand (UI state) + React Query (server state)
- **Routing**: React Router v6
- **UI Primitives**: Radix UI
- **Icons**: Lucide React
- **Animations**: Framer Motion
- **File Upload**: React Dropzone
- **Form Handling**: React Hook Form + Zod validation

## Getting Started

### Prerequisites
- Node.js 18+ and npm

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

The application will be available at `http://localhost:5173`

## Project Structure

```
src/
├── components/
│   ├── ui/              # Reusable UI components
│   ├── features/        # Feature-specific components
│   └── layout/          # Layout components (Header, Sidebar)
├── pages/               # Page components
├── lib/
│   ├── hooks/          # Custom React hooks
│   ├── utils/          # Utility functions
│   └── stores/         # State management
├── types/              # TypeScript type definitions
└── styles/             # Global styles
```

## Design Principles

### Intuitive UX
- **Predictability**: Consistent interaction patterns
- **Feedback**: Immediate visual feedback for all actions (<100ms)
- **Error Handling**: Clear error messages with actionable guidance
- **Progressive Disclosure**: Show most common options first

### Visual Excellence
- **Typography**: Inter for UI, JetBrains Mono for code/data
- **Color System**: Indigo primary, purple secondary, semantic colors
- **Spacing**: 8px grid system for consistent spacing
- **Animations**: Smooth transitions with reduced-motion support

### Accessibility
- **Keyboard Navigation**: Full keyboard support for all interactions
- **Focus Management**: Visible focus indicators
- **Screen Readers**: Proper ARIA labels and semantic HTML
- **Color Contrast**: WCAG AA compliant contrast ratios

## Performance

- **Code Splitting**: Route-based code splitting
- **Lazy Loading**: Async component loading
- **Optimized Images**: WebP format with lazy loading
- **Bundle Size**: <200KB gzipped initial load

## Browser Support

- Chrome/Edge (last 2 versions)
- Firefox (last 2 versions)
- Safari (last 2 versions)
- Mobile Safari
- Mobile Chrome

## Contributing

This is a custom UI component library built specifically for the Document Intelligence platform. All components follow strict accessibility, performance, and design guidelines.

## License

See parent repository LICENSE file.
