import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { MainLayout } from './components/layout/MainLayout';
import { Dashboard } from './pages/Dashboard';
import { Upload } from './pages/Upload';

function App() {
  return (
    <Router>
      <MainLayout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/documents" element={<div className="text-2xl font-bold text-gray-900 dark:text-gray-100">Documents Page - Coming Soon</div>} />
          <Route path="/analytics" element={<div className="text-2xl font-bold text-gray-900 dark:text-gray-100">Analytics Page - Coming Soon</div>} />
          <Route path="/settings" element={<div className="text-2xl font-bold text-gray-900 dark:text-gray-100">Settings Page - Coming Soon</div>} />
        </Routes>
      </MainLayout>
    </Router>
  );
}

export default App;
