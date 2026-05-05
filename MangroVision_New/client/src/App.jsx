import { useEffect } from 'react';
import { Routes, Route, Navigate, useLocation } from 'react-router-dom';
import AppShell from './components/AppShell';
import MapAnalytics from './pages/MapAnalytics';
import PlanterManagement from './pages/PlanterManagement';
import ErodedZoneEditor from './pages/ErodedZoneEditor';
import ImageProcessing from './pages/ImageProcessing';
import FieldApp from './pages/FieldApp';
import LoginScreen from './components/LoginScreen';
import { useAuthStore } from './stores/authStore';

export default function App() {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const hydrateSession = useAuthStore((s) => s.hydrateSession);
  const location = useLocation();

  useEffect(() => {
    hydrateSession();
  }, [hydrateSession]);

  // The /field route is the planter mobile workspace and uses its own auth store,
  // so it must bypass the admin login gate entirely.
  if (location.pathname.startsWith('/field')) {
    return <FieldApp />;
  }

  if (!isAuthenticated) {
    return <LoginScreen />;
  }

  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<MapAnalytics />} />
        <Route path="/planters" element={<PlanterManagement />} />
        <Route path="/processing" element={<ImageProcessing />} />
        <Route path="/zones" element={<ErodedZoneEditor />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppShell>
  );
}
