import { useEffect, useState } from 'react';
import { Routes, Route, Navigate, useLocation } from 'react-router-dom';
import AppShell from './components/AppShell';
import MapAnalytics from './pages/MapAnalytics';
import Dashboard from './pages/Dashboard';
import Scheduling from './pages/Scheduling';
import ActivityLog from './pages/ActivityLog';
import PlanterManagement from './pages/PlanterManagement';
import ErodedZoneEditor from './pages/ErodedZoneEditor';
import ImageProcessing from './pages/ImageProcessing';
import Monitoring from './pages/OrganizationMonitoring';
import MonitoringMapWorkspace from './pages/MonitoringMapWorkspace';
import FieldApp from './pages/FieldApp';
import LoginScreen from './components/LoginScreen';
import SplashScreen from './components/SplashScreen';
import { useAuthStore } from './stores/authStore';

export default function App() {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const hydrateSession = useAuthStore((s) => s.hydrateSession);
  const location = useLocation();

  // Brand splash gate — shown on the first entry of a tab session so the
  // user always sees the MangroVision mark before any login or map UI. We
  // read sessionStorage synchronously in the initial state so a single SPA
  // render covers both "first entry" and "already past splash" cases
  // without flashing the splash on every route change.
  const [splashDone, setSplashDone] = useState(() => {
    try {
      return window.sessionStorage.getItem('mv_splash_seen') === '1';
    } catch {
      return true;
    }
  });

  useEffect(() => {
    hydrateSession();
  }, [hydrateSession]);

  if (!splashDone) {
    return <SplashScreen onDone={() => setSplashDone(true)} />;
  }

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
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/scheduling" element={<Scheduling />} />
        <Route path="/activity" element={<ActivityLog />} />
        <Route path="/monitoring" element={<Monitoring />} />
        <Route path="/monitoring/map" element={<MonitoringMapWorkspace />} />
        <Route path="/planters" element={<PlanterManagement />} />
        <Route path="/processing" element={<ImageProcessing />} />
        <Route path="/zones" element={<ErodedZoneEditor />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppShell>
  );
}
