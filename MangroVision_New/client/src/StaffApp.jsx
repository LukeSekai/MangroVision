import { useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
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
import LoginScreen from './components/LoginScreen';
import AccountSettings from './pages/AccountSettings';
import { useAuthStore } from './stores/authStore';

export default function StaffApp() {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const hydrateSession = useAuthStore((s) => s.hydrateSession);
  const hydrated = useAuthStore((s) => s.hydrated);

  useEffect(() => {
    hydrateSession();
  }, [hydrateSession]);

  if (!hydrated) return <div className="login-screen" role="status">Checking your session…</div>;

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
        <Route path="/account" element={<AccountSettings />} />
        <Route path="/reports" element={<Navigate to="/dashboard" replace />} />
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
