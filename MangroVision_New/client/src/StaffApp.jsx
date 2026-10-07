import { useEffect } from 'react';
import { Route, Navigate } from 'react-router-dom';
import AppShell from './components/AppShell';
import RetainedRoutes from './components/RetainedRoutes';
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

const WORKSPACE_PAGES = ['/map', '/dashboard', '/scheduling', '/activity', '/account',
  '/monitoring', '/monitoring/map', '/planters', '/processing', '/zones'];

export default function StaffApp() {
  const userId = useAuthStore((s) => s.user?.id);
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
      <RetainedRoutes key={userId} paths={WORKSPACE_PAGES}>
        <Route path="/" element={<Navigate to="/map" replace />} />
        <Route path="/map" element={<MapAnalytics />} />
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
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </RetainedRoutes>
    </AppShell>
  );
}
