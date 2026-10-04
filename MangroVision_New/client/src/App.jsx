import { lazy, Suspense, useState } from 'react';
import { useLocation } from 'react-router-dom';
import SplashScreen from './components/SplashScreen';

const FieldApp = lazy(() => import('./pages/FieldApp'));
const StaffApp = lazy(() => import('./StaffApp'));

export default function App() {
  const location = useLocation();
  const [splashDone, setSplashDone] = useState(() => {
    try {
      return window.sessionStorage.getItem('mv_splash_seen') === '1';
    } catch {
      return true;
    }
  });
  if (!splashDone) return <SplashScreen onDone={() => setSplashDone(true)} />;

  // Phones only load the field workspace; LGU maps and charts load separately.
  return (
    <Suspense fallback={
      <div role="status" style={{ position: 'fixed', inset: 0, display: 'grid', placeItems: 'center', background: '#f1f5f9', color: '#166534' }}>
        Loading workspace…
      </div>
    }>
      {location.pathname.startsWith('/field') ? <FieldApp /> : <StaffApp />}
    </Suspense>
  );
}
