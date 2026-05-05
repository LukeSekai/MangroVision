import { useState } from 'react';
import { useAuthStore } from '../stores/authStore';
import './LoginScreen.css';

export default function LoginScreen() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const login = useAuthStore((s) => s.login);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await login(username, password);
    } catch (err) {
      setError(err.message || 'Authentication failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-screen">
      <div className="login-card">
        {/* Left — hero image */}
        <div className="login-hero">
          <img src="/mangrove.jpg" alt="Mangrove forest" className="login-hero-img" />
          <div className="login-hero-overlay" />
          <div className="login-hero-content">
            <h2 className="login-hero-title">MangroVision</h2>
            <p className="login-hero-text">
              AI-powered mangrove planting zone analysis for coastal restoration.
            </p>
          </div>
        </div>

        {/* Right — form */}
        <div className="login-form-side">
          <div className="login-form-inner">
            <div className="login-form-header">
              <div className="login-logo">
                <svg width="36" height="36" viewBox="0 0 32 32" fill="none">
                  <rect width="32" height="32" rx="8" fill="var(--green-700)" />
                  <path d="M16 6c-3.3 0-6 2.7-6 6 0 1.7.7 3.2 1.8 4.3.5.5.8 1.1.8 1.7v1c0 1.1.9 2 2 2h2.8c1.1 0 2-.9 2-2v-1c0-.6.3-1.2.8-1.7C21.3 15.2 22 13.7 22 12c0-3.3-2.7-6-6-6z" fill="#fff" opacity="0.9"/>
                  <path d="M13 22h6v2a1 1 0 0 1-1 1h-4a1 1 0 0 1-1-1v-2z" fill="#fff" opacity="0.6"/>
                </svg>
              </div>
              <h1 className="login-title">Sign In</h1>
              <p className="login-subtitle">Welcome back! Enter your credentials to continue.</p>
            </div>

            <form className="login-form" onSubmit={handleSubmit}>
              <div className="form-group">
                <label className="form-label" htmlFor="login-username">Username</label>
                <input
                  id="login-username"
                  className="form-input"
                  type="text"
                  placeholder="Enter your username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  autoFocus
                  required
                />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="login-password">Password</label>
                <input
                  id="login-password"
                  className="form-input"
                  type="password"
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </div>

              {error && <div className="login-error">{error}</div>}

              <button
                type="submit"
                className="btn btn-primary btn-lg login-submit"
                disabled={loading}
              >
                {loading ? 'Signing in...' : 'Sign In'}
              </button>
            </form>

            <p className="login-footer">
              Default credentials: admin / admin123
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
