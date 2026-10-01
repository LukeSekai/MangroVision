import { useState } from 'react';
import { useAuthStore } from '../stores/authStore';
import Logo from './Logo';
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
                <Logo variant="lockup" size={120} alt="MangroVision" />
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
              Use the account provisioned by your administrator.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
