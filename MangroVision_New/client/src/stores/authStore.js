import { create } from 'zustand';

const API = import.meta.env.VITE_API_BASE || '';

export const useAuthStore = create((set, get) => ({
  isAuthenticated: !!localStorage.getItem('mv_token'),
  token: localStorage.getItem('mv_token') || null,
  user: JSON.parse(localStorage.getItem('mv_user') || 'null'),

  login: async (username, password) => {
    const res = await fetch(`${API}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Login failed');
    }
    const data = await res.json();
    localStorage.setItem('mv_token', data.token);
    localStorage.setItem('mv_user', JSON.stringify({
      id: data.user_id,
      full_name: data.full_name,
      role: data.role,
    }));
    sessionStorage.setItem('mv_show_welcome', '1');
    set({
      isAuthenticated: true,
      token: data.token,
      user: { id: data.user_id, full_name: data.full_name, role: data.role },
    });
  },

  logout: () => {
    const token = get().token;
    if (token) {
      fetch(`${API}/api/auth/logout?token=${token}`, { method: 'POST' }).catch(() => {});
    }
    localStorage.removeItem('mv_token');
    localStorage.removeItem('mv_user');
    set({ isAuthenticated: false, token: null, user: null });
  },

  hydrateSession: async () => {
    const token = get().token;
    if (!token) return;
    try {
      const res = await fetch(`${API}/api/auth/session?token=${encodeURIComponent(token)}`);
      if (!res.ok) {
        localStorage.removeItem('mv_token');
        localStorage.removeItem('mv_user');
        set({ isAuthenticated: false, token: null, user: null });
        return;
      }
      const data = await res.json();
      const user = { id: data.user_id, full_name: data.full_name, role: data.role };
      localStorage.setItem('mv_user', JSON.stringify(user));
      set({ isAuthenticated: true, user });
    } catch {
      // Leave the cached session in place on transient network failures.
    }
  },
}));
