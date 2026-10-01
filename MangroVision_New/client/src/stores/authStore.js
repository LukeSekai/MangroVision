import { create } from 'zustand';
import { useMapStore } from './mapStore';

const API = import.meta.env.VITE_API_BASE || '';

// Remove credentials left by pre-cookie releases. Only non-sensitive display
// data remains cached; the server-side HttpOnly cookie is authoritative.
localStorage.removeItem('mv_token');
localStorage.removeItem('mangrovision_token');

function readStoredUser() {
  try {
    return JSON.parse(localStorage.getItem('mv_user') || 'null');
  } catch {
    return null;
  }
}

const initialUser = readStoredUser();

export const useAuthStore = create((set) => ({
  isAuthenticated: Boolean(initialUser),
  token: initialUser ? 'cookie' : null,
  user: initialUser,

  login: async (username, password) => {
    useMapStore.getState().resetWorkspaceData();
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
    localStorage.setItem('mv_user', JSON.stringify({
      id: data.user_id,
      full_name: data.full_name,
      role: data.role,
    }));
    sessionStorage.setItem('mv_show_welcome', '1');
    set({
      isAuthenticated: true,
      token: 'cookie',
      user: { id: data.user_id, full_name: data.full_name, role: data.role },
    });
  },

  logout: () => {
    useMapStore.getState().resetWorkspaceData();
    fetch(`${API}/api/auth/logout`, { method: 'POST' }).catch(() => {});
    localStorage.removeItem('mv_user');
    set({ isAuthenticated: false, token: null, user: null });
  },

  hydrateSession: async () => {
    try {
      const res = await fetch(`${API}/api/auth/session`);
      if (!res.ok) {
        useMapStore.getState().resetWorkspaceData();
        localStorage.removeItem('mv_user');
        set({ isAuthenticated: false, token: null, user: null });
        return;
      }
      const data = await res.json();
      const user = { id: data.user_id, full_name: data.full_name, role: data.role };
      localStorage.setItem('mv_user', JSON.stringify(user));
      set({ isAuthenticated: true, token: 'cookie', user });
    } catch {
      // Leave the cached session in place on transient network failures.
    }
  },
}));
