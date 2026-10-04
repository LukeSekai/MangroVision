import { create } from 'zustand';
import { useMapStore } from './mapStore';
import { staffAuthRequest } from '../utils/staffAuth';

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
  hydrated: false,
  isAuthenticated: Boolean(initialUser),
  token: initialUser ? 'cookie' : null,
  user: initialUser,

  login: async (username, password) => {
    useMapStore.getState().resetWorkspaceData();
    return staffAuthRequest('login', { username, password });
  },

  verifyLogin: async (code) => {
    const data = await staffAuthRequest('login/verify', { code });
    localStorage.setItem('mv_user', JSON.stringify({
      id: data.user_id,
      full_name: data.full_name,
      role: data.role,
    }));
    sessionStorage.setItem('mv_show_welcome', '1');
    set({
      hydrated: true,
      isAuthenticated: true,
      token: 'cookie',
      user: { id: data.user_id, full_name: data.full_name, role: data.role },
    });
  },

  clearSession: (notice = '') => {
    useMapStore.getState().resetWorkspaceData();
    localStorage.removeItem('mv_user');
    if (notice) sessionStorage.setItem('mv_auth_notice', notice);
    set({ isAuthenticated: false, token: null, user: null, hydrated: true });
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
        set({ isAuthenticated: false, token: null, user: null, hydrated: true });
        return;
      }
      const data = await res.json();
      const user = { id: data.user_id, full_name: data.full_name, role: data.role };
      localStorage.setItem('mv_user', JSON.stringify(user));
      set({ isAuthenticated: true, token: 'cookie', user, hydrated: true });
    } catch {
      // Leave the cached session in place on transient network failures.
      set({ hydrated: true });
    }
  },
}));
