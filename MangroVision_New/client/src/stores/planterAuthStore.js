import { create } from 'zustand';

const API = import.meta.env.VITE_API_BASE || '';

const PLANTER_TOKEN_KEY = 'mv_planter_token';
const PLANTER_USER_KEY = 'mv_planter_user';

function readStoredPlanter() {
  try {
    return JSON.parse(localStorage.getItem(PLANTER_USER_KEY) || 'null');
  } catch {
    return null;
  }
}

export const usePlanterAuthStore = create((set, get) => ({
  token: localStorage.getItem(PLANTER_TOKEN_KEY) || null,
  planter: readStoredPlanter(),
  isAuthenticated: !!localStorage.getItem(PLANTER_TOKEN_KEY),
  status: 'idle', // idle | loading | error
  error: '',

  register: async ({ full_name, username, password, phone, base_label, base_lat, base_lon }) => {
    set({ status: 'loading', error: '' });
    const res = await fetch(`${API}/api/planter-auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        full_name,
        username,
        password,
        phone: phone || '',
        base_label: base_label || '',
        base_lat: base_lat ?? null,
        base_lon: base_lon ?? null,
      }),
    });
    if (!res.ok) {
      const payload = await res.json().catch(() => ({}));
      set({ status: 'error', error: payload.detail || 'Registration failed' });
      throw new Error(payload.detail || 'Registration failed');
    }
    const data = await res.json();
    localStorage.setItem(PLANTER_TOKEN_KEY, data.token);
    localStorage.setItem(PLANTER_USER_KEY, JSON.stringify(data.planter));
    sessionStorage.setItem('mv_field_show_welcome', '1');
    // First-time entry — UI greets with "Welcome", not "Welcome back".
    sessionStorage.setItem('mv_field_welcome_kind', 'register');
    set({
      token: data.token,
      planter: data.planter,
      isAuthenticated: true,
      status: 'idle',
      error: '',
    });
    return data.planter;
  },

  login: async (username, password) => {
    set({ status: 'loading', error: '' });
    const res = await fetch(`${API}/api/planter-auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    if (!res.ok) {
      const payload = await res.json().catch(() => ({}));
      set({ status: 'error', error: payload.detail || 'Login failed' });
      throw new Error(payload.detail || 'Login failed');
    }
    const data = await res.json();
    localStorage.setItem(PLANTER_TOKEN_KEY, data.token);
    localStorage.setItem(PLANTER_USER_KEY, JSON.stringify(data.planter));
    sessionStorage.setItem('mv_field_show_welcome', '1');
    // Returning planter — UI greets with "Welcome back".
    sessionStorage.setItem('mv_field_welcome_kind', 'login');
    set({
      token: data.token,
      planter: data.planter,
      isAuthenticated: true,
      status: 'idle',
      error: '',
    });
    return data.planter;
  },

  logout: async () => {
    const token = get().token;
    if (token) {
      try {
        await fetch(`${API}/api/planter-auth/logout?token=${encodeURIComponent(token)}`, {
          method: 'POST',
        });
      } catch {
        // Ignore network errors on logout — local state still clears.
      }
    }
    localStorage.removeItem(PLANTER_TOKEN_KEY);
    localStorage.removeItem(PLANTER_USER_KEY);
    set({ token: null, planter: null, isAuthenticated: false, status: 'idle', error: '' });
  },

  hydrateSession: async () => {
    const token = get().token;
    if (!token) return;
    try {
      const res = await fetch(
        `${API}/api/planter-auth/session?token=${encodeURIComponent(token)}`,
      );
      if (!res.ok) {
        localStorage.removeItem(PLANTER_TOKEN_KEY);
        localStorage.removeItem(PLANTER_USER_KEY);
        set({ token: null, planter: null, isAuthenticated: false });
        return;
      }
      const data = await res.json();
      localStorage.setItem(PLANTER_USER_KEY, JSON.stringify(data.planter));
      set({ planter: data.planter, isAuthenticated: true });
    } catch {
      // Preserve cached session on transient network issues.
    }
  },

  fetchFieldPoints: async () => {
    const token = get().token;
    if (!token) throw new Error('Not signed in');
    const res = await fetch(
      `${API}/api/planter-auth/me/field-points?token=${encodeURIComponent(token)}`,
    );
    if (!res.ok) {
      const payload = await res.json().catch(() => ({}));
      throw new Error(payload.detail || 'Could not load assigned points');
    }
    const data = await res.json();
    if (data.planter) {
      localStorage.setItem(PLANTER_USER_KEY, JSON.stringify(data.planter));
      set({ planter: data.planter });
    }
    return data.points || [];
  },

  markPointStatus: async (assignmentPointId, status) => {
    const token = get().token;
    if (!token) throw new Error('Not signed in');
    const res = await fetch(
      `${API}/api/planter-auth/me/points/${assignmentPointId}/status?token=${encodeURIComponent(token)}`,
      {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status }),
      },
    );
    if (!res.ok) {
      const payload = await res.json().catch(() => ({}));
      throw new Error(payload.detail || 'Could not update point status');
    }
    return res.json();
  },
}));
