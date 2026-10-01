import { create } from 'zustand';

const API = import.meta.env.VITE_API_BASE || '';

const PLANTER_USER_KEY = 'mv_planter_user';

localStorage.removeItem('mv_planter_token');

function readStoredPlanter() {
  try {
    return JSON.parse(localStorage.getItem(PLANTER_USER_KEY) || 'null');
  } catch {
    return null;
  }
}

// This random identity survives logout so the same device resumes its own slot.
function deviceKey() {
  let key = localStorage.getItem('mv_participant_device');
  if (!key) {
    key = Array.from(crypto.getRandomValues(new Uint8Array(24)), (byte) => byte.toString(16).padStart(2, '0')).join('');
    localStorage.setItem('mv_participant_device', key);
  }
  return key;
}

const initialPlanter = readStoredPlanter();

export const usePlanterAuthStore = create((set, get) => ({
  token: initialPlanter ? 'cookie' : null,
  planter: initialPlanter,
  isAuthenticated: Boolean(initialPlanter),
  status: 'idle', // idle | loading | error
  error: '',

  register: async ({ full_name, username, password, organization_id, participant_count, phone, base_label, base_lat, base_lon }) => {
    set({ status: 'loading', error: '' });
    const res = await fetch(`${API}/api/planter-auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        full_name,
        username,
        password,
        organization_id,
        participant_count,
        device_key: deviceKey(),
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
    localStorage.setItem(PLANTER_USER_KEY, JSON.stringify(data.planter));
    sessionStorage.setItem('mv_field_show_welcome', '1');
    // First-time entry — UI greets with "Welcome", not "Welcome back".
    sessionStorage.setItem('mv_field_welcome_kind', 'register');
    set({
      token: 'cookie',
      planter: data.planter,
      isAuthenticated: true,
      status: 'idle',
      error: '',
    });
    return data.planter;
  },

  login: async (username, password, participantSlot = null, recoverSlot = false) => {
    set({ status: 'loading', error: '' });
    const res = await fetch(`${API}/api/planter-auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password, device_key: deviceKey(), participant_slot: recoverSlot ? participantSlot : null, recover_slot: recoverSlot }),
    });
    if (!res.ok) {
      const payload = await res.json().catch(() => ({}));
      set({ status: 'error', error: payload.detail || 'Login failed' });
      throw new Error(payload.detail || 'Login failed');
    }
    const data = await res.json();
    localStorage.setItem(PLANTER_USER_KEY, JSON.stringify(data.planter));
    sessionStorage.setItem('mv_field_show_welcome', '1');
    // Returning planter — UI greets with "Welcome back".
    sessionStorage.setItem('mv_field_welcome_kind', 'login');
    set({
      token: 'cookie',
      planter: data.planter,
      isAuthenticated: true,
      status: 'idle',
      error: '',
    });
    return data.planter;
  },

  logout: async () => {
    if (get().isAuthenticated) {
      try {
        await fetch(`${API}/api/planter-auth/logout`, {
          method: 'POST',
        });
      } catch {
        // Ignore network errors on logout — local state still clears.
      }
    }
    localStorage.removeItem(PLANTER_USER_KEY);
    set({ token: null, planter: null, isAuthenticated: false, status: 'idle', error: '' });
  },

  hydrateSession: async () => {
    try {
      const res = await fetch(
        `${API}/api/planter-auth/session`,
      );
      if (!res.ok) {
        localStorage.removeItem(PLANTER_USER_KEY);
        set({ token: null, planter: null, isAuthenticated: false });
        return;
      }
      const data = await res.json();
      localStorage.setItem(PLANTER_USER_KEY, JSON.stringify(data.planter));
      set({ token: 'cookie', planter: data.planter, isAuthenticated: true });
    } catch {
      // Preserve cached session on transient network issues.
    }
  },

  fetchFieldPoints: async () => {
    if (!get().isAuthenticated) throw new Error('Not signed in');
    const res = await fetch(
      `${API}/api/planter-auth/me/field-points`,
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
    const projectSitePayload = data.project_sites;
    const projectSites = Array.isArray(projectSitePayload)
      ? projectSitePayload
      : (projectSitePayload?.features || []);
    return {
      points: data.points || [],
      projectSites,
    };
  },

  markPointStatus: async (assignmentPointId, status, skipReason = null) => {
    if (!get().isAuthenticated) throw new Error('Not signed in');
    const res = await fetch(
      `${API}/api/planter-auth/me/points/${assignmentPointId}/status`,
      {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status, skip_reason: status === 'skipped' ? skipReason : null }),
      },
    );
    if (!res.ok) {
      const payload = await res.json().catch(() => ({}));
      throw new Error(payload.detail || 'Could not update point status');
    }
    return res.json();
  },

  markAllPointsCompleted: async (assignmentPointIds) => {
    if (!get().isAuthenticated) throw new Error('Not signed in');
    const res = await fetch(
      `${API}/api/planter-auth/me/points/mark-all-completed`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ assignment_point_ids: assignmentPointIds || [] }),
      },
    );
    if (!res.ok) {
      const payload = await res.json().catch(() => ({}));
      throw new Error(payload.detail || 'Could not mark all points completed');
    }
    return res.json();
  },
}));
