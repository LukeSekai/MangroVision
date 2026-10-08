import { create } from 'zustand';
import { submissionError } from '../utils/formValidation';
import { participantDeviceKey, parseParticipantRecoveryCode, rememberParticipantDeviceKey } from '../utils/participantDevice';

const API = import.meta.env.VITE_API_BASE || '';

const PLANTER_USER_KEY = 'mv_planter_user';

try { localStorage.removeItem('mv_planter_token'); } catch { /* Browser storage may be disabled. */ }

function readStoredPlanter() {
  try {
    return JSON.parse(localStorage.getItem(PLANTER_USER_KEY) || 'null');
  } catch {
    return null;
  }
}

function storePlanter(planter) {
  try {
    if (planter) localStorage.setItem(PLANTER_USER_KEY, JSON.stringify(planter));
    else localStorage.removeItem(PLANTER_USER_KEY);
  } catch { /* The server session remains authoritative. */ }
}

function welcome(kind) {
  try {
    sessionStorage.setItem('mv_field_show_welcome', '1');
    sessionStorage.setItem('mv_field_welcome_kind', kind);
  } catch { /* A welcome message is optional when storage is disabled. */ }
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
    let identity;
    try { identity = participantDeviceKey(username); }
    catch (error) { set({ status: 'error', error: error.message }); throw error; }
    const res = await fetch(`${API}/api/planter-auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        full_name,
        username,
        password,
        organization_id,
        participant_count,
        device_key: identity,
        phone: phone || '',
        base_label: base_label || '',
        base_lat: base_lat ?? null,
        base_lon: base_lon ?? null,
      }),
    });
    if (!res.ok) {
      const payload = await res.json().catch(() => ({}));
      const error = submissionError(payload.detail, 'Registration failed. Check the highlighted fields and try again.');
      set({ status: 'error', error: error.message });
      throw error;
    }
    const data = await res.json();
    storePlanter(data.planter);
    welcome('register');
    set({
      token: 'cookie',
      planter: data.planter,
      isAuthenticated: true,
      status: 'idle',
      error: '',
    });
    return data.planter;
  },

  login: async (username, password, participantSlot = null, recoverSlot = false, recoveryCode = '') => {
    set({ status: 'loading', error: '' });
    let identity;
    try {
      const savedIdentity = participantDeviceKey(username);
      identity = recoveryCode ? parseParticipantRecoveryCode(recoveryCode) : savedIdentity;
    } catch (error) { set({ status: 'error', error: error.message }); throw error; }
    const res = await fetch(`${API}/api/planter-auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password, device_key: identity, participant_slot: recoverSlot ? participantSlot : null, recover_slot: recoverSlot, resume_device: Boolean(recoveryCode) }),
    });
    if (!res.ok) {
      const payload = await res.json().catch(() => ({}));
      const error = submissionError(payload.detail, 'Login failed. Check your shared username and password, then try again.');
      set({ status: 'error', error: error.message });
      throw error;
    }
    const data = await res.json();
    rememberParticipantDeviceKey(identity, username);
    storePlanter(data.planter);
    welcome('login');
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
    storePlanter(null);
    set({ token: null, planter: null, isAuthenticated: false, status: 'idle', error: '' });
  },

  hydrateSession: async () => {
    try {
      const res = await fetch(
        `${API}/api/planter-auth/session`,
      );
      if (!res.ok) {
        storePlanter(null);
        set({ token: null, planter: null, isAuthenticated: false });
        return;
      }
      const data = await res.json();
      storePlanter(data.planter);
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
      storePlanter(data.planter);
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
