import { create } from 'zustand';
import { submissionError } from '../utils/formValidation';
import { normalizePlanterCode } from '../utils/planterCode';

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

  login: async (accessCode) => {
    set({ status: 'loading', error: '' });
    try {
      const res = await fetch(`${API}/api/planter-auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ access_code: normalizePlanterCode(accessCode) }),
      });
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        const credentialIssue = Array.isArray(payload.detail) && payload.detail.find((issue) =>
          issue.loc?.includes('access_code') && issue.type === 'credential_error');
        const error = submissionError(payload.detail, 'Planter sign-in is temporarily unavailable. Refresh the page and try again, or contact LGU staff.', res.status);
        if (credentialIssue && res.status < 500 && res.status !== 429) error.message = credentialIssue.msg;
        if (res.status === 429) {
          const seconds = Number(res.headers.get('Retry-After'));
          error.message = seconds > 0 && Number.isFinite(seconds)
            ? `Too many sign-in attempts. Try again in ${Math.ceil(seconds / 60)} ${seconds <= 60 ? 'minute' : 'minutes'}.`
            : 'Too many sign-in attempts. Wait a few minutes and try again.';
        } else if (res.status >= 500) error.message = 'The sign-in service is temporarily unavailable. Try again shortly, or contact LGU staff.';
        throw error;
      }
      const data = await res.json().catch(() => null);
      if (!data?.planter?.id || !data?.planter?.participant_slot) {
        throw new Error('Could not verify your sign-in. Please try again, or contact LGU staff.');
      }
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
    } catch (failure) {
      const error = failure instanceof TypeError ? new Error('Could not reach the server. Check your connection and try again.') : failure;
      set({ status: 'error', error: error.message });
      throw error;
    }
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
