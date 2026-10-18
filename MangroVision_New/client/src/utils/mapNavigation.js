import { PLANTING_TOOLS } from '../components/plantingWorkspaceContext.js';

const PAGE_LABELS = {
  '/dashboard': 'Dashboard',
  '/scheduling': 'Scheduling',
  '/organizations': 'Organizations',
  '/monitoring': 'Monitoring',
  '/monitoring/replanting': 'LGU Replanting',
  '/activity': 'Activity Logs',
  '/account': 'Account',
};

export function isMapWorkspace(pathname) {
  return pathname === '/monitoring/map' || PLANTING_TOOLS.some((tool) => tool.to === pathname);
}

function validReturnTarget(target) {
  return target && Object.hasOwn(PAGE_LABELS, target.pathname)
    && (!target.search || (typeof target.search === 'string' && target.search.startsWith('?')))
    && (!target.hash || (typeof target.hash === 'string' && target.hash.startsWith('#')));
}

// Router state survives a map refresh. Keep the source's exact URL and state,
// without carrying an older return chain into the next map visit.
export function mapNavigationState(location, state = {}) {
  const { mapReturnTo: _previous, ...sourceState } = location.state || {};
  const target = Object.hasOwn(PAGE_LABELS, location.pathname) ? {
    pathname: location.pathname, search: location.search || '', hash: location.hash || '',
    state: sourceState,
  } : validReturnTarget(location.state?.mapReturnTo) ? location.state.mapReturnTo : undefined;
  return { ...state, mapReturnTo: target };
}

export function mapReturnTarget(location) {
  if (!isMapWorkspace(location.pathname)) return null;
  const target = location.state?.mapReturnTo;
  if (validReturnTarget(target)) return { ...target, label: PAGE_LABELS[target.pathname] };
  return location.pathname === '/monitoring/map'
    ? { pathname: '/monitoring', search: '', hash: '', label: 'Monitoring' }
    : null;
}
