export const DASHBOARD_ENDPOINTS = {
  overview: '/api/dashboard/overview',
  operations: '/api/dashboard/operations',
  ecology: '/api/dashboard/ecology',
  sites: '/api/dashboard/sites',
};

// Each report tab loads its own data.
export function dashboardSectionsForTab(tab) {
  return [tab in DASHBOARD_ENDPOINTS ? tab : 'overview'];
}
