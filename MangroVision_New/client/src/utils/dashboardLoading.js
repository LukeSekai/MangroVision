export const DASHBOARD_ENDPOINTS = {
  overview: '/api/dashboard/overview',
  operations: '/api/dashboard/operations',
  ecology: '/api/dashboard/ecology',
  sites: '/api/dashboard/sites',
};

// Each tab loads its own report. Record notices use a separate small read.
export function dashboardSectionsForTab(tab) {
  return [tab in DASHBOARD_ENDPOINTS ? tab : 'overview'];
}
