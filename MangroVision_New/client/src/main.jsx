// The public root opens LIKE without loading staff maps, charts, or styles.
if (window.location.pathname === '/' || window.location.pathname === '/index.html') {
  window.location.replace(`${import.meta.env.BASE_URL}like.html${window.location.search}${window.location.hash}`);
} else {
  void import('./workspace-main.jsx');
}
