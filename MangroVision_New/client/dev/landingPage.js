// Keep the public website's own HTML entry at a separate, readable URL.
export default function landingPage() {
  const install = (server) => {
    server.middlewares.use((request, response, next) => {
      if (['GET', 'HEAD'].includes(request.method)) {
        request.url = (request.url || '').replace(/^\/landing-page\/?(?=\?|$)/, '/like.html');
      }
      next();
    });
  };
  return {
    name: 'like-landing-page',
    configureServer: install,
    configurePreviewServer: install,
  };
}
