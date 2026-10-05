// Evaluated by Vercel at deployment time. This URL is public, not a credential.
const backend = process.env.MANGROVISION_BACKEND_URL;
if (!backend || !/^https:\/\/[a-z0-9-]+\.trycloudflare\.com\/?$/.test(backend)) {
  throw new Error('Set MANGROVISION_BACKEND_URL to the running laptop Quick Tunnel URL before deploying.');
}
const origin = backend.replace(/\/$/, '');

export const config = {
  framework: 'vite',
  buildCommand: 'npm run build',
  outputDirectory: 'dist',
  rewrites: [
    { source: '/api/:path*', destination: `${origin}/api/:path*` },
    { source: '/tiles/:path*', destination: `${origin}/tiles/:path*` },
    { source: '/monitoring_uploads/:path*', destination: `${origin}/monitoring_uploads/:path*` },
    { source: '/((?!api(?:/|$)|tiles(?:/|$)|monitoring_uploads(?:/|$)|assets(?:/|$)).*)', destination: '/index.html' },
  ],
  headers: [
    { source: '/api/:path*', headers: [{ key: 'Cache-Control', value: 'private, no-store' }] },
    { source: '/:path*', headers: [{ key: 'X-Content-Type-Options', value: 'nosniff' }] },
  ],
};
