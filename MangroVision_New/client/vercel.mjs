// Git deployments validate this export before executing configuration code.
// Literal routing placeholders are resolved from each deployment's public env.
// deploy_testing.py validates and updates the current Quick Tunnel origin.
export const config = {
  framework: 'vite',
  buildCommand: 'npm run build',
  outputDirectory: 'dist',
  routes: [
    {
      src: '^/.*$',
      headers: { 'X-Content-Type-Options': 'nosniff' },
      continue: true,
    },
    {
      src: '^/api(?:/(.*))?$',
      dest: '$MANGROVISION_BACKEND_URL/api/$1',
      env: ['MANGROVISION_BACKEND_URL'],
      headers: { 'Cache-Control': 'private, no-store' },
      respectOriginCacheControl: false,
    },
    {
      src: '^/tiles(?:/(.*))?$',
      dest: '$MANGROVISION_BACKEND_URL/tiles/$1',
      env: ['MANGROVISION_BACKEND_URL'],
    },
    {
      src: '^/monitoring_uploads(?:/(.*))?$',
      dest: '$MANGROVISION_BACKEND_URL/monitoring_uploads/$1',
      env: ['MANGROVISION_BACKEND_URL'],
    },
    { handle: 'filesystem' },
    {
      src: '^/((?!api(?:/|$)|tiles(?:/|$)|monitoring_uploads(?:/|$)|assets(?:/|$)).*)$',
      dest: '/index.html',
    },
  ],
};
