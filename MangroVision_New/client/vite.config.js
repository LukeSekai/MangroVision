import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'
import localOrthophotoTiles from './dev/localOrthophotoTiles.js'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), localOrthophotoTiles()],
  build: {
    rollupOptions: {
      input: {
        mangrovision: fileURLToPath(new URL('./index.html', import.meta.url)),
        like: fileURLToPath(new URL('./like.html', import.meta.url)),
      },
    },
  },
  server: {
    host: true,
    proxy: {
      // Preserve the public/LAN host so the API can validate same-origin
      // requests from phones. Vite's string shorthand rewrites it to localhost.
      '/api': { target: 'http://localhost:8000', changeOrigin: false },
      '/tiles': 'http://localhost:8000',
    },
    allowedHosts: [
      '.trycloudflare.com',
      '.ngrok-free.app',
      '.ngrok.app',
    ],
  },
})
