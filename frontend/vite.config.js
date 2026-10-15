import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// the browser only talks to vite's origin, and vite forwards /api to the model server on the same machine,
// so the backend needs no cors and only port 5173 has to be tunneled
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        // 127.0.0.1, not localhost: uvicorn listens on ipv4 only, and node may try localhost as ipv6 (::1) first
        target: 'http://127.0.0.1:8000',
        // the backend's routes have no /api prefix, so /api/objects has to arrive as /objects
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})