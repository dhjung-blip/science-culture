import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,        // 0.0.0.0 — 컨테이너/네트워크에서 접속 가능
    port: 5173,
    open: false,       // 컨테이너엔 브라우저 없음 (로컬은 localhost:5173 직접 열기)
    watch: { usePolling: true },  // 마운트 볼륨 파일변경 감지 (macOS Docker Desktop)
  },
})

