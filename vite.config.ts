import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { visitorPlugin } from './server/visitorPlugin.ts';
export default defineConfig({ plugins: [react(), visitorPlugin()], base: './' });
