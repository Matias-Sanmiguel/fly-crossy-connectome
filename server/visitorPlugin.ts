import { resolve } from 'node:path';
import type { Plugin } from 'vite';
import { openVisitorStore } from './visitorStore.ts';
import { createVisitorApi } from './visitorApi.ts';

export function visitorPlugin(): Plugin {
  let root = '';
  const attach = (server: { middlewares: { use: (handler: ReturnType<typeof createVisitorApi>) => void }; httpServer: { once: (event: 'close', callback: () => void) => unknown } | null }) => {
    const store = openVisitorStore(process.env.VISITOR_DB_PATH ?? resolve(root, 'data/visitor-ranking.sqlite'));
    server.middlewares.use(createVisitorApi(store));
    server.httpServer?.once('close', () => store.close());
  };
  return {
    name: 'visitor-ranking-sqlite',
    configResolved(config) { root = config.root; },
    configureServer: attach,
    configurePreviewServer: attach,
  };
}
