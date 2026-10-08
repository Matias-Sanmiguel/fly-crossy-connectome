import type { IncomingMessage, ServerResponse } from 'node:http';
import { VisitorError } from './visitorStore.ts';
import type { VisitorStore } from './visitorStore.ts';

const BODY_LIMIT = 32_768;
async function readJson(request: IncomingMessage): Promise<Record<string, unknown>> {
  if (!request.headers['content-type']?.startsWith('application/json')) throw new VisitorError('Se requiere JSON.', 415);
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > BODY_LIMIT) throw new VisitorError('La solicitud es demasiado grande.', 413);
    chunks.push(Buffer.from(chunk));
  }
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw Error();
    return value as Record<string, unknown>;
  } catch { throw new VisitorError('El JSON no es válido.'); }
}

export function createVisitorApi(store: VisitorStore) {
  // One local installation: bounded write rate, no public deployment assumptions.
  let writeWindow = Date.now(), writes = 0;
  const send = (response: ServerResponse, status: number, body: unknown) => {
    response.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' });
    response.end(JSON.stringify(body));
  };
  return async (request: IncomingMessage, response: ServerResponse, next: () => void) => {
    const path = request.url?.split('?')[0] ?? '';
    if (!path.startsWith('/api/visitors/')) { next(); return; }
    try {
      if (request.method === 'GET' && path === '/api/visitors/ranking') {
        send(response, 200, { entries: store.ranking() }); return;
      }
      if (request.method !== 'POST') throw new VisitorError('Método no permitido.', 405);
      const origin = request.headers.origin;
      if (origin && origin !== `http://${request.headers.host}` && origin !== `https://${request.headers.host}`) {
        throw new VisitorError('Origen no permitido.', 403);
      }
      if (Date.now() - writeWindow > 60_000) { writeWindow = Date.now(); writes = 0; }
      if (++writes > 60) throw new VisitorError('Demasiadas solicitudes. Probá en un minuto.', 429);
      const value = await readJson(request);
      if (path === '/api/visitors/rounds') {
        send(response, 201, store.start(value.name)); return;
      }
      const match = /^\/api\/visitors\/rounds\/([a-f0-9-]{36})\/finish$/.exec(path);
      if (!match) throw new VisitorError('Ruta no encontrada.', 404);
      send(response, 200, store.finish(match[1]!, value.actions));
    } catch (error) {
      if (error instanceof VisitorError) send(response, error.status, { error: error.message });
      else if (error instanceof Error && /nombre|acciones/.test(error.message)) send(response, 400, { error: error.message });
      else { console.error('Visitor ranking API:', error); send(response, 500, { error: 'No se pudo guardar el resultado. Probá de nuevo.' }); }
    }
  };
}
