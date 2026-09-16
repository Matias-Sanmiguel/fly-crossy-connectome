import { useEffect, useState } from 'react';

import { loadAtlas, type Atlas } from '../lib/atlas.ts';

export function useAtlas(): { atlas: Atlas | null; error: string | null } {
  const [atlas, setAtlas] = useState<Atlas | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const abort = new AbortController();
    void loadAtlas(abort.signal)
      .then((value) => {
        if (abort.signal.aborted) return;
        setAtlas(value);
        setError(null);
      })
      .catch((reason: unknown) => {
        if (!abort.signal.aborted) {
          setError(reason instanceof Error ? reason.message : String(reason));
        }
      });
    return () => abort.abort();
  }, []);

  return { atlas, error };
}
