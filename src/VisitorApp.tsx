import { useCallback, useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { GameScene } from './components/GameScene.tsx';
import { useGame } from './hooks/useGame.ts';
import { createGame, DECISION_SECONDS } from './game/simulation.ts';
import type { Action } from './game/types.ts';
import { asset } from './lib/atlas.ts';
import { finishVisitorRound, loadRanking, startVisitorRound } from './visitor/client.ts';
import { normalizeVisitorName, VISITOR_MAX_STEPS } from './visitor/protocol.ts';
import type { RankingEntry, VisitorRound } from './visitor/protocol.ts';
import './visitor.css';

const CONTROLS: readonly { action: Action; label: string; key: string }[] = [
  { action: 'forward', label: 'Avanzar', key: '↑' },
  { action: 'left', label: 'Izquierda', key: '←' },
  { action: 'backward', label: 'Retroceder', key: '↓' },
  { action: 'right', label: 'Derecha', key: '→' },
  { action: 'wait', label: 'Esperar', key: '−' },
];

function VisitorGame({ round, onSaved, onNext }: {
  round: VisitorRound; onSaved: (entry: RankingEntry) => void; onNext: () => void;
}) {
  const game = useGame({ seed: round.seed, mode: 'human' });
  const stage = useRef<HTMLDivElement>(null);
  const actions = useRef<Action[]>([]);
  const lastStep = useRef(0);
  const invalidTrace = useRef(false);
  const submission = useRef<Promise<void> | null>(null);
  const [save, setSave] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [error, setError] = useState('');
  const [result, setResult] = useState<RankingEntry | null>(null);
  const ended = game.state.terminal !== null || game.state.step >= VISITOR_MAX_STEPS;
  const remaining = Math.max(0, Math.ceil((VISITOR_MAX_STEPS - game.state.step) * DECISION_SECONDS));

  const submit = useCallback(() => {
    if (submission.current) return;
    if (invalidTrace.current) {
      setError('Se perdió parte del registro de la ronda. No se guardará un puntaje incompleto.');
      setSave('error'); return;
    }
    setSave('saving'); setError('');
    submission.current = finishVisitorRound(round.id, actions.current.slice()).then(entry => {
      setResult(entry); setSave('saved'); onSaved(entry);
    }).catch((cause: unknown) => {
      submission.current = null;
      setSave('error'); setError(cause instanceof Error ? cause.message : 'No se pudo guardar la ronda.');
    });
  }, [onSaved, round.id]);

  useEffect(() => { stage.current?.focus(); }, []);
  useEffect(() => {
    if (game.state.step !== lastStep.current) {
      if (game.state.step !== lastStep.current + 1) invalidTrace.current = true;
      actions.current.push(game.state.previousAction);
      lastStep.current = game.state.step;
    }
    if (ended) {
      if (!game.paused) game.onTogglePause();
      submit();
    }
  }, [ended, game.state.step, game.state.previousAction, game.paused, game.onTogglePause, submit]);

  const move = (action: Action) => { if (!ended) game.onAction(action); stage.current?.focus(); };
  return (
    <>
      <div className="visitor-round-heading">
        <div><span>Jugando</span><h2>{round.name}</h2></div>
        <span className="visitor-timer" aria-label={`${remaining} segundos restantes`}>{Math.floor(remaining / 60)}:{String(remaining % 60).padStart(2, '0')}</span>
      </div>
      <div className="visitor-stage" ref={stage} tabIndex={-1} aria-label="Juego, usá WASD o las flechas">
        <GameScene state={game.state} events={game.events} />
        <div className="visitor-score"><span>Puntaje</span><strong>{game.state.score}</strong></div>
        {(ended || game.paused) && <div className="visitor-overlay">
          <section className="visitor-dialog" aria-label={ended ? 'Resultado de la ronda' : 'Juego pausado'}>
            {ended ? <>
              <p className="visitor-eyebrow">{game.state.terminal ? 'Fin de la ronda' : 'Tiempo cumplido'}</p>
              <h2>¡Bien jugado, {round.name}!</h2>
              <p className="visitor-final-score">{result?.score ?? game.state.score}<span> puntos</span></p>
              <p role="status">{save === 'saved' ? 'Tu puntaje ya está en el ranking.' : save === 'saving' ? 'Guardando tu puntaje…' : 'El puntaje todavía no se guardó.'}</p>
              {error && <p className="visitor-error" role="alert">{error}</p>}
              {save === 'error' && !invalidTrace.current && <button type="button" onClick={submit}>Reintentar guardado</button>}
              <button type="button" onClick={onNext} disabled={save === 'saving'}>{save === 'saved' ? 'Siguiente participante' : 'Volver al inicio sin guardar'}</button>
            </> : <>
              <p className="visitor-eyebrow">Pausa</p><h2>Seguimos cuando quieras.</h2>
              <button type="button" onClick={() => { game.onTogglePause(); stage.current?.focus(); }}>Continuar</button>
            </>}
          </section>
        </div>}
      </div>
      <div className="visitor-game-controls">
        <div className="visitor-direction-controls" aria-label="Controles de movimiento">
          {CONTROLS.map(control => <button key={control.action} type="button" aria-label={control.label} disabled={ended || game.paused} onClick={() => move(control.action)}>{control.key}</button>)}
        </div>
        <button className="visitor-pause" type="button" disabled={ended} onClick={() => { game.onTogglePause(); stage.current?.focus(); }}>{game.paused ? 'Continuar' : 'Pausa'}</button>
      </div>
    </>
  );
}

export function VisitorApp() {
  const [round, setRound] = useState<VisitorRound | null>(null);
  const [name, setName] = useState('');
  const [starting, setStarting] = useState(false);
  const startPending = useRef(false);
  const [error, setError] = useState('');
  const [rankingError, setRankingError] = useState('');
  const [entries, setEntries] = useState<RankingEntry[]>([]);
  const [rankingReady, setRankingReady] = useState(false);
  const [lastResult, setLastResult] = useState<RankingEntry | null>(null);
  const [rankingRevision, setRankingRevision] = useState(0);
  const [preview] = useState(() => createGame('visitor-preview'));

  useEffect(() => {
    const abort = new AbortController();
    let timer: number;
    const refresh = async () => {
      try {
        const value = await loadRanking(abort.signal);
        if (!abort.signal.aborted) { setEntries(value.entries); setRankingReady(true); setRankingError(''); }
      } catch (cause: unknown) {
        if (!abort.signal.aborted) setRankingError(cause instanceof Error ? cause.message : 'No se pudo cargar el ranking.');
      } finally { if (!abort.signal.aborted) timer = window.setTimeout(refresh, 5000); }
    };
    void refresh();
    return () => { abort.abort(); window.clearTimeout(timer); };
  }, [rankingRevision]);

  const saved = useCallback((entry: RankingEntry) => {
    setLastResult(entry);
    setRankingRevision(value => value + 1);
  }, []);
  const next = () => { setRound(null); setName(''); setError(''); };
  const start = async (event: FormEvent) => {
    event.preventDefault();
    if (startPending.current) return;
    try {
      const validName = normalizeVisitorName(name);
      startPending.current = true; setStarting(true); setError('');
      setRound(await startVisitorRound(validName));
    } catch (cause: unknown) { setError(cause instanceof Error ? cause.message : 'No se pudo iniciar la ronda.'); }
    finally { startPending.current = false; setStarting(false); }
  };

  return (
    <main className="visitor-page">
      <header className="visitor-header">
        <div><h1>fly<span>-</span>crossy</h1><p>Ahora te toca a vos.</p></div>
        <img src={asset('assets/expo/imas-logo-figma.svg')} alt="imas+ tech club" />
      </header>
      <div className="visitor-layout">
        <section className="visitor-intro" aria-label="Cómo jugar">
          <p className="visitor-eyebrow">Desafío abierto</p><h2>¿Hasta dónde llegás?</h2>
          <p>Cruzá calles, ríos y vías. Un paso en falso y termina tu ronda.</p>
          <div className="visitor-key-guide"><kbd>W</kbd><div><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd></div></div>
          <p>WASD o flechas para moverte.<br />Espacio para esperar.<br />Esc para pausar.</p>
          <p className="visitor-small">Una ronda por turno. Un mapa nuevo cada vez. Hasta 3 minutos para superar tu marca.</p>
        </section>
        <section className="visitor-game-column" aria-label="Partida del visitante">
          {round ? <VisitorGame key={round.id} round={round} onSaved={saved} onNext={next} /> : <>
            <div className="visitor-stage">
              <GameScene state={preview} events={[]} />
              <div className="visitor-overlay">
                <form className="visitor-dialog" onSubmit={start}>
                  <p className="visitor-eyebrow">Listo para cruzar</p><h2>Dejá tu nombre.<br />Hacé tu marca.</h2>
                  <label htmlFor="visitor-name">¿Cómo te llamás?</label>
                  <input id="visitor-name" autoFocus autoComplete="off" maxLength={32} value={name} onChange={event => setName(event.target.value)} placeholder="Tu nombre o apodo" required disabled={starting} aria-describedby="visitor-name-help" />
                  <p id="visitor-name-help" className="visitor-small">Tu nombre y puntaje se mostrarán en el ranking local.</p>
                  {error && <p className="visitor-error" role="alert">{error}</p>}
                  <button type="submit" disabled={starting}>{starting ? 'Preparando ronda…' : 'Jugar una ronda'}</button>
                </form>
              </div>
            </div>
          </>}
          <p className="visitor-control-hint">WASD, flechas o botones para moverte. Espacio para esperar. Esc para pausar.</p>
        </section>
        <aside className="visitor-ranking" aria-label="Ranking de visitantes">
          <p className="visitor-eyebrow">Las mejores rondas</p><h2>Ranking<span> / TOP 10</span></h2>
          {rankingError && <p className="visitor-error" role="alert">{rankingError}</p>}
          {!rankingReady && !rankingError && <p role="status">Cargando ranking…</p>}
          {rankingReady && entries.length === 0 && <p className="visitor-ranking-empty">El primer puesto<br />todavía puede ser tuyo.</p>}
          <ol>
            {entries.map((entry, index) => <li key={entry.id} className={entry.id === lastResult?.id ? 'visitor-ranking-latest' : undefined}>
              <span className="visitor-rank-position">{String(index + 1).padStart(2, '0')}</span><span className="visitor-rank-name">{entry.name}</span><strong>{entry.score}</strong>
            </li>)}
          </ol>
          {lastResult && !entries.some(entry => entry.id === lastResult.id) && <p className="visitor-last-result">Última ronda: {lastResult.name}, {lastResult.score} puntos.</p>}
          <p className="visitor-small">Un puntaje por ronda. En los empates, primero quien lo consiguió antes.</p>
        </aside>
      </div>
      <footer className="visitor-footer">Desarrollado por Nicolás Luca Giordano y Matías Adrián Sanmiguel.</footer>
    </main>
  );
}
