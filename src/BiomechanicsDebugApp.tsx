import { GameScene } from './components/GameScene';
import { useSimulationStation } from './hooks/useSimulationStation';

const SERVER_URL =
  'ws://127.0.0.1:8001/api/simulation';

export function BiomechanicsDebugApp() {
  const {
    station,
    transport,
    error,
  } = useSimulationStation({
    seed: 'biomechanics-debug',
    url: SERVER_URL,
    population: 80,
    backendPreference: 'cpu',
    speed: 1,
  });

  const pending = station.pending;

  return (
    <>
      <header className="site-header">
        <div className="brand">
          <span className="eyebrow">
            Native biomechanics smoke test
          </span>

          <h1>Fly Crossy · Physical Gate</h1>
        </div>

        <span className="header-context">
          MuJoCo FlyBody → physical keyboard → game
        </span>
      </header>

      <main>
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}

        <div
          className="toolbar"
          aria-label="Biomechanics debug status"
        >
          <span
            className="status"
            role="status"
            aria-live="polite"
          >
            Transport: {transport.phase}
            {' · '}
            Station: {station.phase}
            {' · '}
            Step: {station.game.step}
          </span>
        </div>

        <div className="workbench">
          <section className="panel environment-panel">
            <h2>
              01 / CROSSING ENVIRONMENT
              <span>Physical confirmation only</span>
            </h2>

            <GameScene
              state={station.game}
              events={[...station.events]}
            />

            <div className="panel-bottom">
              <span>
                Score {station.game.score}
              </span>

              <span>
                Step {station.game.step}
              </span>
            </div>
          </section>

          <section className="panel">
            <h2>
              02 / PHYSICAL GATE
              <span>MuJoCo</span>
            </h2>

            <div style={{
              padding: '1.25rem',
              display: 'grid',
              gap: '1rem',
            }}>
              <div>
                <strong>Requested action</strong>
                <div style={{ fontSize: '1.5rem' }}>
                  {pending?.requestedAction ?? '—'}
                </div>
              </div>

              <div>
                <strong>Requested key</strong>
                <div style={{ fontSize: '1.5rem' }}>
                  {pending?.requestedKey ?? '—'}
                </div>
              </div>

              <div>
                <strong>Last applied action</strong>
                <div style={{ fontSize: '1.5rem' }}>
                  {station.lastAppliedAction ?? '—'}
                </div>
              </div>

              <div>
                <strong>Transport</strong>
                <div>
                  {transport.phase}
                </div>
              </div>

              <div>
                <strong>Station</strong>
                <div>
                  {station.phase}
                </div>
              </div>

              <p>
                An intention does not move Crossy.
                The game advances only after the
                biomechanical world returns a physical
                action result.
              </p>
            </div>
          </section>
        </div>
      </main>
    </>
  );
}