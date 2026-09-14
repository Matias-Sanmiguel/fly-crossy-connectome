import type { Action } from '../game/types';

type GameControlsProps = {
  onAction: (action: Action) => void;
  paused: boolean;
  terminal: boolean;
  onTogglePause: () => void;
};

export function GameControls({ onAction, paused, terminal, onTogglePause }: GameControlsProps) {
  const actionButton = (action: Action, label: string, glyph: string, key: string) => (
    <button
      className={`game-action game-action-${action}`}
      type="button"
      aria-label={label}
      title={`${label} (${key})`}
      disabled={paused || terminal}
      onClick={() => onAction(action)}
    >
      <strong aria-hidden="true">{glyph}</strong>
      <span aria-hidden="true">{key}</span>
    </button>
  );

  return (
    <div className="game-controls" aria-label="Fly movement controls">
      <div className="game-dpad">
        {actionButton('forward', 'Move forward', '↑', 'W')}
        {actionButton('left', 'Move left', '←', 'A')}
        {actionButton('wait', 'Wait one step', '•', 'SPACE')}
        {actionButton('right', 'Move right', '→', 'D')}
        {actionButton('backward', 'Move backward', '↓', 'S')}
      </div>
      <button className="game-pause" type="button" disabled={terminal} onClick={onTogglePause}>
        {paused ? 'Resume' : 'Pause'}
        <span aria-hidden="true">ESC</span>
      </button>
    </div>
  );
}
