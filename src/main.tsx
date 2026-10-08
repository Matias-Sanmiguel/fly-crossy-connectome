import React from 'react';
import ReactDOM from 'react-dom/client';

import { App } from './App';
import { BiomechanicsDebugApp } from './BiomechanicsDebugApp';
import { CheckpointReplayExpoApp } from './CheckpointReplayExpoApp';
import { ExpoApp } from './ExpoApp';
import { VisitorApp } from './VisitorApp';

import './style.css';

const biomechanicsDebug =
  new URLSearchParams(
    window.location.search,
  ).get('biomechanics') === '1';
const expoMode = new URLSearchParams(window.location.search).get('expo') === '1';
const checkpointReplayMode = new URLSearchParams(window.location.search).get('replay') === '1';
const visitorMode = new URLSearchParams(window.location.search).get('play') === '1';

ReactDOM
  .createRoot(
    document.getElementById('root')!,
  )
  .render(
    <React.StrictMode>
      {expoMode
        ? checkpointReplayMode
          ? <CheckpointReplayExpoApp />
          : <ExpoApp />
        : visitorMode
        ? <VisitorApp />
        : biomechanicsDebug
        ? <BiomechanicsDebugApp />
        : <App />}
    </React.StrictMode>,
  );
