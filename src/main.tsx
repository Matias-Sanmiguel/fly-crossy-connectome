import React from 'react';
import ReactDOM from 'react-dom/client';

import { App } from './App';
import { BiomechanicsDebugApp } from './BiomechanicsDebugApp';
import { ExpoApp } from './ExpoApp';

import './style.css';

const biomechanicsDebug =
  new URLSearchParams(
    window.location.search,
  ).get('biomechanics') === '1';
const expoMode = new URLSearchParams(window.location.search).get('expo') === '1';

ReactDOM
  .createRoot(
    document.getElementById('root')!,
  )
  .render(
    <React.StrictMode>
      {expoMode
        ? <ExpoApp />
        : biomechanicsDebug
        ? <BiomechanicsDebugApp />
        : <App />}
    </React.StrictMode>,
  );
