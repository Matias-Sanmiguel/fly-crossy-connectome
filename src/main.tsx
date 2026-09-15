import React from 'react';
import ReactDOM from 'react-dom/client';

import { App } from './App';
import { BiomechanicsDebugApp } from './BiomechanicsDebugApp';

import './style.css';

const biomechanicsDebug =
  new URLSearchParams(
    window.location.search,
  ).get('biomechanics') === '1';

ReactDOM
  .createRoot(
    document.getElementById('root')!,
  )
  .render(
    <React.StrictMode>
      {biomechanicsDebug
        ? <BiomechanicsDebugApp />
        : <App />}
    </React.StrictMode>,
  );