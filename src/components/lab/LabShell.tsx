import type { ReactNode } from 'react';

type LabShellProps = {
  header: ReactNode;
  controls: ReactNode;
  alerts?: ReactNode;
  environment: ReactNode;
  brain: ReactNode;
  telemetry: ReactNode;
  body?: ReactNode;
  secondary?: ReactNode;
  footer?: ReactNode;
};

export function LabShell({
  header,
  controls,
  alerts,
  environment,
  brain,
  telemetry,
  body,
  secondary,
  footer,
}: LabShellProps) {
  return (
    <div className="lab-shell">
      {header}
      <main className="lab-main">
        {controls}
        {alerts && <div className="lab-alerts">{alerts}</div>}
        <div className={`lab-workbench${body ? ' has-body' : ''}`}>
          {environment}
          {brain}
          {telemetry}
          {body}
        </div>
        {secondary}
      </main>
      {footer}
    </div>
  );
}
