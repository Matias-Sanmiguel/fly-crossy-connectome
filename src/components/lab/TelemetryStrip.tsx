import type { ReactNode } from 'react';

export type TelemetryItem = {
  label: string;
  value: ReactNode;
};

type TelemetryStripProps = {
  heading: string;
  items: readonly TelemetryItem[];
  detail?: ReactNode;
  error?: string | null;
};

export function TelemetryStrip({ heading, items, detail, error }: TelemetryStripProps) {
  return (
    <section className="telemetry-strip" aria-label="Controller telemetry">
      <strong>{heading}</strong>
      <dl>
        {items.map((item) => (
          <div key={item.label}>
            <dt>{item.label}</dt>
            <dd>{item.value}</dd>
          </div>
        ))}
      </dl>
      {detail && <div className="telemetry-detail">{detail}</div>}
      {error && <p className="error" role="alert">{error}</p>}
    </section>
  );
}
