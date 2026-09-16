import type { ReactNode } from 'react';

type PanelFrameProps = {
  title: string;
  badge?: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
  className?: string;
};

export function PanelFrame({ title, badge, children, footer, className = '' }: PanelFrameProps) {
  return (
    <section className={`panel ${className}`.trim()}>
      <h2>{title}{badge && <span>{badge}</span>}</h2>
      <div className="panel-content">{children}</div>
      {footer && <div className="panel-bottom">{footer}</div>}
    </section>
  );
}
