type LabHeaderProps = {
  eyebrow: string;
  title: string;
  context: string;
  status: string;
};

export function LabHeader({ eyebrow, title, context, status }: LabHeaderProps) {
  return (
    <header className="lab-header">
      <div className="lab-brand">
        <span className="eyebrow">{eyebrow}</span>
        <h1>{title}</h1>
      </div>
      <span className="header-context">{context}</span>
      <span className="header-runtime" role="status" aria-live="polite">
        <span aria-hidden="true" />{status}
      </span>
      <a className="header-link" href="https://github.com/Matias-Sanmiguel/fly-crossy-connectome">
        GitHub ↗
      </a>
    </header>
  );
}
