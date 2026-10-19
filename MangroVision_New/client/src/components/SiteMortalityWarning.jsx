import './SiteMortalityWarning.css';

export default function SiteMortalityWarning({ warning }) {
  if (warning?.active !== true) return null;
  return <aside className="monitoring-site-mortality-warning" role="note">
    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
      <path d="m12 3 10 18H2L12 3Z" /><path d="M12 9v5m0 3v1" />
    </svg>
    <div><strong>Warning: High mortality</strong>
      <span>{Number(warning.dead_count).toLocaleString()} of {Number(warning.total_planted).toLocaleString()} planted seedlings have recorded deaths ({warning.mortality_rate_pct}% mortality).</span>
      <span>All recorded causes are included. This site is an automatic warning zone at 30% mortality.</span>
    </div>
  </aside>;
}
