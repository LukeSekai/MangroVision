import { Link } from 'react-router-dom';
import './NextActions.css';

export default function NextActions({ actions, compact = false }) {
  if (!actions?.length) return null;
  return <section className={`next-actions${compact ? ' is-compact' : ''}`} aria-label="Next steps">
    <h2>Next steps</h2>
    <div className="next-actions-list">
      {actions.map((action) => action.onClick
        ? <button key={action.label} type="button" onClick={action.onClick} disabled={action.disabled}>
          <strong>{action.label}</strong>{action.description && <span>{action.description}</span>}
        </button>
        : <Link key={action.label} to={action.to} state={action.state} reloadDocument={action.reloadDocument}>
          <strong>{action.label}</strong>{action.description && <span>{action.description}</span>}
        </Link>)}
    </div>
  </section>;
}
