import { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { useProcessingStore } from '../stores/processingStore';
import { PlantingWorkspaceContext, PLANTING_TOOLS } from './plantingWorkspaceContext';
import './PlantingMapWorkspace.css';

function ToolIcon({ type }) {
  return <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {type === 'map' && <><path d="m3 5 6-2 6 2 6-2v16l-6 2-6-2-6 2Z" /><path d="M9 3v16M15 5v16" /></>}
    {type === 'zone' && <><path d="m4 5 14-2 3 13-12 5-6-8Z" /><circle cx="4" cy="5" r="1" /><circle cx="18" cy="3" r="1" /><circle cx="9" cy="21" r="1" /></>}
    {type === 'assign' && <><circle cx="9" cy="7" r="3" /><path d="M3 21v-3a6 6 0 0 1 12 0v3M16 11l2 2 4-4" /></>}
    {type === 'image' && <><rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="8" cy="8" r="1.5" /><path d="m3 17 6-6 4 4 3-3 5 5" /></>}
  </svg>;
}

// Keep the route tree in one stable container. RetainedRoutes preserves each
// tool's forms and pauses its map effects while a different tool is visible.
export default function PlantingMapWorkspace({ children }) {
  const { pathname } = useLocation();
  const active = PLANTING_TOOLS.some((tool) => tool.to === pathname);
  const processing = useProcessingStore((s) => s.processing);
  const [hidden, setHidden] = useState(false);

  return <PlantingWorkspaceContext.Provider value={true}>
    {active && <button type="button"
      className={`panel-toggle ${hidden ? 'panel-toggle-hidden' : ''}`}
      onClick={() => setHidden((value) => !value)}
      aria-expanded={!hidden} aria-controls="planting-map-workspace"
      title={hidden ? 'Show Planting Map panel' : 'Hide Planting Map panel'}>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true" style={{ transform: hidden ? 'rotate(180deg)' : undefined }}><path d="m9 6 6 6-6 6" /></svg>
      <span>{hidden ? 'Show' : 'Hide'}</span>
    </button>}
    <div id="planting-map-workspace"
      className={active ? `floating-panel planting-workspace-panel ${hidden ? 'floating-panel-hidden' : ''}` : 'planting-workspace-full-page'}
      inert={active && hidden}>
      {active && <div className="floating-panel-header planting-workspace-header">
        <h2 className="floating-panel-title">Planting Map</h2>
        <nav className="planting-tool-nav" aria-label="Planting Map tools">
          {PLANTING_TOOLS.map((tool) => {
            const locked = processing && tool.to === '/zones';
            return <NavLink key={tool.to} to={tool.to} end
              className={({ isActive }) => `planting-tool-link ${isActive ? 'active' : ''}`}
              aria-disabled={locked}
              title={locked ? 'Zone Editor is locked while image processing is running' : tool.title}
              onClick={(event) => { if (locked) event.preventDefault(); }}>
              <ToolIcon type={tool.icon} /><span>{tool.label}</span>
            </NavLink>;
          })}
        </nav>
      </div>}
      <div className="planting-workspace-pages">{children}</div>
    </div>
  </PlantingWorkspaceContext.Provider>;
}
