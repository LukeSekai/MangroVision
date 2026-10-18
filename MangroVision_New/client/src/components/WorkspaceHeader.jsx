import { NavLink } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';
import WorkspaceClock from './WorkspaceClock';
import GuideButton from './GuideButton';
import NotificationBell from './NotificationBell';
import './WorkspaceHeader.css';

export default function WorkspaceHeader() {
  const user = useAuthStore((state) => state.user);
  const name = user?.full_name?.trim() || 'Your account';
  return <header className="workspace-header">
    <span className="workspace-header-title">MangroVision workspace</span>
    <div className="workspace-header-actions">
      <WorkspaceClock />
      <GuideButton />
      <NotificationBell />
      <NavLink to="/account" className="workspace-profile" aria-label="Account settings" title={`${name} — Account settings`}>
        <span className="workspace-profile-avatar" aria-hidden="true">{user?.full_name?.trim().charAt(0).toUpperCase() || 'U'}</span>
        <span className="workspace-profile-name">{name}</span>
      </NavLink>
    </div>
  </header>;
}
