import ActivityFeed from '../components/ActivityFeed';
import './ActivityLog.css';

export default function ActivityLog() {
  return <main className="activity-log-page">
    <header className="activity-log-header">
      <div>
        <span className="activity-log-eyebrow">Workspace history</span>
        <h1>Activity Logs</h1>
        <p>Track image processing, planting, scheduling, monitoring, and other actions by LGU staff and organization participants.</p>
      </div>
    </header>
    <ActivityFeed scope="staff" />
  </main>;
}
