import { Link } from 'react-router-dom';
import { PanelCard } from './Panel';

// Keep older workspace imports pointed at the current staff workflow.
export default function ReplantingPanel() {
  return <PanelCard panelKey="needs-replanting" title="LGU Replanting">
    <p>Review dead locations and record replacement seedlings planted by LGU staff.</p>
    <Link to="/monitoring/replanting">Open LGU Replanting →</Link>
  </PanelCard>;
}
