import { Link, useLocation } from 'react-router-dom';
import { mapNavigationState } from '../utils/mapNavigation';

export default function MapLink({ state, ...props }) {
  const location = useLocation();
  return <Link {...props} state={mapNavigationState(location, state)} />;
}
