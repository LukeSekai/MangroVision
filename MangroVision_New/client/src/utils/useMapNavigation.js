import { useLocation, useNavigate } from 'react-router-dom';
import { mapNavigationState } from './mapNavigation';

export default function useMapNavigation() {
  const location = useLocation();
  const navigate = useNavigate();
  return (to, options = {}) => navigate(to, {
    ...options, state: mapNavigationState(location, options.state),
  });
}
