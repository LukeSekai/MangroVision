import { createContext, useContext } from 'react';

export const PlantingWorkspaceContext = createContext(false);
export const usePlantingWorkspace = () => useContext(PlantingWorkspaceContext);

export const PLANTING_TOOLS = [
  { to: '/map', label: 'Overview', title: 'Planting Map overview', icon: 'map' },
  { to: '/zones', label: 'Zone Editor', title: 'Zone Editor', icon: 'zone' },
  { to: '/planters', label: 'Assign Points', title: 'Point Assignment', icon: 'assign' },
  { to: '/processing', label: 'Analyze Image', title: 'Analyze Image', icon: 'image' },
];
