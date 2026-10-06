import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import LikeWebsite from './LikeWebsite';
import './like.css';

createRoot(document.getElementById('root')).render(
  <StrictMode><LikeWebsite /></StrictMode>,
);
