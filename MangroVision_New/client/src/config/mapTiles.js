const DEFAULT_TILE_SERVER = '/tiles';
const DEFAULT_TILESET_PATH = 'TRIAL MAP';
const DEFAULT_MAX_NATIVE_ZOOM = 20;
const DEFAULT_TILE_EXTENSION = 'png';

export const TILE_SERVER = (import.meta.env.VITE_TILE_SERVER || DEFAULT_TILE_SERVER).replace(/\/+$/, '');

const rawTilesetPath = import.meta.env.VITE_TILESET_PATH || DEFAULT_TILESET_PATH;

export const TILESET_PATH = rawTilesetPath
  .split('/')
  .filter(Boolean)
  .map((segment) => encodeURIComponent(segment))
  .join('/');

const rawTileExtension = import.meta.env.VITE_TILE_EXTENSION || DEFAULT_TILE_EXTENSION;
export const TILE_EXTENSION = rawTileExtension.replace(/^\./, '').toLowerCase();

export const ORTHOPHOTO_TILE_URL = `${TILE_SERVER}/${TILESET_PATH}/{z}/{x}/{y}.${TILE_EXTENSION}`;

export const ORTHOPHOTO_MAX_NATIVE_ZOOM = Number.parseInt(
  import.meta.env.VITE_ORTHOPHOTO_MAX_NATIVE_ZOOM || DEFAULT_MAX_NATIVE_ZOOM,
  10,
);
