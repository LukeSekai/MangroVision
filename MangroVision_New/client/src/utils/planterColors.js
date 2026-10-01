// Organization color shared by staff maps, participant maps, and activity reports.
// Callers supply the organization id; the helper name is retained for compatibility.

const PLANTER_PALETTE = [
  { fill: '#2563eb', tint: '#dbeafe' }, // blue
  { fill: '#a21caf', tint: '#fae8ff' }, // fuchsia
  { fill: '#0369a1', tint: '#e0f2fe' }, // sky
  { fill: '#4338ca', tint: '#e0e7ff' }, // indigo
  { fill: '#c026d3', tint: '#f5d0fe' }, // bright fuchsia
  { fill: '#1e40af', tint: '#bfdbfe' }, // royal blue
  { fill: '#86198f', tint: '#f5d0fe' }, // dark fuchsia
  { fill: '#3730a3', tint: '#c7d2fe' }, // dark indigo
  { fill: '#075985', tint: '#bae6fd' }, // dark sky
  { fill: '#1e3a8a', tint: '#bfdbfe' }, // navy
];

const FALLBACK = PLANTER_PALETTE[0];

function paletteEntryForPlanter(planterId) {
  if (planterId === null || planterId === undefined) return FALLBACK;
  const numeric = Number(planterId);
  if (!Number.isFinite(numeric)) return FALLBACK;
  const seed = Math.abs(Math.trunc(numeric));
  // seed=1 → index 0 (blue) so the first planter keeps the historical look.
  const index = (seed - 1 + PLANTER_PALETTE.length) % PLANTER_PALETTE.length;
  return PLANTER_PALETTE[index] || FALLBACK;
}

export function getPlanterColor(planterId) {
  return paletteEntryForPlanter(planterId).fill;
}

// Soft tint of the planter color, suitable for popup badge backgrounds.
export function getPlanterTint(planterId) {
  return paletteEntryForPlanter(planterId).tint;
}
