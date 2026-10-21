// Shared screen sizes for saved points and the Assign Points selection layer.
export const getPointRadius = (zoom, selected = false) => {
  const base = zoom >= 22 ? 3.5 : zoom >= 21 ? 2.4 : zoom >= 20 ? 1.6
    : zoom >= 19 ? 1.05 : zoom >= 18 ? 0.8 : zoom >= 17 ? 0.65 : 0.55;
  return selected ? Math.max(base + 1.2, base * 1.8) : base;
};

export const getPointWeight = (zoom, selected = false) => {
  if (selected) return zoom >= 20 ? 1.8 : 1.2;
  if (zoom >= 22) return 1.1;
  if (zoom >= 21) return 0.75;
  if (zoom >= 20) return 0.45;
  if (zoom >= 19) return 0.2;
  return 0;
};
