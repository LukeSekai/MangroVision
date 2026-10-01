// Labels retained for older, manually categorized visit records.
export const GROWTH_STAGES = [
  { value: 'seedling', label: 'Seedling', description: 'Up to 1 metre tall, with a thin stem (less than 4 cm across).' },
  { value: 'young', label: 'Young mangrove', description: 'Over 1 metre tall, with a thin trunk (up to 4 cm across).' },
  { value: 'larger', label: 'Larger mangrove', description: 'Over 1 metre tall, with a thicker trunk (more than 4 cm across).' },
  { value: 'mixed', label: 'Mixed sizes', description: 'Different growth stages are present in the organization’s planting areas.' },
  { value: 'not_checked', label: 'Not checked this visit', description: 'Plant size could not be checked. No growth stage will be assumed.' },
];

export function growthStageLabel(value) {
  if (value === 'no_living') return 'No living seedlings';
  return GROWTH_STAGES.find((stage) => stage.value === value)?.label || 'Not recorded';
}
