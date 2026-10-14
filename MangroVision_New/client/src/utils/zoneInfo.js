const ZONE_NAMES = {
  eroded: 'Eroded Zone',
  forbidden: 'Forbidden Zone',
  warning: 'Warning Zone',
  site: 'Project Site',
};

export function zoneName(feature, index, kind) {
  const props = feature?.properties || {};
  return [props.name, props.label].map((value) => String(value ?? '').trim()).find(Boolean)
    || `${ZONE_NAMES[kind] || 'Zone'} ${index + 1}`;
}

// Use text nodes for saved names and notes; Leaflet treats plain strings as HTML.
export function createZoneInfo(feature, index, kind, doc = document) {
  const props = feature?.properties || {};
  const card = doc.createElement('div');
  card.className = 'zone-tooltip-inner';
  const addText = (className, value) => {
    const element = doc.createElement('div');
    element.className = className;
    element.textContent = value;
    card.append(element);
  };
  addText('zone-tooltip-title', zoneName(feature, index, kind));
  const description = kind === 'forbidden'
    ? 'Restricted area. No planting allowed.'
    : kind === 'eroded'
      ? 'Eroded area. Planting points here are unavailable.'
      : props.warning_label || 'Check this area before planting.';
  addText('zone-tooltip-detail', description);
  if (kind === 'warning' && props.severity) {
    addText('zone-tooltip-detail', `Severity: ${props.severity}`);
  }
  for (const note of new Set([props.description, props.notes].filter(Boolean))) {
    addText('zone-tooltip-detail', note);
  }
  return card;
}

export function bindZoneInfo(layer, feature, index, kind) {
  const content = () => createZoneInfo(feature, index, kind);
  layer.bindTooltip(content, {
    className: `zone-tooltip zone-tooltip-${kind}`, sticky: true, direction: 'auto', opacity: 1,
  });
  // Tapping a boundary gives touch users the same description.
  layer.bindPopup(content, { className: `zone-tooltip-${kind}`, maxWidth: 260 });
  layer.on('popupopen', () => layer.closeTooltip());
  layer.on('tooltipopen', () => { if (layer.isPopupOpen()) layer.closeTooltip(); });
}
