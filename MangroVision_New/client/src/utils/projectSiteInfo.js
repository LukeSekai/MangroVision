import { countMapPointStatuses } from './mapPointStats.js';
import { POINT_STATUS_COLORS } from './pointStatus.js';

export function projectSiteId(feature) {
  return feature?.id ?? feature?.properties?.id;
}

// Staff points use their resolved site ownership. Field points use the site
// of the participant's assignment. Never infer ownership from an organization.
export function projectSitePointCounts(points, { scope = 'site' } = {}) {
  const groups = new Map();
  const seen = new Map();
  for (const point of points) {
    const id = scope === 'participant'
      ? point.project_site_id
      : (point.source_project_site_id ?? point.source_site_id);
    if (id == null || String(id).trim() === '') continue;
    const key = String(id);
    if (!groups.has(key)) {
      groups.set(key, []);
      seen.set(key, new Set());
    }
    const pointId = point.planting_point_id ?? point.id;
    if (pointId != null) {
      if (seen.get(key).has(String(pointId))) continue;
      seen.get(key).add(String(pointId));
    }
    groups.get(key).push(point);
  }
  return new Map([...groups].map(([id, rows]) => [id, countMapPointStatuses(rows)]));
}

export function emptyProjectSiteCounts() {
  return countMapPointStatuses([]);
}

const EARTH_RADIUS_M = 6371008.8;
const RADIANS = Math.PI / 180;

function ringArea(ring) {
  if (!Array.isArray(ring) || ring.length < 3
    || ring.some((point) => !Array.isArray(point)
      || !Number.isFinite(point[0]) || !Number.isFinite(point[1]))) return null;
  // Local tangent-plane area is sufficient for the park's small boundaries.
  // Subtract an origin first to keep small polygons numerically stable.
  const [originLon, originLat] = ring[0];
  const longitudeScale = EARTH_RADIUS_M * RADIANS * Math.cos(originLat * RADIANS);
  const latitudeScale = EARTH_RADIUS_M * RADIANS;
  let area = 0;
  for (let index = 0; index < ring.length; index += 1) {
    const current = ring[index];
    const next = ring[(index + 1) % ring.length];
    area += (current[0] - originLon) * longitudeScale * (next[1] - originLat) * latitudeScale
      - (next[0] - originLon) * longitudeScale * (current[1] - originLat) * latitudeScale;
  }
  return Math.abs(area) / 2;
}

export function projectSiteAreaM2(geometry) {
  const polygons = geometry?.type === 'Polygon' ? [geometry.coordinates]
    : geometry?.type === 'MultiPolygon' ? geometry.coordinates : null;
  if (!Array.isArray(polygons) || !polygons.length) return null;
  let total = 0;
  for (const polygon of polygons) {
    if (!Array.isArray(polygon) || !polygon.length) return null;
    const areas = polygon.map(ringArea);
    if (areas.some((area) => area == null)) return null;
    total += Math.max(0, areas[0] - areas.slice(1).reduce((sum, area) => sum + area, 0));
  }
  return total;
}

const number = (value) => new Intl.NumberFormat('en-PH', { maximumFractionDigits: 2 }).format(value);

function addText(parent, tag, className, value) {
  const element = parent.ownerDocument.createElement(tag);
  element.className = className;
  element.textContent = value;
  parent.append(element);
  return element;
}

function addStatusDonut(parent, counts, statuses, scope) {
  const doc = parent.ownerDocument;
  const chart = addText(parent, 'div', 'project-site-info-chart', '');
  const svg = doc.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'project-site-info-donut');
  svg.setAttribute('viewBox', '0 0 120 120');
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', counts.mapped
    ? `Planting status breakdown: ${statuses.map(([key, label]) => `${label} ${number(counts[key])}`).join(', ')}.`
    : 'No planting points recorded.');
  const circle = (color) => {
    const ring = doc.createElementNS(svg.namespaceURI, 'circle');
    for (const [key, value] of Object.entries({ cx: 60, cy: 60, r: 46, fill: 'none', stroke: color, 'stroke-width': 13, pathLength: 100 })) {
      ring.setAttribute(key, String(value));
    }
    svg.append(ring);
    return ring;
  };
  circle('#e9eef3');
  const populated = statuses.filter(([key]) => counts[key] > 0);
  let offset = 0;
  for (const [key] of populated) {
    const share = counts[key] / counts.mapped * 100;
    // Keep tiny categories visible; a single category fills the entire ring.
    const gap = populated.length > 1 ? Math.min(0.8, share / 4) : 0;
    const ring = circle(POINT_STATUS_COLORS[key]);
    ring.setAttribute('stroke-dasharray', `${share - gap} ${100 - share + gap}`);
    ring.setAttribute('stroke-dashoffset', String(-offset - gap / 2));
    ring.setAttribute('transform', 'rotate(-90 60 60)');
    offset += share;
  }
  chart.append(svg);
  const total = addText(chart, 'div', 'project-site-info-total', '');
  addText(total, 'strong', '', number(counts.mapped));
  addText(total, 'span', '', scope === 'participant' ? 'Your assigned points' : 'Mapped points');
}

// DOM text nodes keep user-entered site names and notes safe in Leaflet overlays.
export function createProjectSiteInfo(feature, { counts = null, scope = 'site', document: doc = document } = {}) {
  const props = feature?.properties || {};
  const name = props.name || 'Unnamed project site';
  const owner = props.organization_name || props.organization || 'No organization assigned';
  const card = doc.createElement('section');
  card.className = 'project-site-info';
  addText(card, 'div', 'project-site-info-kind', 'Project site');
  addText(card, 'h3', 'project-site-info-title', name);
  if (String(name).trim().toLowerCase() !== String(owner).trim().toLowerCase()) {
    addText(card, 'div', 'project-site-info-owner', owner);
  }

  const details = addText(card, 'dl', 'project-site-info-details', '');
  const detail = (label, value, kind = '') => {
    const row = addText(details, 'div', `project-site-info-detail${kind ? ` is-${kind}` : ''}`, '');
    addText(row, 'dt', '', label);
    addText(row, 'dd', '', value);
  };
  const area = projectSiteAreaM2(feature?.geometry);
  if (area != null) detail('Approx. site area', area >= 10000 ? `${number(area / 10000)} ha` : `${number(area)} m²`, 'area');
  for (const [key, label] of [['analysis_count', 'Saved analyses'], ['assignment_count', 'Assignments'], ['schedule_count', 'Schedules']]) {
    if (props[key] != null && Number.isFinite(Number(props[key]))) detail(label, number(Number(props[key])));
  }

  if (counts) {
    const statuses = scope === 'participant'
      ? [['assigned', 'Pending'], ['planted', 'Planted'], ['dead', 'Recorded dead'], ['skipped', 'Skipped'], ['unavailable', 'Unavailable']]
      : [['planned', 'Available'], ['assigned', 'Assigned'], ['planted', 'Planted'], ['dead', 'Recorded dead'], ['skipped', 'Skipped'], ['unavailable', 'Unavailable']];
    const analytics = addText(card, 'div', 'project-site-info-analytics', '');
    addStatusDonut(analytics, counts, statuses, scope);
    const grid = addText(analytics, 'dl', 'project-site-info-stats', '');
    for (const [key, label] of statuses) {
      const stat = addText(grid, 'div', `project-site-info-stat is-${key}${counts[key] === 0 ? ' is-zero' : ''}`, '');
      stat.style.setProperty('--site-status-color', POINT_STATUS_COLORS[key]);
      addText(stat, 'dt', '', label);
      addText(stat, 'dd', '', number(counts[key]));
    }
    addText(card, 'p', 'project-site-info-scope', scope === 'participant'
      ? 'Counts cover your assigned work in this site.'
      : 'Current mapped locations, across all planting dates.');
  } else if (props.point_count != null && Number.isFinite(Number(props.point_count))) {
    detail('Linked planting points', number(Number(props.point_count)), 'linked');
  }
  if (props.notes) addText(card, 'p', 'project-site-info-notes', props.notes);
  if (!details.childElementCount) details.remove();
  return card;
}

export function bindProjectSiteInfo(layer, feature, options) {
  const content = () => createProjectSiteInfo(feature, options);
  layer.bindTooltip(content, {
    className: 'project-site-tooltip', sticky: true, direction: 'auto', opacity: 1,
  });
  layer.bindPopup(content, {
    className: 'project-site-popup', maxWidth: 320, minWidth: 180,
    maxHeight: 360, autoPanPadding: [12, 12],
    autoPanPaddingTopLeft: [12, 60],
    ...options?.popupOptions,
  });
  layer.on('popupopen', () => layer.closeTooltip());
  layer.on('tooltipopen', () => { if (layer.isPopupOpen()) layer.closeTooltip(); });
}
