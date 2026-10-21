import { getMapPointStatus } from './mapPointStats.js';
import { manilaDateKey } from './monitoringSchedule.js';

export const EMPTY_EXPORT_FILTERS = { siteId: '', analysisId: '', species: '' };
const UNRECORDED = 'unrecorded';
const OTHER_SPECIES = 'other-recorded';
const SPECIES_LABELS = { bungalon: 'Bungalon', rhizophora: 'Rhizophora' };
const speciesKey = (point) => {
  const planted = ['planted', 'dead'].includes(getMapPointStatus(point)) ? point.planted_species : null;
  // Preserve recorded values in files without inferring species from spacing.
  return String(planted || point.species || point.analysis_species || '').trim().toLowerCase() || UNRECORDED;
};
const speciesFilterKey = (point) => {
  const key = speciesKey(point);
  return key === UNRECORDED || Object.hasOwn(SPECIES_LABELS, key) ? key : OTHER_SPECIES;
};
const siteId = (point) => point.source_project_site_id ?? point.source_site_id;
export const pointExportSpeciesLabel = (value) => Object.hasOwn(SPECIES_LABELS, value)
  ? SPECIES_LABELS[value] : value === UNRECORDED ? 'Not recorded' : 'Other recorded species';

export function exportablePoints(points) {
  return points.filter((point) => getMapPointStatus(point) !== null);
}

export function pointExportOptions(points, analyses = []) {
  const sites = new Map(), images = new Map(), species = new Map();
  const analysisById = new Map(analyses.map((analysis) => [String(analysis.id), analysis]));
  for (const point of exportablePoints(points)) {
    const owner = siteId(point);
    const siteKey = owner == null ? UNRECORDED : String(owner);
    sites.set(siteKey, { value: siteKey, label: owner == null ? 'No current project site'
      : point.source_project_site_name || point.source_site_name || `Project site #${owner}` });
    if (point.analysis_id != null) {
      const key = String(point.analysis_id);
      const analysis = analysisById.get(key);
      const title = analysis?.display_number != null ? `Analysis ${analysis.display_number}`
        : point.image_name || `Saved analysis (ID ${key})`;
      const source = analysis?.source_image_name;
      images.set(key, { value: key, label: source && source !== title ? `${title} — ${source}` : title });
    }
    const key = speciesFilterKey(point);
    species.set(key, { value: key, label: pointExportSpeciesLabel(key) });
  }
  const sorted = (values) => [...values.values()].sort((a, b) => a.label.localeCompare(b.label, 'en', { numeric: true }));
  return { sites: sorted(sites), analyses: sorted(images), species: sorted(species) };
}

export function filterExportPoints(points, filters = EMPTY_EXPORT_FILTERS) {
  return exportablePoints(points).filter((point) => (
    (!filters.siteId || String(siteId(point) ?? UNRECORDED) === String(filters.siteId))
    && (!filters.analysisId || String(point.analysis_id) === String(filters.analysisId))
    && (!filters.species || speciesFilterKey(point) === filters.species)
  ));
}

export function pointExportScope(filters, options) {
  const label = (list, value, all) => value ? list.find((option) => option.value === String(value))?.label || 'Selection no longer available' : all;
  return [label(options.sites, filters.siteId, 'All project sites'),
    label(options.analyses, filters.analysisId, 'All saved analyses'),
    label(options.species, filters.species, 'All species')];
}

const selectedLabel = (options, list, value, fallback) => options?.[list]?.find((item) => item.value === String(value))?.label || fallback;
const exportLabels = (filters, options) => [
  filters.siteId && selectedLabel(options, 'sites', filters.siteId, `Project site ${filters.siteId}`),
  filters.analysisId && selectedLabel(options, 'analyses', filters.analysisId, `Analysis ${filters.analysisId}`),
  filters.species && selectedLabel(options, 'species', filters.species, pointExportSpeciesLabel(filters.species)),
].filter(Boolean);

export function pointExportName(filters, options, exportedAt = new Date()) {
  const labels = exportLabels(filters, options);
  const selection = (labels.length ? labels.join(' — ') : 'All saved points').replace(/\s+/g, ' ').slice(0, 170);
  return `MangroVision — ${selection} — Planting points — ${manilaDateKey(exportedAt)}`;
}

export function pointExportFilename(filters, format, options, exportedAt = new Date()) {
  const safe = (value) => String(value).normalize('NFKD').replace(/[\u0300-\u036f]/g, '')
    .toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 45);
  const labels = exportLabels(filters, options);
  const ids = [filters.siteId && `site-${safe(filters.siteId)}`, filters.analysisId && `analysis-${safe(filters.analysisId)}`].filter(Boolean);
  const scope = [...labels.map(safe).filter(Boolean), ...ids].join('_').slice(0, 110) || 'all-saved-points';
  const extension = format === 'kml' ? 'kmz' : format;
  return `MangroVision_${scope}_planting-points_${manilaDateKey(exportedAt)}.${extension}`;
}

export function pointExportPayload(points, scope, analyses = [], metadata = {}) {
  const byId = new Map(analyses.map((analysis) => [String(analysis.id), analysis]));
  return {
    image_name: scope.join(' | '),
    ...metadata,
    waypoints: points.map((point, index) => {
      const analysis = byId.get(String(point.analysis_id));
      const analysisName = analysis?.display_number != null ? `Analysis ${analysis.display_number}`
        : point.image_name || (point.analysis_id != null ? `Saved analysis ${point.analysis_id}` : 'Saved planting point');
      const number = String(point.point_num ?? index + 1).padStart(3, '0');
      const navigationName = point.id != null ? `MV-${String(point.id).padStart(6, '0')}`
        : `MV-A${point.analysis_id ?? 0}-P${number}`;
      return {
        lat: point.latitude, lon: point.longitude, point_num: point.point_num,
        buffer_m: point.buffer_m, area_m2: point.area_m2, status: getMapPointStatus(point),
        name: `${analysisName} — Point ${number}${point.id != null ? ` (ID ${point.id})` : ''}`,
        navigation_name: navigationName,
        point_id: point.id, analysis_id: point.analysis_id, image_name: point.image_name,
        project_site_id: siteId(point) ?? null,
        project_site_name: siteId(point) == null ? null : point.source_project_site_name || point.source_site_name || null,
        species: speciesKey(point) === UNRECORDED ? null : speciesKey(point),
        eroded_unavailable: Boolean(point.eroded_unavailable || point.inside_eroded_zone),
        availability_reason: point.availability_reason || null,
      };
    }),
  };
}
