// Provisional exposed-substrate rule, NOT an ecological or personnel-safety rating.
export const PLANTING_STATES = Object.freeze({
  safe: { color: '#15803D', symbol: '\u2713', label: 'Safe to plant' },
  unsafe: { color: '#B91C1C', symbol: '\u00d7', label: 'Not safe to plant' },
  unknown: { color: '#4B5563', symbol: '\u2014', label: 'Tide forecast unavailable' },
});

const OUTDATED_FORECAST_MESSAGE = 'These water levels are out of date. Refresh before planning your visit.';

export function finiteNumber(value) {
  if (!['number', 'string'].includes(typeof value) || String(value).trim() === '') return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function instantMs(value) {
  if (typeof value === 'number') return Number.isFinite(value) ? value : NaN;
  // Reject local wall-clock strings: schedule times must have an explicit offset.
  return typeof value === 'string' && /(?:Z|[+-]\d{2}:\d{2})$/i.test(value)
    ? Date.parse(value) : NaN;
}

export function locationKey(site) {
  const lat = finiteNumber(site?.centroid_lat);
  const lon = finiteNumber(site?.centroid_lon);
  return lat !== null && lon !== null && Math.abs(lat) <= 90 && Math.abs(lon) <= 180
    ? `${lat.toFixed(4)},${lon.toFixed(4)}` : null;
}

export function forecastIssue(payload, now = Date.now()) {
  if (payload?.available !== true) return 'Water levels are unavailable. Please refresh the tides.';
  const fetched = finiteNumber(payload.fetched_at);
  if (payload.stale !== false || fetched === null || fetched * 1000 > now + 60_000
      || now - fetched * 1000 >= 6 * 3600_000) return OUTDATED_FORECAST_MESSAGE;
  if (payload.datum !== 'MSL' || !payload.datum_reference) return 'We cannot confirm how these water levels were measured yet.';
  if (payload.series_available !== true) return 'Some water levels are missing. We need readings throughout the day to check planting times.';
  return '';
}

export function plantingThreshold(site, payload) {
  const calibration = site?.tide_calibration;
  const elevation = finiteNumber(calibration?.elevation_m);
  const siteKey = locationKey(site);
  if (elevation === null || calibration?.datum_compatibility_confirmed !== true
      || payload?.datum !== 'MSL' || !payload?.datum_reference
      || calibration.datum_reference !== payload.datum_reference || !siteKey
      || locationKey({ centroid_lat: payload.lat, centroid_lon: payload.lon }) !== siteKey
      || locationKey({ centroid_lat: calibration.forecast_lat, centroid_lon: calibration.forecast_lon }) !== siteKey) return null;
  return elevation;
}

export function levelState(height, threshold) {
  const h = finiteNumber(height);
  const limit = finiteNumber(threshold);
  return h === null || limit === null ? 'unknown' : h <= limit ? 'safe' : 'unsafe';
}

export function levelSeries(payload) {
  if (!Array.isArray(payload?.heights)) return [];
  return payload.heights.map((row) => ({
    occurred_at: finiteNumber(row?.timestamp) === null ? NaN : Number(row.timestamp) * 1000,
    height_m: finiteNumber(row?.height_m),
  }));
}

export function chartLevelSeries(payload) {
  const rows = levelSeries(payload);
  if (rows.some((row, i) => !Number.isFinite(row.occurred_at)
      || (i > 0 && row.occurred_at <= rows[i - 1].occurred_at))) return [];
  return rows.flatMap((row, i) => {
    const previous = rows[i - 1];
    // A null separator prevents drawing a fabricated curve across missing times.
    return previous && row.occurred_at - previous.occurred_at > 3600_000
      ? [{ occurred_at: (previous.occurred_at + row.occurred_at) / 2, height_m: null }, row] : [row];
  });
}

// Integrate time above ground, interpolating crossings between hourly samples.
// A short forecast estimates the baseline; it is not long-term site monitoring.
export function inundationPercentage(rows, elevation) {
  const limit = finiteNumber(elevation);
  if (limit === null || rows.length < 2) return null;
  let underwater = 0;
  let total = 0;
  for (let i = 1; i < rows.length; i += 1) {
    const a = rows[i - 1];
    const b = rows[i];
    const duration = b.occurred_at - a.occurred_at;
    if (!Number.isFinite(a.occurred_at) || !Number.isFinite(b.occurred_at)
        || duration <= 0 || duration > 3600_000
        || finiteNumber(a.height_m) === null || finiteNumber(b.height_m) === null) return null;
    const low = Math.min(Number(a.height_m), Number(b.height_m));
    const high = Math.max(Number(a.height_m), Number(b.height_m));
    const fraction = low > limit ? 1 : high <= limit ? 0 : (high - limit) / (high - low);
    underwater += duration * fraction;
    total += duration;
  }
  return underwater / total * 100;
}

export function inundationBaseline(percentage, elevation) {
  if (percentage === null || elevation === null) return { status: 'unknown', label: 'Tide forecast unavailable' };
  if (percentage >= 100) return { status: 'unsafe', label: 'Underwater the whole time; avoid planting' };
  if (elevation < 0) return { status: 'unsafe', label: 'Ground is below average sea level; avoid planting' };
  if (percentage <= 30 + 1e-9) return { status: 'safe', label: 'Good for planting' };
  if (percentage <= 50) return { status: 'unsafe', label: 'Check the area first' };
  return { status: 'unsafe', label: 'Avoid planting' };
}

export function forecastPlantingGuide({ site, payload, now = Date.now(), allowOutdated = false }) {
  const rows = levelSeries(payload);
  const surveyed = plantingThreshold(site, payload);
  const issue = forecastIssue(payload, now);
  const blockingIssue = issue && !(allowOutdated && issue === OUTDATED_FORECAST_MESSAGE);
  if (blockingIssue || inundationPercentage(rows, 0) === null) {
    return { threshold: surveyed, percentage: null, preview: surveyed === null,
      status: 'unknown', label: 'Tide forecast unavailable' };
  }
  let threshold = surveyed;
  if (threshold === null) {
    // Minimum hypothetical elevation with <=30% inundation, constrained to >=MSL.
    // Never save this forecast-derived reference as a surveyed site elevation.
    let low = Math.min(...rows.map((row) => row.height_m));
    let high = Math.max(...rows.map((row) => row.height_m));
    for (let i = 0; i < 60; i += 1) {
      const middle = (low + high) / 2;
      if (inundationPercentage(rows, middle) > 30) low = middle;
      else high = middle;
    }
    threshold = Math.max(0, high);
  }
  const percentage = inundationPercentage(rows, threshold);
  return { threshold, percentage, preview: surveyed === null, ...inundationBaseline(percentage, threshold) };
}

export function guideLevelState(height, guide) {
  const level = levelState(height, guide.threshold);
  if (level === 'unknown' || guide.status === 'unknown') return 'unknown';
  return guide.status === 'safe' ? level : 'unsafe';
}

export function coloredTideSeries(rows, guide) {
  const expanded = rows.flatMap((row, i) => {
    const previous = rows[i - 1];
    if (guide.threshold === null || !previous || previous.height_m === null || row.height_m === null
        || (previous.height_m - guide.threshold) * (row.height_m - guide.threshold) >= 0) return [row];
    return [{ occurred_at: previous.occurred_at + (row.occurred_at - previous.occurred_at)
      * (guide.threshold - previous.height_m) / (row.height_m - previous.height_m),
    height_m: guide.threshold, crossing: true }, row];
  });
  return expanded.map((row) => {
    const state = guideLevelState(row.height_m, guide);
    // Share boundary points so neither color stops short of the reference line.
    const boundary = row.height_m === guide.threshold && guide.status === 'safe';
    return { ...row, safe: state === 'safe' || boundary ? row.height_m : null,
      unsafe: state === 'unsafe' || boundary ? row.height_m : null,
      unknown: state === 'unknown' ? row.height_m : null };
  });
}

// Scheduling uses the exact coastal series and estimated limit displayed by the
// graph. A site assignment is administrative and does not change this estimate.
export function assessGraphWindow({ payload, startAt, endAt, now = Date.now(), allowOutdated = false }) {
  return assessWindow({ payload, startAt, endAt, now, useGraphReference: true, allowOutdated });
}

// Check a selected start or end time even before the other field is filled in.
export function assessGraphTime({ payload, at, now = Date.now(), allowOutdated = false }) {
  const guide = forecastPlantingGuide({ payload, now, allowOutdated });
  const time = instantMs(at);
  if (guide.status === 'unknown' || !Number.isFinite(time)) return { status: 'unknown' };
  const rows = levelSeries(payload);
  for (let i = 1; i < rows.length; i += 1) {
    const a = rows[i - 1];
    const b = rows[i];
    if (time < a.occurred_at || time > b.occurred_at) continue;
    const height = a.height_m + (b.height_m - a.height_m) * (time - a.occurred_at) / (b.occurred_at - a.occurred_at);
    return { status: guideLevelState(height, guide), threshold_m: guide.threshold, maximum_m: height, estimated: true };
  }
  return { status: 'unknown' };
}

export function assessWindow({ site, payload, startAt, endAt, now = Date.now(), useGraphReference = false, allowOutdated = false }) {
  const unknown = (reason, label = 'Refresh tides to check this activity') => ({ status: 'unknown', reason, label });
  if (!site && !useGraphReference) return unknown('Choose a planting area to check this activity.');
  const start = instantMs(startAt);
  const end = instantMs(endAt);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return unknown('Enter valid start and end times, using Philippine time.', 'Set the activity start and end times');
  const issue = forecastIssue(payload, now);
  if (issue && !(allowOutdated && issue === OUTDATED_FORECAST_MESSAGE)) return unknown(issue);
  const guide = forecastPlantingGuide({ site: useGraphReference ? null : site, payload, now, allowOutdated });
  if (useGraphReference && guide.status === 'unknown') return unknown('Some water levels are missing from the graph. Please refresh the tides.');
  const threshold = useGraphReference ? guide.threshold : plantingThreshold(site, payload);
  if (threshold === null) return unknown('The ground height at this planting area must be measured and checked against the water levels first.');
  if (!useGraphReference && start < now) return unknown('Today’s water predictions cannot check activities that have already happened.');
  const rows = levelSeries(payload);
  if (rows.length < 2 || rows.some((row, i) => !Number.isFinite(row.occurred_at)
      || (i > 0 && row.occurred_at <= rows[i - 1].occurred_at))) return unknown('We could not read the water levels. Please refresh the tides.');
  let coveredUntil = start;
  let maximum = -Infinity;
  for (let i = 1; i < rows.length; i += 1) {
    const a = rows[i - 1];
    const b = rows[i];
    if (b.occurred_at <= start || a.occurred_at >= end) continue;
    const gap = b.occurred_at - a.occurred_at;
    if (gap > 3600_000 || a.height_m === null || b.height_m === null) return unknown('Some water levels are missing during this activity.');
    const left = Math.max(start, a.occurred_at);
    const right = Math.min(end, b.occurred_at);
    if (left > coveredUntil) return unknown('The graph does not cover the full activity. Choose a time within the dates shown.', 'Tide forecast not available for this date');
    const interpolate = (t) => a.height_m + (b.height_m - a.height_m) * (t - a.occurred_at) / gap;
    maximum = Math.max(maximum, interpolate(left), interpolate(right));
    coveredUntil = right;
  }
  if (coveredUntil < end || !Number.isFinite(maximum)) return unknown('The graph does not cover the full activity. Choose a time within the dates shown.', 'Tide forecast not available for this date');
  // Include provider extrema to avoid missing peaks between regular samples.
  for (const row of !useGraphReference && Array.isArray(payload.extremes) ? payload.extremes : []) {
    const timestamp = finiteNumber(row?.timestamp);
    if (timestamp === null) return unknown('We could not read the high and low tide times. Please refresh the tides.');
    if (timestamp * 1000 < start || timestamp * 1000 > end) continue;
    const height = finiteNumber(row.height_m);
    if (height === null) return unknown('A water level is missing during this activity.');
    maximum = Math.max(maximum, height);
  }
  if (guide.status === 'unknown') return unknown('We need water levels for every hour to estimate how long the ground is underwater.');
  const status = guideLevelState(maximum, guide);
  return {
    status, maximum_m: maximum, depth_m: Math.max(0, maximum - threshold), threshold_m: threshold,
    inundation_percentage: guide.percentage,
    estimated: useGraphReference,
    reason: useGraphReference
      ? status === 'safe'
        ? 'Water stays at or below the graph\'s planting limit for the whole activity. This is an estimate; check your planting area before going.'
        : 'Water rises above the graph\'s planting limit during this activity. Choose a time when the graph stays green from start to finish.'
      : status === 'safe'
      ? 'Water is expected to stay below the ground or just reach it throughout this activity. Check the path, waves and weather before going.'
      : guide.status === 'unsafe' ? `${guide.label}: ${guide.percentage.toFixed(1)}% of the time underwater. Choose ground at or above average sea level that is underwater no more than 30% of the time.`
        : `Water may reach ${(maximum - threshold).toFixed(2)} m above the ground during this activity.`,
  };
}
