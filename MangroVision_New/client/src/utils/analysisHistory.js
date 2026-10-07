import { manilaDay } from './restorationReports.js';

export const DEFAULT_HISTORY_FILTERS = {
  search: '', date: 'all', sort: 'newest',
};

export const HISTORY_SORT_OPTIONS = [
  ['newest', 'Newest first'], ['oldest', 'Oldest first'],
  ['plantable-desc', 'Largest plantable area'], ['plantable-asc', 'Smallest plantable area'],
  ['canopy-desc', 'Highest canopy coverage'], ['canopy-asc', 'Lowest canopy coverage'],
];

const historyDay = value => value ? manilaDay(value) : '';

export function analysisNumber(value) {
  if (value === null || value === undefined || value === '' || typeof value === 'boolean') return null;
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? number : null;
}

export function canopyCoverage(analysis) {
  const coverage = analysisNumber(analysis.canopy_coverage_pct);
  if (coverage !== null) return Math.min(100, coverage);
  const area = analysisNumber(analysis.canopy_area_m2);
  const total = analysisNumber(analysis.total_area_m2);
  return area !== null && total > 0 ? Math.min(100, area / total * 100) : null;
}

export function analysisDateOptions(analyses) {
  const months = [...new Set(analyses.map(a => historyDay(a.analyzed_at).slice(0, 7)).filter(Boolean))].sort().reverse();
  return months.map(month => [month, new Intl.DateTimeFormat('en-PH', {
    month: 'long', year: 'numeric', timeZone: 'Asia/Manila',
  }).format(new Date(`${month}-01T00:00:00+08:00`))]);
}

function dateMatches(value, selected, today) {
  if (selected === 'all') return true;
  const day = historyDay(value);
  if (!day) return false;
  if (/^\d{4}-\d{2}$/.test(selected)) return day.startsWith(selected);
  const days = selected === 'today' ? 1 : Number(selected);
  const start = new Date(`${today}T00:00:00Z`);
  start.setUTCDate(start.getUTCDate() - days + 1);
  return day >= start.toISOString().slice(0, 10) && day <= today;
}

export function filterAnalysisHistory(analyses, filters = DEFAULT_HISTORY_FILTERS, today = manilaDay()) {
  const search = filters.search.trim().toLocaleLowerCase();
  const filtered = analyses.filter(analysis => {
    const name = `${analysis.image_name || ''} ${analysis.source_image_name || ''}`.toLocaleLowerCase();
    return name.includes(search)
      && dateMatches(analysis.analyzed_at, filters.date, today);
  });
  const timestamp = analysis => {
    const parsed = Date.parse(analysis.analyzed_at);
    return Number.isFinite(parsed) ? parsed : null;
  };
  const byRecent = (a, b) => (timestamp(b) ?? -Infinity) - (timestamp(a) ?? -Infinity)
    || Number(b.id) - Number(a.id);
  const metric = filters.sort.startsWith('plantable')
    ? a => analysisNumber(a.plantable_area_m2) : canopyCoverage;
  return filtered.sort((a, b) => {
    if (filters.sort === 'newest' || filters.sort === 'oldest') {
      const av = timestamp(a), bv = timestamp(b);
      if (av === null || bv === null) return av === bv ? byRecent(a, b) : av === null ? 1 : -1;
      return (av - bv) * (filters.sort === 'newest' ? -1 : 1) || byRecent(a, b);
    }
    const av = metric(a), bv = metric(b);
    if (av === null || bv === null) return av === bv ? byRecent(a, b) : av === null ? 1 : -1;
    return (av - bv) * (filters.sort.endsWith('desc') ? -1 : 1) || byRecent(a, b);
  });
}

export function formatAnalysisDate(value) {
  const day = historyDay(value);
  if (!day) return 'Date unavailable';
  return new Intl.DateTimeFormat('en-PH', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'Asia/Manila' })
    .format(new Date(`${day}T00:00:00+08:00`));
}

export function formatAnalysisNumber(value, digits = 1) {
  const number = analysisNumber(value);
  return number === null ? '—' : number.toLocaleString('en-PH', { maximumFractionDigits: digits });
}
