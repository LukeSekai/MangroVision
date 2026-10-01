export function selectAnalysis(rows = [], selectedId = '') {
  const analyses = rows.map((row) => ({
    ...row,
    label: row.image_name || 'Saved analysis',
  })).sort((a, b) => (a.analysis_number ?? 0) - (b.analysis_number ?? 0));
  return {
    analyses,
    selected: analyses.find((row) => String(row.analysis_id) === String(selectedId)) || analyses[0] || null,
  };
}

function measurement(value) {
  if (value === null || value === undefined || value === '') return null;
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? number : null;
}

export function analysisPieData(analysis) {
  const total = measurement(analysis?.total_area_m2);
  const plantable = measurement(analysis?.plantable_area_m2);
  const risk = measurement(analysis?.danger_area_m2);
  const canopy = measurement(analysis?.canopy_coverage_pct);
  const areaSlices = [];
  let areaMessage = 'Area measurements were not recorded for this analysis.';
  if (total > 0 && plantable !== null && risk !== null) {
    const classified = plantable + risk;
    if (classified <= total + Math.max(0.01, total * 0.000001)) {
      areaSlices.push(
        { key: 'available', name: 'Plantable area', value: plantable },
        { key: 'danger', name: 'Risk area', value: risk },
      );
      if (total > classified) areaSlices.push({ key: 'missing', name: 'Other area', value: total - classified });
    } else {
      areaMessage = 'The recorded areas exceed the image area. Review this analysis before comparing proportions.';
    }
  }
  return {
    areaSlices, areaMessage,
    canopySlices: canopy !== null && canopy <= 100 ? [
      { key: 'canopy', name: 'Tree canopy', value: canopy },
      { key: 'missing', name: 'Without canopy', value: 100 - canopy },
    ] : [],
  };
}
