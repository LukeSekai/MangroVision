import GrowthGuide from './GrowthGuide';

export default function AutomaticGrowth({ snapshot, alive, guide = true }) {
  if (!snapshot) return <p className="org-monitoring-growth-help">Loading planting dates...</p>;
  return <div className="automatic-growth">
    <strong>{alive === 0 ? 'No living seedlings' : snapshot.label}</strong>
    <p className="org-monitoring-growth-help">Calculated from planting dates to {snapshot.as_of}. Each new planting starts as a seedling; age groups advance every 14 days.</p>
    {alive !== 0 && snapshot.cohorts?.length ? <div className="growth-guide-table"><table>
      <thead><tr><th>Mangrove</th><th>Automatic age group</th><th>Originally planted</th></tr></thead>
      <tbody>{snapshot.cohorts.map((group) => <tr key={`${group.species}-${group.completed_cycles}`}>
        <td>{group.species}</td><td>{group.stage_label} · {group.age_label}</td><td>{group.planted_count}</td>
      </tr>)}</tbody>
    </table></div> : null}
    {snapshot.missing_date_count > 0 ? <p className="org-monitoring-growth-help">{snapshot.missing_date_count} planting dates are missing; their ages cannot be calculated.</p> : null}
    <p className="org-monitoring-growth-help">These are age-based estimates, not measured size or maturity. Deaths are recorded for the organization, so the batch counts above include past deaths.</p>
    {guide ? <GrowthGuide /> : null}
  </div>;
}
