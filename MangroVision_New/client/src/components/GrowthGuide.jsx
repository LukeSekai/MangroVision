import './GrowthGuide.css';

export default function GrowthGuide() {
  return <details className="growth-guide">
    <summary>How the groups are counted</summary>
    <p>Every new plant starts in the 0 to 2 week group. After that, it moves to the 2 to 4 week group, then the 4 to 6 week group, and so on.</p>
    <p>The count starts from the date each seedling was planted. A new planting always begins in the first group.</p>
    <div className="growth-guide-table"><table>
      <caption>Examples from Philippine studies</caption>
      <thead><tr><th>Mangrove type</th><th>Usual growth in 2 weeks</th></tr></thead>
      <tbody>
        <tr><td><a href="https://bioflux.com.ro/docs/2023.534-545.pdf" target="_blank" rel="noreferrer">Rhizophora apiculata</a></td><td>2.6–2.9 cm</td></tr>
        <tr><td><a href="https://forestist.org/Content/files/sayilar/450/241-246%281%29.pdf" target="_blank" rel="noreferrer">Bungalon (Avicennia marina)</a></td><td>0.54–0.60 cm</td></tr>
      </tbody>
    </table></div>
    <p>These examples come from different places and growing conditions, so actual growth at your site may be different.</p>
    <p>The age group updates automatically at every visit. It shows time since planting, not the measured height of the seedling.</p>
  </details>;
}
