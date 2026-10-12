import {
  CartesianGrid, Line, LineChart, ReferenceArea, ReferenceDot, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { coloredTideSeries, finiteNumber, guideLevelState, PLANTING_STATES } from '../utils/plantingTides';

const TIMEZONE = 'Asia/Manila';
const clock = (at) => new Intl.DateTimeFormat('en-US', {
  hour: 'numeric', minute: '2-digit', timeZone: TIMEZONE,
}).format(new Date(at));
const dateLabel = (at) => new Intl.DateTimeFormat('en-US', {
  month: 'short', day: 'numeric', timeZone: TIMEZONE,
}).format(new Date(at));
const kind = (tide) => String(tide.tide_type ?? tide.type ?? '').toLowerCase();

export default function TideChart({ rows, guide, tides = [], day, selectedTide, connectPoints = true, datumLabel = 'meters' }) {
  const data = coloredTideSeries(rows, guide);
  const values = rows.map((row) => finiteNumber(row.height_m)).filter((value) => value !== null);
  if (guide.threshold !== null) values.push(guide.threshold);
  const low = Math.min(0, ...values);
  const high = Math.max(0, ...values);
  const padding = Math.max(0.15, (high - low) * 0.15);
  const domain = [Math.floor((low - padding) * 10) / 10, Math.ceil((high + padding) * 10) / 10];
  const dayStart = day ? Date.parse(`${day}T00:00:00+08:00`) : null;
  const timeDomain = day ? [dayStart, dayStart + 86400_000] : ['dataMin', 'dataMax'];
  const shortForecast = rows.length > 0 && rows.at(-1).occurred_at - rows[0].occurred_at <= 86400_000;
  const clockAxis = Boolean(day || shortForecast);
  const referenceLabel = guide.preview ? 'Estimated planting limit' : 'Measured ground height';
  const markers = tides.filter((tide) => ['high', 'low'].includes(kind(tide))
    && finiteNumber(tide.height_m) !== null
    && tide.occurred_at >= rows[0]?.occurred_at && tide.occurred_at <= rows.at(-1)?.occurred_at);
  const statusLabel = (height) => {
    const status = guideLevelState(height, guide);
    return status === 'safe' ? 'Water low enough · Good time to plant'
      : status === 'unsafe' ? guide.status === 'safe' ? 'Water too high · Avoid planting' : 'Avoid planting'
        : 'Planting guidance unavailable';
  };

  return <div className={`schedule-tide-chart${day ? ' schedule-tide-day-chart' : ''}`} role="img"
    aria-label={`Expected water level (${datumLabel}); ${day ? `hourly view for ${day}` : 'full forecast'}, Philippine time. ${referenceLabel}: ${guide.threshold === null ? 'unknown' : `${guide.threshold.toFixed(2)} m`}. Green shows suggested planting times; red means avoid planting.`}>
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: day ? 32 : 24, right: 28, bottom: 30, left: 14 }}>
        <CartesianGrid stroke="#dbe4e7" strokeDasharray="3 5" vertical={false} />
        <XAxis dataKey="occurred_at" type="number" domain={timeDomain}
          ticks={day ? [0, 6, 12, 18, 24].map((hour) => dayStart + hour * 3600_000) : undefined}
          tickFormatter={clockAxis ? clock : dateLabel} minTickGap={40}
          label={{ value: `${clockAxis ? 'Time' : 'Date'} (Philippine time)`, position: 'bottom', offset: 12, fill: '#334155', fontSize: 12 }} />
        <YAxis domain={domain} tickFormatter={(value) => `${Number(value.toFixed(2))} m`} width={64}
          label={{ value: 'Water level (meters)', angle: -90, position: 'insideLeft', dx: -10, style: { textAnchor: 'middle', fontSize: 11 }, fill: '#334155' }} />
        {guide.status !== 'unknown' && guide.threshold !== null ? <>
          <ReferenceArea y1={domain[0]} y2={guide.threshold} fill={guide.status === 'safe' ? '#16a34a' : '#dc2626'} fillOpacity={0.07} />
          <ReferenceArea y1={guide.threshold} y2={domain[1]} fill="#dc2626" fillOpacity={0.07} />
          <ReferenceLine y={guide.threshold} stroke="#475569" strokeWidth={1.5} strokeDasharray="6 4" />
        </> : null}
        <Tooltip content={({ active, payload: points, label }) => {
          const row = points?.find((point) => point.value !== null)?.payload;
          const peak = markers.find((tide) => tide.occurred_at === Number(label));
          return active && row ? <div className="schedule-tide-tooltip">
            <strong>{day ? clock(label) : new Intl.DateTimeFormat('en-US', {
              dateStyle: 'medium', timeStyle: 'short', timeZone: TIMEZONE,
            }).format(new Date(label))}</strong>
            {peak ? <span>{kind(peak) === 'low' ? 'Low tide' : 'High tide'}</span> : null}
            <span>Water level: {row.height_m.toFixed(2)} m</span>
            <span style={{ color: PLANTING_STATES[guideLevelState(row.height_m, guide)].color }}>{statusLabel(row.height_m)}</span>
          </div> : null;
        }} />
        {Object.entries(PLANTING_STATES).map(([status, style]) => <Line key={status} type="linear" dataKey={status}
          name={style.label} stroke={connectPoints ? style.color : 'none'} strokeWidth={3}
          connectNulls={false} isAnimationActive={false} dot={connectPoints ? false : { r: 3, fill: style.color }} activeDot={{ r: 5, strokeWidth: 2 }} />)}
        {markers.map((tide) => <ReferenceDot key={`${tide.occurred_at}:${kind(tide)}`}
          x={tide.occurred_at} y={tide.height_m} r={tide.occurred_at === selectedTide?.occurred_at ? 6 : 4}
          fill={kind(tide) === 'low' ? '#15803d' : '#b91c1c'} stroke="#fff" strokeWidth={2}
          label={day ? { value: kind(tide) === 'low' ? 'Low tide' : 'High tide',
            position: kind(tide) === 'low' ? 'bottom' : 'top', fontSize: 11, fill: '#334155' } : undefined} />)}
      </LineChart>
    </ResponsiveContainer>
  </div>;
}
