import { useEffect, useState } from 'react';
import useTideForecasts from '../utils/useTideForecasts';
import { assessGraphWindow, COASTAL_FORECAST_KEY } from '../utils/plantingTides';
import { displayBookingTime, plantingBookingWindows } from './bookingTides';
import { validBookingDate } from './booking';

export default function PlantingTimeSuggestions({ date, startTime, endTime, onChoose, busy }) {
  const [retry, setRetry] = useState(0);
  const [now, setNow] = useState(Date.now);
  const records = useTideForecasts(validBookingDate(date) ? [COASTAL_FORECAST_KEY] : [], retry);
  const record = records[COASTAL_FORECAST_KEY];
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 60_000);
    return () => window.clearInterval(timer);
  }, []);
  const loading = validBookingDate(date) && (!record || record.loading);
  const guide = plantingBookingWindows({ payload: record?.error ? null : record?.payload, date, now });
  const assessment = !loading && !record?.error && validBookingDate(date) && startTime && endTime
    ? assessGraphWindow({ payload: record?.payload, startAt: `${date}T${startTime}:00+08:00`, endAt: `${date}T${endTime}:00+08:00`, now }) : null;
  const source = record?.payload?.source;
  const attributionUrl = source === 'WorldTides' ? 'https://www.worldtides.info'
    : source === 'Open-Meteo Marine API' ? 'https://open-meteo.com/en/docs/marine-weather-api' : null;

  return <section className="like-tide-guide" aria-label="Planting time suggestions" aria-busy={loading}>
    <div className="like-tide-heading"><strong>Suggested planting times</strong>{validBookingDate(date) && <button className="like-tide-refresh" type="button" disabled={loading || busy} onClick={() => { setNow(Date.now()); setRetry((value) => value + 1); }}>Refresh tides</button>}</div>
    <div aria-live="polite">
      {loading ? <p>Checking MangroVision's tide forecast…</p> : <>
        {guide.windows.length ? <ul className="like-tide-windows">{guide.windows.map((window) => <li key={window.start}>
          <div><strong>{displayBookingTime(window.start)} – {displayBookingTime(window.end)}</strong><span>Start by {displayBookingTime(window.latestStart)} for a two-hour visit.</span></div>
          <button type="button" disabled={busy} onClick={() => onChoose(window.startTime)}>Use {displayBookingTime(window.start)}</button>
        </li>)}</ul> : <p>{record?.error && validBookingDate(date) ? 'We could not load the tide forecast. You can still request a time for LGU review.' : guide.reason}</p>}
        {assessment?.status === 'unsafe' && <p className="like-tide-caution">Water is predicted to rise above the planting limit during your two-hour visit. Choose a suggested time, or ask the LGU to review this time.</p>}
        {assessment?.status === 'safe' && <p className="like-tide-match">Your full two-hour visit fits the predicted lower-water period.</p>}
        {assessment?.status === 'unknown' && guide.status === 'available' && <p>We cannot check the full visit at this time. The LGU will review it.</p>}
      </>}
    </div>
    {validBookingDate(date) && <p className="like-tide-note">Daytime suggestions (6 AM–6 PM) use the same estimated tide limit as MangroVision. The LGU will confirm the time and planting conditions.{attributionUrl && <> <a href={attributionUrl} target="_blank" rel="noopener noreferrer">{record.payload.attribution || source}</a></>}</p>}
  </section>;
}
