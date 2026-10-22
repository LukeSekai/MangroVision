import { assessPlantingBooking, bookingTideEvents, displayBookingTime, plantingBookingWindows } from './bookingTides';
import { bookingPolicy, validBookingDate } from './booking';

export default function PlantingTimeSuggestions({ date, startTime, onChoose, busy, record, now, onRefresh }) {
  const loading = validBookingDate(date) && (!record || record.loading);
  const payload = loading || record?.error ? null : record?.payload;
  const guide = plantingBookingWindows({ payload, date, now });
  const assessment = assessPlantingBooking({ payload, date, startTime, now });
  const events = bookingTideEvents({ payload, date, now });
  const source = record?.payload?.source;
  const attributionUrl = source === 'WorldTides' ? 'https://www.worldtides.info'
    : source === 'Open-Meteo Marine API' ? 'https://open-meteo.com/en/docs/marine-weather-api' : null;

  return <section className="like-tide-guide" aria-label="Planting time suggestions" aria-busy={loading}>
    <div className="like-tide-heading"><strong>Available planting times</strong>{validBookingDate(date) && <button className="like-tide-refresh" type="button" disabled={loading || busy} onClick={onRefresh}>Refresh tides</button>}</div>
    <div aria-live="polite">
      {loading ? <p>Checking MangroVision's tide forecast…</p> : <>
        {events.length > 0 && <>
          <p className="like-tide-events-title">High and low tides (Philippine time)</p>
          <ul className="like-tide-events" aria-label="High and low tides">{events.map((event) => <li key={`${event.at}-${event.type}`}>
            <span>{event.type === 'low' ? 'Low tide' : 'High tide'}</span>
            <strong>{displayBookingTime(event.at)}</strong><span>{event.height_m.toFixed(2)} m</span>
          </li>)}</ul>
          <p>Low tide marks the lowest water level. Suggested planting windows cover the hours when water stays at or below the estimated planting limit, including times before or after low tide.</p>
        </>}
        {guide.partialCoverage && guide.windows.length > 0 && <p>Forecast covers only part of this date. Suggested visits stay within the available forecast.</p>}
        {guide.windows.length ? <ul className="like-tide-windows" aria-label="Suggested planting windows">{guide.windows.map((window) => <li key={window.start}>
          <div><strong>{displayBookingTime(window.start)} – {displayBookingTime(window.end)}</strong><span>Start by {displayBookingTime(window.latestStart)} for a two-hour visit.</span></div>
          <button type="button" disabled={busy} onClick={() => onChoose(window.startTime)}>Use {displayBookingTime(window.start)}</button>
        </li>)}</ul> : <p>{record?.error && validBookingDate(date) ? 'We could not load the tide forecast. You can still request a time for LGU review.' : guide.reason}</p>}
        {['unsafe', 'closed'].includes(assessment.status) && <p className="like-tide-caution">{assessment.reason}</p>}
        {assessment?.status === 'safe' && <p className="like-tide-match">Your full two-hour visit fits the predicted lower-water period{bookingPolicy.enforcePlantingHours && ' and LIKE’s 6 AM–5 PM hours'}.</p>}
        {startTime && assessment?.status === 'unknown' && guide.status === 'available' && <p>We cannot check the full visit at this time. The LGU will review it.</p>}
      </>}
    </div>
    {validBookingDate(date) && <p className="like-tide-note">{bookingPolicy.enforcePlantingHours && 'LIKE planting hours: 6 AM–5 PM. '}Available times use the same estimated tide limit as MangroVision, and your whole two-hour visit must fit. Dates without a forecast can be requested for LGU review.{attributionUrl && <> <a href={attributionUrl} target="_blank" rel="noopener noreferrer">{record.payload.attribution || source}</a></>}</p>}
  </section>;
}
