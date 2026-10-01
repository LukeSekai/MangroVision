import { useState } from 'react';
import { finiteNumber, forecastIssue } from '../utils/plantingTides';

export default function TideCalibration({ site, payload, onSave, now }) {
  const [elevation, setElevation] = useState(site?.tide_calibration?.elevation_m ?? '');
  const [reference, setReference] = useState(site?.tide_calibration?.survey_reference ?? '');
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [failed, setFailed] = useState(false);
  const issue = forecastIssue(payload, now);

  async function save(event, clear = false) {
    event.preventDefault();
    if (!clear && (finiteNumber(elevation) === null || !confirmed || reference.trim().length < 10 || issue)) {
      setFailed(true);
      setMessage(issue || 'Enter the surveyed elevation, survey/benchmark reference (at least 10 characters), and confirm datum compatibility.');
      return;
    }
    setBusy(true);
    setMessage('');
    try {
      await onSave(clear ? null : {
        elevation_m: Number(elevation), survey_reference: reference.trim(),
        datum_reference: payload.datum_reference, datum_compatibility_confirmed: true,
      });
      setFailed(false);
      setMessage(clear ? 'Calibration cleared. Suitability is Unknown.' : 'Surveyed threshold saved for this site.');
      setConfirmed(false);
      if (clear) { setElevation(''); setReference(''); }
    } catch (error) {
      setFailed(true);
      setMessage(error.message || 'Could not save the surveyed threshold.');
    } finally { setBusy(false); }
  }

  return (
    <details className="schedule-calibration">
      <summary>Set surveyed planting level for {site.name}</summary>
      <p>Enter the measured ground elevation to estimate the time this site is underwater.
        A suitable baseline requires elevation at or above MSL and no more than 30% forecast inundation.
        Green planting windows also require water at or below ground level.</p>
      <form onSubmit={save}>
        <label>Surveyed ground elevation (m in this forecast&apos;s MSL reference)
          <input type="number" step="any" value={elevation} onChange={(event) => setElevation(event.target.value)} required disabled={busy} />
        </label>
        <label>Survey / benchmark and datum-conversion reference
          <textarea value={reference} onChange={(event) => setReference(event.target.value)} minLength={10} maxLength={1000} required disabled={busy} />
        </label>
        <label className="schedule-calibration-check">
          <input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} required disabled={busy} />
          I have verified the survey against this provider&apos;s vertical reference. Matching “MSL” labels alone is not sufficient.
        </label>
        {issue ? <p role="status">{issue}</p> : null}
        {message ? <p role={failed ? 'alert' : 'status'}>{message}</p> : null}
        <button type="submit" disabled={busy || Boolean(issue)}>{busy ? 'Saving...' : 'Save surveyed threshold'}</button>
        {site.tide_calibration ? <button type="button" disabled={busy} onClick={(event) => save(event, true)}>Clear calibration</button> : null}
      </form>
    </details>
  );
}
