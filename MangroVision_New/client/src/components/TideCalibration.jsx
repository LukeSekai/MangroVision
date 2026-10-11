import { useState } from 'react';
import { finiteNumber, forecastIssue } from '../utils/plantingTides';
import useFormFeedback from '../utils/useFormFeedback';
import { FieldError, FormErrorSummary, FormMessage } from './FormFeedback';

export default function TideCalibration({ site, payload, onSave, now }) {
  const [elevation, setElevation] = useState(site?.tide_calibration?.elevation_m ?? '');
  const [reference, setReference] = useState(site?.tide_calibration?.survey_reference ?? '');
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [failed, setFailed] = useState(false);
  const issue = forecastIssue(payload, now);
  const feedback = useFormFeedback({ elevation: { label: 'Surveyed elevation', aliases: ['elevation_m'] }, reference: { label: 'Survey reference', aliases: ['survey_reference'] }, confirmed: { label: 'Datum compatibility confirmation' } });

  async function save(event, clear = false) {
    event.preventDefault();
    if (busy || (!clear && !feedback.validate())) return;
    if (!clear && (finiteNumber(elevation) === null || !confirmed || reference.trim().length < 10 || issue)) {
      setFailed(true);
      if (issue) setMessage(issue);
      else feedback.reject({ confirmed: 'Confirm that the measured elevation uses this forecast’s vertical reference.' });
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
      if (!feedback.fromServer(error)) setMessage(error.message || 'Could not save the surveyed threshold.');
    } finally { setBusy(false); }
  }

  return (
    <details className="schedule-calibration">
      <summary>Set surveyed planting level for {site.name}</summary>
      <p>Enter the measured ground elevation to estimate the time this site is underwater.
        A suitable baseline requires elevation at or above MSL and no more than 30% forecast inundation.
        Green planting windows also require water at or below ground level.</p>
      <form noValidate onChangeCapture={feedback.onChange} aria-busy={busy} onSubmit={save}>
        <FormErrorSummary feedback={feedback} />
        <label>Surveyed ground elevation (m in this forecast&apos;s MSL reference)
          <input {...feedback.props('elevation')} type="number" step="any" value={elevation} onChange={(event) => setElevation(event.target.value)} required disabled={busy} /><FieldError feedback={feedback} field="elevation" />
        </label>
        <label>Survey / benchmark and datum-conversion reference
          <textarea {...feedback.props('reference')} value={reference} onChange={(event) => setReference(event.target.value)} minLength={10} maxLength={1000} required disabled={busy} /><FieldError feedback={feedback} field="reference" />
        </label>
        <label className="schedule-calibration-check">
          <input {...feedback.props('confirmed')} type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} required disabled={busy} />
          I have verified the survey against this provider&apos;s vertical reference. Matching “MSL” labels alone is not sufficient.
        </label>
        <FieldError feedback={feedback} field="confirmed" />
        {issue ? <p role="status">{issue}</p> : null}
        <FormMessage error={failed ? message : ''} success={!failed ? message : ''} loading={busy ? 'Saving surveyed threshold…' : ''} />
        <button type="submit" disabled={busy || Boolean(issue)}>{busy ? 'Saving...' : 'Save surveyed threshold'}</button>
        {site.tide_calibration ? <button type="button" disabled={busy} onClick={(event) => save(event, true)}>Clear calibration</button> : null}
      </form>
    </details>
  );
}
