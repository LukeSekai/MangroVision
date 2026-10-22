import { assessGraphWindow, finiteNumber, forecastIssue, forecastPlantingGuide, levelSeries } from '../utils/plantingTides.js';
import { APPOINTMENT_DURATION_MINUTES, PLANTING_OPEN_TIME, PLANTING_CLOSE_TIME, appointmentEndTime, bookingPolicy, plantingHoursError, validBookingDate } from './booking.js';

const MINUTE = 60_000;
const SLOT = 15 * MINUTE;
const DURATION = APPOINTMENT_DURATION_MINUTES * MINUTE;
const timeFormatter = new Intl.DateTimeFormat('en-PH', { timeZone: 'Asia/Manila', hour: 'numeric', minute: '2-digit' });
const inputFormatter = new Intl.DateTimeFormat('en-GB', { timeZone: 'Asia/Manila', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });
export const displayBookingTime = (time) => timeFormatter.format(new Date(time));

export function bookingTideEvents({ payload, date, now = Date.now() }) {
  if (!validBookingDate(date) || forecastIssue(payload, now)) return [];
  const dayStart = Date.parse(`${date}T00:00:00+08:00`);
  const dayEnd = dayStart + 24 * 60 * MINUTE;
  return (Array.isArray(payload.extremes) ? payload.extremes : []).flatMap((row) => {
    const timestamp = finiteNumber(row?.timestamp);
    const height = finiteNumber(row?.height_m);
    const type = String(row?.type).toLowerCase();
    const at = timestamp === null ? NaN : timestamp * 1000;
    return ['high', 'low'].includes(type) && height !== null && at >= dayStart && at < dayEnd
      ? [{ at, type, height_m: height }] : [];
  }).sort((a, b) => a.at - b.at);
}

export function assessPlantingBooking({ payload, date, startTime, now = Date.now() }) {
  if (!startTime || !validBookingDate(date)) return { status: 'unknown', reason: '' };
  if (!appointmentEndTime(startTime)) return { status: 'closed', reason: 'Choose a valid start time before 10 PM so the two-hour visit ends on the same day.' };
  const hoursError = plantingHoursError(startTime);
  if (hoursError) return { status: 'closed', reason: hoursError };
  const assessment = assessGraphWindow({ payload, startAt: `${date}T${startTime}:00+08:00`,
    endAt: `${date}T${appointmentEndTime(startTime)}:00+08:00`, now });
  return { ...assessment, reason: assessment.status === 'unsafe'
    ? 'Water is predicted to rise above the planting limit during your two-hour visit. Choose a lower-water time.' : assessment.reason };
}

// Public guidance uses exactly the coastal graph's estimated planting reference.
// Trim crossing intervals inward to quarter-hours so the full visit fits.
export function plantingBookingWindows({ payload, date, now = Date.now() }) {
  const unavailable = (reason) => ({ status: 'unavailable', windows: [], reason });
  if (!validBookingDate(date)) return { status: 'choose-date', windows: [], reason: 'Choose a date to see suggested planting times.' };
  const issue = forecastIssue(payload, now);
  if (issue) return unavailable(issue);
  const guide = forecastPlantingGuide({ payload, now });
  if (guide.status === 'unknown') return unavailable('Some tide readings are missing. Refresh to check planting times.');
  // Keep visits on one Philippine date while the operating-hours limit is off.
  const dayStart = Date.parse(`${date}T${bookingPolicy.enforcePlantingHours ? PLANTING_OPEN_TIME : '00:00'}:00+08:00`);
  const dayEnd = Date.parse(`${date}T${bookingPolicy.enforcePlantingHours ? PLANTING_CLOSE_TIME : '23:59'}:00+08:00`);
  const rows = levelSeries(payload);
  if (rows.length < 2 || rows.some((row, i) => !Number.isFinite(row.occurred_at) || row.height_m === null
      || (i > 0 && (row.occurred_at <= rows[i - 1].occurred_at || row.occurred_at - rows[i - 1].occurred_at > 3600_000)))) {
    return unavailable('Some tide readings are missing. Refresh to check planting times.');
  }
  const coveredStart = Math.max(dayStart, rows[0].occurred_at);
  const coveredEnd = Math.min(dayEnd, rows.at(-1).occurred_at);
  if (coveredEnd <= coveredStart) {
    return unavailable('The tide forecast does not cover this date yet. You can still request a time for LGU review.');
  }
  const partialCoverage = coveredStart > dayStart || coveredEnd < dayEnd;
  const periods = [];
  for (let i = 1; i < rows.length; i += 1) {
    const a = rows[i - 1], b = rows[i];
    if (a.height_m > guide.threshold && b.height_m > guide.threshold) continue;
    const crossing = a.occurred_at + (b.occurred_at - a.occurred_at)
      * (guide.threshold - a.height_m) / (b.height_m - a.height_m);
    const start = a.height_m <= guide.threshold ? a.occurred_at : crossing;
    const end = b.height_m <= guide.threshold ? b.occurred_at : crossing;
    if (end <= start) continue;
    const last = periods.at(-1);
    if (last && Math.abs(last.end - start) < 1) last.end = end;
    else periods.push({ start, end });
  }
  const windows = periods.flatMap((period) => {
    const start = Math.ceil(Math.max(period.start, coveredStart, now + 1) / SLOT) * SLOT;
    const end = Math.floor(Math.min(period.end, coveredEnd) / SLOT) * SLOT;
    if (end - start < DURATION) return [];
    // Keep the shared assessment as the final guard against incomplete coverage.
    if (assessGraphWindow({ payload, startAt: start, endAt: end, now }).status !== 'safe') return [];
    return [{ start, end, startTime: inputFormatter.format(start), latestStart: end - DURATION }];
  });
  return { status: windows.length ? 'available' : partialCoverage ? 'unavailable' : 'none', windows, partialCoverage,
    reason: windows.length ? '' : partialCoverage
      ? 'The forecast covers only part of this date, with no complete two-hour planting slot. You can still request a time for LGU review.'
      : `No two-hour planting slot fits the predicted lower-water periods${bookingPolicy.enforcePlantingHours ? ' within LIKE’s 6 AM–5 PM hours' : ''} on this date. Choose another date.` };
}
