/** Calendar-only reminder calculations, shared by the Edge function and tests. */

export type SitePlanting = {
  id: string;
  planting_point_id: string | null;
  project_site_id: string | null;
  site_name: string;
  planted_day: string;
  observed_intervals: number[];
};

export type OrganizationPlanting = {
  organization_id: string;
  organization_name: string;
  planted_day: string;
  latest_day: string | null;
};

export type ReminderDigest = {
  eventKey: string;
  subject: string;
  body: string;
  sendOn: string;
  dueDay: string;
};

const DAY_MS = 24 * 60 * 60 * 1000;
const PHASES = [
  { daysBefore: 3, key: "three_days_before", phrase: "in 3 days" },
  { daysBefore: 1, key: "tomorrow", phrase: "tomorrow" },
  { daysBefore: 0, key: "today", phrase: "today" },
] as const;

function utcDay(day: string): Date {
  return new Date(`${day}T00:00:00.000Z`);
}

export function addDays(day: string, amount: number): string {
  return new Date(utcDay(day).getTime() + amount * DAY_MS).toISOString().slice(0, 10);
}

export function daysBetween(first: string, second: string): number {
  return Math.round((utcDay(second).getTime() - utcDay(first).getTime()) / DAY_MS);
}

export function manilaDay(now: Date = new Date()): string {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "Asia/Manila", year: "numeric", month: "2-digit", day: "2-digit",
  }).formatToParts(now);
  const part = (type: string) => parts.find((item) => item.type === type)?.value;
  return `${part("year")}-${part("month")}-${part("day")}`;
}

export function isWorkday(day: string): boolean {
  const weekday = utcDay(day).getUTCDay();
  return weekday !== 0 && weekday !== 6;
}

export function monitoringDay(plantedDay: string, round: number): string {
  const nominal = addDays(plantedDay, 14 * round);
  const weekday = utcDay(nominal).getUTCDay();
  return addDays(nominal, weekday === 6 ? 2 : weekday === 0 ? 1 : 0);
}

export function roundDueOn(plantedDay: string, dueDay: string): number | null {
  const round = Math.floor(daysBetween(plantedDay, dueDay) / 14);
  return round >= 1 && monitoringDay(plantedDay, round) === dueDay ? round : null;
}

export function organizationNextDay(rows: OrganizationPlanting[]): string | null {
  let next: string | null = null;
  for (const row of rows) {
    const elapsed = row.latest_day === null ? -1 : daysBetween(row.planted_day, row.latest_day);
    const round = Math.max(1, Math.floor(elapsed / 14) + 1);
    const due = monitoringDay(row.planted_day, round);
    if (next === null || due < next) next = due;
  }
  return next;
}

export function buildMonitoringDigests(
  today: string,
  sitePlantings: SitePlanting[],
  organizationPlantings: OrganizationPlanting[],
): ReminderDigest[] {
  // Reminder phases use calendar days, but LGU email is never sent on weekends.
  // A weekend phase is skipped rather than moved, which avoids two emails on
  // Friday when Monday monitoring has both a Friday 3-day and Sunday 1-day phase.
  if (!isWorkday(today)) return [];
  const organizations = new Map<string, OrganizationPlanting[]>();
  for (const row of organizationPlantings) {
    const group = organizations.get(row.organization_id) ?? [];
    group.push(row);
    organizations.set(row.organization_id, group);
  }

  const digests: ReminderDigest[] = [];
  for (const phase of PHASES) {
    const dueDay = addDays(today, phase.daysBefore);
    const sites = new Map<string, { name: string; pointKeys: Set<string> }>();
    for (const row of sitePlantings) {
      const round = roundDueOn(row.planted_day, dueDay);
      if (round === null || row.observed_intervals.some((value) => Number(value) === round * 14)) continue;
      const siteKey = row.project_site_id ?? "unknown";
      const site = sites.get(siteKey) ?? { name: row.site_name, pointKeys: new Set<string>() };
      site.pointKeys.add(row.planting_point_id === null ? `event:${row.id}` : `point:${row.planting_point_id}`);
      sites.set(siteKey, site);
    }

    const siteLines = [...sites.values()]
      .sort((a, b) => a.name.localeCompare(b.name))
      .map((site) => `- ${site.name}: ${site.pointKeys.size} point(s)`);
    const organizationLines = [...organizations.values()]
      .filter((rows) => organizationNextDay(rows) === dueDay)
      .map((rows) => `- ${rows[0].organization_name}`)
      .sort((a, b) => a.localeCompare(b));
    if (siteLines.length === 0 && organizationLines.length === 0) continue;

    const body = [`MangroVision monitoring is due ${phase.phrase} (${dueDay}, Asia/Manila).`];
    if (siteLines.length) body.push("", "Site seedling inspections:", ...siteLines);
    if (organizationLines.length) body.push("", "Organization monitoring visits:", ...organizationLines);
    body.push("", "Open MangroVision to record the monitoring work.");
    digests.push({
      eventKey: `monitoring:${dueDay}:${phase.key}`,
      subject: `MangroVision monitoring ${phase.phrase} - ${dueDay}`,
      body: body.join("\n"),
      sendOn: today,
      dueDay,
    });
  }
  return digests;
}
