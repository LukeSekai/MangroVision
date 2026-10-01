export function monitoringCounts(plantedValue, deadValue, previousDeadValue = 0) {
  const planted = Number(plantedValue);
  const previousDead = Number(previousDeadValue);
  const invalid = (error) => ({ planted, alive: null, dead: null, survival: null, error });
  if (!Number.isInteger(planted) || planted <= 0) return invalid('This organization has no planted seedlings to monitor yet.');
  if (deadValue === '' || deadValue === null || deadValue === undefined || String(deadValue).trim() === '') {
    return invalid('Enter newly dead seedlings, including 0 if no more have died.');
  }
  const newlyDead = Number(deadValue);
  if (!Number.isInteger(previousDead) || previousDead < 0 || previousDead > planted) return invalid('Reopen the organization to load its saved counts.');
  const available = planted - previousDead;
  if (!Number.isInteger(newlyDead) || newlyDead < 0 || newlyDead > available) return invalid(`New deaths must be a whole number from 0 to ${available}.`);
  const dead = previousDead + newlyDead;
  const alive = planted - dead;
  return { planted, dead, newlyDead, alive, survival: alive / planted * 100, error: '' };
}

export function visitFormFromLatest(latest, today) {
  return { monitored_at: today, dead_count: '0',
    death_reason_category: '', death_reason_notes: '',
    health_status: latest?.health_status || '', actions_taken: latest?.actions_taken || '' };
}
