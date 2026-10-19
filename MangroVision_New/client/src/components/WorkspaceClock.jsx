import { useEffect, useState } from 'react';

export default function WorkspaceClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  const date = now.toLocaleDateString('en-PH', {
    timeZone: 'Asia/Manila', weekday: 'short', month: 'short', day: 'numeric',
  });
  const time = now.toLocaleTimeString('en-PH', {
    timeZone: 'Asia/Manila', hour: 'numeric', minute: '2-digit', second: '2-digit', hour12: true,
  });
  return <time className="workspace-clock" dateTime={now.toISOString()} title={`${date}, ${time} — Philippine time`}>
    <svg className="workspace-clock-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></svg>
    <span className="workspace-clock-text"><span className="workspace-clock-date">{date}</span><span className="workspace-clock-time">{time}</span></span>
  </time>;
}
