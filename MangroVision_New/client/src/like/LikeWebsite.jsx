import { useEffect, useRef, useState } from 'react';
import { bookingPayload, manilaToday, submitBooking } from './booking';
import { APPOINTMENT_TYPES, appointmentTypeLabel } from '../utils/appointmentTypes';
import LikeMapDialog from './LikeMapDialog';

function Icon({ name = 'leaf', size = 24, ...props }) {
  const paths = {
    leaf: <><path d="M20 4C9 3 3 8 5 15c2 7 13 6 15-11Z" /><path d="M4 21 15 10M9 16l-1-5m5 1 5 1" /></>,
    arrow: <><path d="M5 12h14m-6-6 6 6-6 6" /></>,
    pin: <><path d="M20 10c0 6-8 11-8 11S4 16 4 10a8 8 0 1 1 16 0Z" /><circle cx="12" cy="10" r="2.5" /></>,
    map: <><path d="m3 6 6-3 6 3 6-3v15l-6 3-6-3-6 3V6Z" /><path d="M9 3v15m6-12v15" /></>,
    people: <><circle cx="9" cy="8" r="3" /><path d="M3 20v-2a6 6 0 0 1 12 0v2M16 5a3 3 0 0 1 0 6m2 3a5 5 0 0 1 3 4v2" /></>,
    calendar: <><rect x="3" y="5" width="18" height="16" rx="3" /><path d="M7 3v4m10-4v4M3 11h18m-13 4h2m4 0h2" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    photo: <><rect x="3" y="3" width="18" height="18" rx="3" /><circle cx="8" cy="8" r="1.5" /><path d="m3 17 5-5 4 4 4-6 5 7" /></>,
    menu: <path d="M4 6h16M4 12h16M4 18h16" />,
    close: <path d="m6 6 12 12M6 18 18 6" />,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>{paths[name] || paths.leaf}</svg>;
}

function Brand({ light = false }) {
  return <a className={`like-brand${light ? ' is-light' : ''}`} href="#home" aria-label="LIKE home">
    <span className="like-brand-mark"><Icon size={26} /></span>
    <span><strong>LIKE<span className="like-brand-dot">.</span></strong><small>LEGANES · ILOILO</small></span>
  </a>;
}

function PhotoPlaceholder({ label, className = '', number }) {
  return <div className={`like-photo ${className}`} role="img" aria-label={`Photo placeholder: ${label}`}>
    <div className="like-photo-pattern" aria-hidden="true" />
    <span className="like-photo-label"><Icon name="photo" size={17} /> PHOTO PLACEHOLDER {number ? ` / ${number}` : ''}</span>
    <div className="like-photo-caption"><span>{label}</span><small>A little glimpse of LIKE, coming soon.</small></div>
  </div>;
}

function Photo({ file, alt, label, className = '', priority = false }) {
  return <div className={`like-photo has-image ${className}`}>
    <img src={`${import.meta.env.BASE_URL}like/${encodeURIComponent(file)}`} alt={alt} loading={priority ? 'eager' : 'lazy'} decoding="async" />
    {label ? <div className="like-photo-caption"><span>{label}</span></div> : null}
  </div>;
}

// Activity titles use the supplied photo filenames without their extensions.
const pastActivities = [
  {
    id: '01', photo: 'NGO Love Our Own Brethren (LOOB) Inc. Field Visit.jpg', date: null,
    description: 'A group of 21 Japanese youth and ISAT-U volunteers joined LOOB Inc. for mangrove bagging and a guided tour of LIKE, learning about conservation and coastal ecosystems.',
    alt: 'Visitors wearing yellow shirts posing together inside an ecopark viewing tower',
  },
  {
    id: '02', photo: 'University of the Philippines Visayas Field Visit.jpg', date: null,
    description: 'UPV’s STS class, led by Dr. Diana Paguntalan, and Tourism Management students from Green International Technological College visited LIKE to learn about mangrove conservation, ecotourism, and sustainable community development.',
    alt: 'A group posing indoors beside the LIKE sign with a green ecotourism banner',
  },
  {
    id: '03', photo: 'UPV IFPDS Field Visit.jpg', date: null,
    description: 'LIKE welcomed UPV IFPDS for a field visit focused on mangrove protection and sustainability, sharing how the ecopark supports environmental learning, ecotourism, and community empowerment.',
    alt: 'Visitors gathered in a viewing tower beneath a painted ceiling, with an IFPDS banner along the bottom',
  },
  {
    id: '04', photo: 'ZSL - Mangrove Caravan.jpg', date: null,
    description: 'Grade 9 students from Leganes National High School joined the Mangrove Caravan at LIKE for World Mangrove Day, exploring mangrove ecosystems and the importance of caring for them beyond planting.',
    alt: 'A group outside LIKE holding certificates beneath the Mangrove Caravan event heading',
  },
].map((activity) => ({ ...activity, title: activity.photo ? activity.photo.replace(/\.[^.]+$/, '') : 'Activity title to be added' }));

const initialForm = {
  organization: '', contact_name: '', phone: '', email: '', title: '',
  appointment_type: '',
  date: '', start_time: '', end_time: '', participants: '', notes: '', consent: false, website: '',
};

function AppointmentForm() {
  const [form, setForm] = useState(initialForm);
  const [step, setStep] = useState(1);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [receipt, setReceipt] = useState(null);
  const attempt = useRef(null);
  const lock = useRef(false);
  const firstField = useRef(null);
  const receiptCard = useRef(null);
  useEffect(() => {
    if (!receipt) return;
    receiptCard.current?.focus({ preventScroll: true });
    receiptCard.current?.scrollIntoView({ block: 'start' });
  }, [receipt]);
  const update = (event) => setForm((current) => ({ ...current, [event.target.name]: event.target.type === 'checkbox' ? event.target.checked : event.target.value }));

  const next = (event) => {
    event.preventDefault();
    setError('');
    setStep(2);
    window.setTimeout(() => firstField.current?.focus(), 0);
  };
  const submit = async (event) => {
    event.preventDefault();
    if (lock.current) return;
    setError('');
    try {
      // Retrying an identical request reuses its key, including after a lost response.
      const fingerprint = JSON.stringify(form);
      if (attempt.current?.fingerprint !== fingerprint) attempt.current = { fingerprint, key: crypto.randomUUID() };
      const payload = bookingPayload(form, attempt.current.key);
      lock.current = true;
      setBusy(true);
      setReceipt(await submitBooking(payload));
    } catch (issue) {
      setError(issue.message || 'We could not send your request. Check your connection and try again.');
    } finally {
      setBusy(false);
      lock.current = false;
    }
  };

  if (receipt) return <div className="like-receipt" ref={receiptCard} role="status" tabIndex={-1}>
    <span className="like-success-icon"><Icon name="check" size={32} /></span>
    <span className="like-eyebrow">YOUR NEXT CHAPTER STARTS HERE</span>
    <h3>Request received.</h3>
    <p>Thank you for wanting to be part of LIKE. Save your reference number below.</p>
    <strong className="like-reference">{receipt.reference}</strong>
    <p><b>Your appointment is pending LGU confirmation.</b> Staff will contact you to agree on the schedule. After approval, your confirmed date and time will be emailed to you. Tree planting appointments also include planter access details.</p>
    <button className="like-button is-secondary" type="button" onClick={() => { setReceipt(null); setForm(initialForm); setStep(1); attempt.current = null; }}>Make another request <Icon name="arrow" size={18} /></button>
  </div>;

  return <div className="like-booking-card">
    <div className="like-form-heading"><span className="like-eyebrow">LET’S PLAN YOUR VISIT</span><h3>A small step.<br />A lasting difference.</h3><p>Choose an activity and tell us about your group and preferred time.</p></div>
    <ol className="like-form-steps" aria-label="Appointment request steps">
      <li className={step >= 1 ? 'is-current' : ''}><span>{step === 2 ? <Icon name="check" size={14} /> : '1'}</span>Your group</li>
      <li className={step === 2 ? 'is-current' : ''}><span>2</span>Your preferred schedule</li>
    </ol>
    <form onSubmit={step === 1 ? next : submit}>
      {error ? <p className="like-form-error" role="alert">{error}</p> : null}
      {step === 1 ? <div className="like-form-fields" key="group">
        <label>Appointment type <span>*</span><select name="appointment_type" value={form.appointment_type} onChange={update} required><option value="" disabled>Choose your activity</option>{APPOINTMENT_TYPES.map((type) => <option key={type.value} value={type.value}>{type.label}</option>)}</select></label>
        <label>Organization or group name <span>*</span><input name="organization" value={form.organization} onChange={update} required maxLength={200} autoComplete="organization" placeholder="e.g. Your school, company, or community" /></label>
        <label>Contact person <span>*</span><input name="contact_name" value={form.contact_name} onChange={update} required maxLength={100} autoComplete="name" placeholder="Full name" /></label>
        <div className="like-form-row">
          <label>Phone number <span>*</span><input name="phone" type="tel" value={form.phone} onChange={update} required minLength={7} maxLength={32} autoComplete="tel" placeholder="09XX XXX XXXX" /></label>
          <label>Email <span>*</span><input name="email" type="email" value={form.email} onChange={update} required maxLength={254} autoComplete="email" placeholder="you@example.com" /></label>
        </div>
        <p className="like-field-hint">Use an email address you can access. Your appointment confirmation will be sent here.</p>
        <button className="like-button" type="submit">Continue to schedule <Icon name="arrow" size={18} /></button>
      </div> : <div className="like-form-fields" key="schedule">
        <label>Preferred date <span>*</span><input ref={firstField} name="date" type="date" value={form.date} onChange={update} required min={manilaToday()} /></label>
        <div className="like-form-row">
          <label>Start time <span>*</span><input name="start_time" type="time" value={form.start_time} onChange={update} required /></label>
          <label>End time <span>*</span><input name="end_time" type="time" value={form.end_time} onChange={update} required /></label>
        </div>
        <p className="like-field-hint">All dates and times are in Philippine time (Asia/Manila).</p>
        <div className="like-form-row">
          <label>Participants <span>*</span><input name="participants" type="number" min="1" max="10000" step="1" value={form.participants} onChange={update} required placeholder="How many are joining?" /></label>
          <label>Activity name <small>optional</small><input name="title" value={form.title} onChange={update} maxLength={200} placeholder={appointmentTypeLabel(form.appointment_type)} /></label>
        </div>
        <label>Anything we should know? <small>optional</small><textarea name="notes" value={form.notes} onChange={update} maxLength={2000} rows={3} placeholder="Tell us about your group or any questions you have." /></label>
        <label className="like-honeypot" aria-hidden="true">Website<input name="website" value={form.website} onChange={update} tabIndex={-1} autoComplete="off" /></label>
        <label className="like-consent"><input name="consent" type="checkbox" checked={form.consent} onChange={update} required /><span>I understand this is a request, subject to LGU confirmation. LIKE staff may use my contact details to coordinate this appointment.</span></label>
        <div className="like-form-actions"><button className="like-back" type="button" disabled={busy} onClick={() => { setStep(1); setError(''); }}>Back</button><button className="like-button" type="submit" disabled={busy}>{busy ? 'Sending request…' : 'Send appointment request'}<Icon name="arrow" size={18} /></button></div>
      </div>}
    </form>
    <p className="like-form-footnote"><Icon name="calendar" size={16} /> A request today. A confirmed schedule after LGU review.</p>
  </div>;
}

export default function LikeWebsite() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [mapOpen, setMapOpen] = useState(false);
  const root = useRef(null);
  useEffect(() => {
    if (!('IntersectionObserver' in window)) return;
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) { entry.target.classList.add('is-visible'); observer.unobserve(entry.target); }
      });
    }, { threshold: 0.12 });
    root.current?.querySelectorAll('[data-reveal]').forEach((node) => {
      node.classList.add('will-reveal'); observer.observe(node);
    });
    return () => observer.disconnect();
  }, []);
  const closeMenu = () => setMenuOpen(false);
  return <div className="like-site" ref={root}>
    <a className="like-skip" href="#main">Skip to content</a>
    <header className="like-header"><div className="like-container like-header-inner">
      <Brand />
      <button className="like-menu-toggle" type="button" aria-label={menuOpen ? 'Close navigation' : 'Open navigation'} aria-expanded={menuOpen} aria-controls="like-navigation" onClick={() => setMenuOpen((open) => !open)}><Icon name={menuOpen ? 'close' : 'menu'} /></button>
      <nav id="like-navigation" className={menuOpen ? 'is-open' : ''} aria-label="Main navigation">
        <a href="#about" onClick={closeMenu}>About LIKE</a><a href="#experience" onClick={closeMenu}>The experience</a><a href="#activities" onClick={closeMenu}>Past activities</a><a href="#guidelines" onClick={closeMenu}>Plan your visit</a>
        <button className="like-nav-map" type="button" aria-haspopup="dialog" onClick={() => { closeMenu(); setMapOpen(true); }}>View map</button>
        <a className="like-button is-small" href="#appointment" onClick={closeMenu}>Book an appointment <Icon name="arrow" size={16} /></a>
      </nav>
    </div></header>

    <main id="main">
      <section className="like-hero" id="home"><div className="like-container like-hero-grid">
        <div className="like-hero-copy"><span className="like-eyebrow"><span className="like-live-dot" /> ROOTED IN LEGANES. GROWING TOGETHER.</span>
          <h1>A little seedling.<br />A <em>greener</em><br />tomorrow.</h1>
          <p>Welcome to Leganes Integrated Katunggan Ecopark. Discover the mangroves, meet the community, and be part of something that grows.</p>
          <div className="like-hero-actions"><a className="like-button" href="#appointment">Book an appointment <Icon name="arrow" size={19} /></a><button className="like-button is-secondary like-view-map" type="button" aria-haspopup="dialog" onClick={() => setMapOpen(true)}>View map <Icon name="map" size={19} /></button><a className="like-text-link" href="#about">Get to know LIKE <span>↗</span></a></div>
          <div className="like-hero-location"><span><Icon name="pin" size={18} /></span><div><strong>A greener corner of Leganes</strong><small>Leganes, Iloilo · Philippines</small></div></div>
        </div>
        <div className="like-hero-visual"><Photo className="like-hero-photo" file="hero.jpg" alt="Aerial view of LIKE's mangrove forest, elevated walkways, viewing towers, and coastal pavilions" label="The mangroves of LIKE" priority />
          <div className="like-visual-stamp"><Icon size={22} /><span>Small roots.<br /><b>Lasting impact.</b></span></div>
          <div className="like-visual-note"><span>01 / THE ECOPARK</span><p>Where the land meets the tide,<br />and a community takes root.</p></div>
          <span className="like-orbit" aria-hidden="true" />
        </div>
      </div><div className="like-container like-hero-bottom"><span>COME CURIOUS. LEAVE CONNECTED.</span><a href="#about" aria-label="Scroll to discover LIKE">SCROLL TO DISCOVER <span>↓</span></a></div></section>

      <div className="like-values-strip" aria-label="Our focus"><div className="like-container"><span><Icon /> Mangrove conservation</span><i /><span><Icon name="people" /> Community participation</span><i /><span><Icon name="calendar" /> Meaningful planting experiences</span></div></div>

      <section className="like-section like-about" id="about"><div className="like-container like-about-grid">
        <div className="like-about-images" data-reveal><Photo file="about.jpg" alt="A viewing tower and boardwalk among mangroves overlooking the sea at LIKE" label="A view from the ecopark" /><div className="like-about-tag"><Icon /><span>Rooted in nature.<br /><strong>Made for community.</strong></span></div></div>
        <div className="like-about-copy" data-reveal><span className="like-eyebrow">MORE THAN A PLACE TO VISIT</span><h2>A place to connect.<br />A reason to <em>care.</em></h2><p>LIKE brings Leganes’ mangrove conservation efforts and community participation together. It is a place to learn about the coast and take part in caring for it.</p><p>Whether you’re coming with a school, an organization, or your community, your next planting activity can start here.</p><a className="like-text-link" href="#experience">Find your way to take part <Icon name="arrow" size={18} /></a><div className="like-about-signature"><span />LEGANES INTEGRATED KATUNGGAN ECOPARK</div></div>
      </div></section>

      <section className="like-section like-experience" id="experience"><div className="like-container">
        <div className="like-section-heading" data-reveal><div><span className="like-eyebrow">A LITTLE CLOSER TO NATURE</span><h2>Come for the mangroves.<br />Stay for the <em>meaning.</em></h2></div><p>Make room for a day that connects your group to the coast, and to each other.</p></div>
        <div className="like-experience-grid">{[
          ['01', 'Plant with purpose', 'Bring your group together for a mangrove planting activity coordinated with LIKE and the LGU.', 'Mangrove planting in action', 'leaf', 'planting.jpg', 'An adult and child planting a mangrove seedling together in the mud'],
          ['02', 'Learn from the coast', 'Get to know the mangroves and the role they play in the environment around us.', 'Discovering the mangroves', 'pin', 'mangroves.jpg', 'Visitors walking along a narrow boardwalk surrounded by green mangroves'],
          ['03', 'Grow as a community', 'Share an experience with the people working to care for Leganes’ mangrove areas.', 'Community at the ecopark', 'people', 'community.jpg', 'A group of visitors posing together beside the LIKE ecopark sign'],
        ].map(([number, title, description, photo, icon, file, alt]) => <article className="like-experience-card" key={number} data-reveal><Photo file={file} alt={alt} label={photo} /><div className="like-experience-text"><div><span>{number}</span><Icon name={icon} /></div><h3>{title}</h3><p>{description}</p></div></article>)}</div>
      </div></section>

      <section className="like-invitation"><div className="like-container" data-reveal><span className="like-eyebrow">LET’S GROW SOMETHING GOOD</span><h2>The next chapter of the coast<br />could start with <em>you.</em></h2><a className="like-button is-white" href="#appointment">Request a planting appointment <Icon name="arrow" size={18} /></a><span className="like-invitation-leaf" aria-hidden="true"><Icon size={240} /></span></div></section>

      <section className="like-section like-activities" id="activities" aria-labelledby="like-activities-heading"><div className="like-container">
        <div className="like-section-heading" data-reveal><div><span className="like-eyebrow">MOMENTS FROM LIKE</span><h2 id="like-activities-heading">Past activities.<br /><em>Lasting</em> memories.</h2></div><p>A space for the planting days, visits, and community activities at the ecopark.</p></div>
        <div className="like-activities-grid">{pastActivities.map((activity) => <article className="like-activity-card" key={activity.id} data-reveal>
          {activity.photo ? <Photo file={activity.photo} alt={activity.alt} /> : <PhotoPlaceholder label="Activity photo" number={activity.id} />}
          <div className="like-activity-text"><div className="like-activity-meta"><span>ACTIVITY {activity.id}</span><span><Icon name="calendar" size={13} />{activity.date ? <time dateTime={activity.date}>{new Date(`${activity.date}T00:00:00+08:00`).toLocaleDateString('en-PH', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'Asia/Manila' })}</time> : 'Date to be added'}</span></div><h3>{activity.title}</h3><p>{activity.description}</p></div>
        </article>)}</div>
      </div></section>

      <section className="like-section like-guidelines" id="guidelines"><div className="like-container like-guidelines-grid">
        <div data-reveal><span className="like-eyebrow">A LITTLE PREPARATION GOES A LONG WAY</span><h2>Plan a visit.<br />Make it <em>count.</em></h2><p>LIKE and the LGU will help coordinate your planting activity. Start with a request, and we’ll work out the details together.</p><div className="like-walk-in"><Icon name="people" /><div><strong>Prefer a little help?</strong><p>You can still arrange an activity directly with LGU staff. They can enter your schedule for you in MangroVision.</p></div></div></div>
        <div className="like-faq" data-reveal>{[
          ['Is my appointment confirmed when I submit?', 'Your submission is a request. The LGU reviews your preferred time, staff availability, existing activities, and planting conditions before contacting you for confirmation.'],
          ['What appointments can we request?', 'Choose a field visit, clean-up drive, or tree planting activity. Each request is reviewed by the LGU before confirmation.'],
          ['Can we choose our preferred time?', 'Yes. Tell us your preferred date, start time, and end time. Staff check availability and site conditions. For tree planting, this includes the tide, so they may suggest a different time.'],
          ['What should our group prepare?', 'Ask LGU staff about appropriate clothing, footwear, materials, and any activity requirements when they confirm your appointment. Final visitor guidelines will be added here once approved.'],
          ['Do we need an account to send a request?', 'No account is needed. Provide your group and contact details so the LGU can coordinate with you.'],
        ].map(([question, answer], index) => <details key={question} open={index === 0}><summary>{question}<span aria-hidden="true">+</span></summary><p>{answer}</p></details>)}</div>
      </div></section>

      <section className="like-section like-appointment" id="appointment"><div className="like-container like-appointment-grid">
        <div className="like-appointment-copy" data-reveal><span className="like-eyebrow">FROM INTEREST TO IMPACT</span><h2>Bring your people.<br />We’ll help with<br />the <em>next step.</em></h2><p>Request a field visit, clean-up drive, or tree planting activity for your group. The LGU will review your preferred schedule and contact you to coordinate.</p><ol className="like-process"><li><span>01</span><div><h3>Tell us about your group</h3><p>A few details and a way to reach you.</p></div></li><li><span>02</span><div><h3>Choose a preferred schedule</h3><p>Pick a date and time that work for you.</p></div></li><li><span>03</span><div><h3>Hear from the LGU</h3><p>Staff review, contact you, and confirm the details.</p></div></li></ol><div className="like-contact-card"><Icon name="pin" /><div><strong>Leganes, Iloilo, Philippines</strong><p>Official contact number, email, and visiting hours will be added here.</p></div></div></div>
        <AppointmentForm />
      </div></section>
    </main>
    {mapOpen ? <LikeMapDialog onClose={() => setMapOpen(false)} /> : null}

    <footer className="like-footer"><div className="like-container"><div className="like-footer-top"><div><Brand light /><p>Small roots. Shared responsibility.<br />A greener tomorrow for Leganes.</p></div><div><span>EXPLORE</span><a href="#about">About the ecopark</a><a href="#activities">Past activities</a><a href="#guidelines">Visitor information</a></div><div><span>TAKE PART</span><a href="#appointment">Request an appointment</a><a href="#guidelines">Staff-assisted scheduling</a><a href="/dashboard">MangroVision staff sign-in ↗</a></div></div><div className="like-footer-bottom"><span>© {new Date().getFullYear()} LIKE · Leganes Integrated Katunggan Ecopark</span><span>GROWING TOGETHER, ONE SEEDLING AT A TIME.</span></div></div></footer>
  </div>;
}
