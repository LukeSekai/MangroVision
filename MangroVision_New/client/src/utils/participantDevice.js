const DEVICE_KEY = 'mv_participant_device';
const COOKIE_MAX_AGE = 365 * 24 * 60 * 60;

function storageKey(username) {
  const account = String(username || '').trim().toLowerCase();
  return account ? `${DEVICE_KEY}:${account}` : DEVICE_KEY;
}

function validKey(key) {
  return typeof key === 'string' && key.length >= 16 && key.length <= 200;
}

function readLocal(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}

function readCookie(key) {
  try {
    const prefix = `${encodeURIComponent(key)}=`;
    const value = document.cookie.split(';').map((part) => part.trim()).find((part) => part.startsWith(prefix));
    return value ? decodeURIComponent(value.slice(prefix.length)) : null;
  } catch { return null; }
}

export function readParticipantDeviceKey(username) {
  const key = storageKey(username);
  return [readLocal(key), readCookie(key), readLocal(DEVICE_KEY), readCookie(DEVICE_KEY)].find(validKey) || null;
}

export function rememberParticipantDeviceKey(identity, username) {
  if (!validKey(identity)) throw new Error('Invalid device identity.');
  const key = storageKey(username);
  const keys = [key];
  // Retain the original global identity for existing accounts on this browser.
  // Importing one organization's code must not change another's identity.
  if (key !== DEVICE_KEY && !validKey(readLocal(DEVICE_KEY)) && !validKey(readCookie(DEVICE_KEY))) keys.push(DEVICE_KEY);
  for (const name of keys) {
    try { localStorage.setItem(name, identity); } catch { /* Cookies can persist the identity instead. */ }
    try {
      const secure = document.location.protocol === 'https:' ? '; Secure' : '';
      document.cookie = `${encodeURIComponent(name)}=${encodeURIComponent(identity)}; Path=/; Max-Age=${COOKIE_MAX_AGE}; SameSite=Lax${secure}`;
    } catch { /* Local storage can persist the identity instead. */ }
  }
  if (readLocal(key) !== identity && readCookie(key) !== identity) {
    throw new Error('Allow cookies or browser storage before signing in so your participant number can be saved.');
  }
  return identity;
}

export function participantDeviceKey(username) {
  const existing = readParticipantDeviceKey(username);
  const identity = existing || Array.from(crypto.getRandomValues(new Uint8Array(24)), (byte) => byte.toString(16).padStart(2, '0')).join('');
  return rememberParticipantDeviceKey(identity, username);
}

export function participantRecoveryCode(username) {
  // Never generate a new identity when displaying an existing participant's code.
  const identity = readParticipantDeviceKey(username);
  if (!/^[a-f0-9]{48}$/i.test(identity || '')) {
    throw new Error('This browser no longer has its device identity. Ask the LGU to recover your participant number.');
  }
  return `MV1-${identity}`;
}

export function parseParticipantRecoveryCode(code) {
  const compact = String(code || '').trim().replace(/[\s-]/g, '');
  const match = /^MV1([a-f0-9]{48})$/i.exec(compact);
  if (!match) throw new Error('Enter the complete device recovery code beginning with MV1.');
  return match[1].toLowerCase();
}
