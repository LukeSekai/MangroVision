// Pasted codes can contain a typographic dash, especially on mobile keyboards.
export const normalizePlanterCode = (value) => String(value || '').replace(/[\s\-\u2010-\u2015\u2212\ufe63\uff0d]/g, '').toUpperCase();

export const isPlanterCode = (value) => /^[A-HJ-NP-Z2-9]{8}$/.test(normalizePlanterCode(value));
