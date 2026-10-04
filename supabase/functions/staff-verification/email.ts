import template from './email-template.json' with { type: 'json' };
import type { Challenge } from './logic.ts';

// A JSON module bundles with the function and is shared with local SMTP.
// All copy is trusted template text; the only dynamic value is a six-digit code.
export function verificationEmail(purpose: Challenge['purpose'], code: string) {
  if (!Object.hasOwn(template.copy, purpose) || !/^[0-9]{6}$/.test(code)) {
    throw new Error('Invalid verification email.');
  }
  const copy = template.copy[purpose];
  const values: Record<string, string> = { ...copy, code };
  const render = (body: string) => body.replace(/\{\{([a-z]+)\}\}/g, (_, key: string) => values[key]);
  return {
    subject: `MangroVision ${copy.label} code`,
    textContent: render(template.text),
    htmlContent: render(template.html.join('\n')),
  };
}
