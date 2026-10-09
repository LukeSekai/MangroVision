export function isRendered(element) {
  if (!element || !element.isConnected || !element.getClientRects().length) return false;
  for (let node = element; node && node.nodeType === 1; node = node.parentElement) {
    const style = window.getComputedStyle(node);
    if (style.display === 'none' || style.visibility === 'hidden') return false;
  }
  return true;
}

export function findRendered(selector) {
  return [...document.querySelectorAll(selector)].find(isRendered) || null;
}

function delay(ms, signal) {
  return new Promise((resolve) => {
    if (signal.aborted) { resolve(); return; }
    const finish = () => {
      window.clearTimeout(timer);
      signal.removeEventListener('abort', finish);
      resolve();
    };
    const timer = window.setTimeout(finish, ms);
    signal.addEventListener('abort', finish, { once: true });
  });
}

function revealSection(element) {
  let changed = false;
  const panel = element.closest('.floating-panel');
  if (panel?.classList.contains('floating-panel-hidden')) {
    // The paired toggle is the Panel's previous sibling, never a submit action.
    panel.previousElementSibling?.matches('.panel-toggle') && panel.previousElementSibling.click();
    changed = true;
  }
  const card = element.closest('.panel-card');
  const header = card?.querySelector(':scope > .panel-card-header');
  if (header?.getAttribute('aria-expanded') === 'false') {
    header.click();
    changed = true;
  }
  return changed;
}

/** Wait for route rendering, reveal only panel UI, and preserve conditional tips. */
export async function prepareGuideStep(definition, { navigate, getPath, signal, timeout = 1800 }) {
  if (signal.aborted) return null;
  if (definition.path && definition.path !== getPath()) navigate(definition.path);
  const deadline = Date.now() + timeout;
  while (!signal.aborted && Date.now() < deadline) {
    if (!definition.path || getPath() === definition.path) {
      // A collapsed card still has a rendered header; its fields may have no size.
      const candidates = [...document.querySelectorAll(definition.target)];
      const element = candidates.find((node) => isRendered(node)
        || isRendered(node.closest('.panel-card')));
      if (element) {
        if (revealSection(element)) await delay(320, signal);
        if (signal.aborted) return null;
        if (isRendered(element) && !element.closest('[inert], [aria-hidden="true"]')) {
          // Anchor section tips to their headings. A section containing many
          // records can be taller than the screen and has no usable outer edge.
          const anchor = element.matches('.panel-card')
            ? element.querySelector(':scope > .panel-card-header') || element
            : element;
          anchor.scrollIntoView?.({ block: 'center', inline: 'nearest', behavior: 'instant' });
          await delay(80, signal);
          return signal.aborted ? null : { element: anchor, unavailable: false };
        }
      }
    }
    await delay(40, signal);
  }
  if (signal.aborted) return null;
  // Keep the instructions for results, registrations, or controls that are not
  // available yet, rather than dropping that part of the workflow.
  const element = findRendered('.workspace-header, .field-header, .login-form-header, .field-auth-brand');
  return element ? { element, unavailable: true } : null;
}
