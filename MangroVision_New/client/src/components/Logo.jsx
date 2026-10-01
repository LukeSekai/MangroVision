/**
 * Brand mark renderer.
 *
 * Variants:
 *   - 'icon'    : icon-only (default), use for sidebar, processing badges,
 *                 modals, anywhere with limited horizontal space.
 *   - 'small'   : simplified icon for sizes below 48px (favicon, badges).
 *   - 'lockup'  : horizontal "icon + MangroVision" wordmark for wide
 *                 contexts (login, splash, report covers).
 *
 * The `size` prop controls the rendered HEIGHT in pixels; width auto-scales
 * via CSS so the lockup never gets stretched. Below 48px the component
 * automatically swaps to the 'small' icon for legibility.
 *
 * Brand integrity rules (don't change without design approval):
 *   - never recolor, rotate, or apply effects to the image
 *   - keep at least 25% of icon-width as clear space around the mark
 *   - always provide an alt for accessibility
 */
export default function Logo({
  variant = 'icon',
  size = 32,
  className = '',
  alt = 'MangroVision',
  style,
  ...rest
}) {
  let src;
  if (variant === 'lockup') {
    src = '/logo-lockup.png';
  } else if (variant === 'small' || (variant === 'icon' && size < 48)) {
    src = '/logo-icon-small.png';
  } else {
    src = '/logo-icon.png';
  }

  return (
    <img
      src={src}
      alt={alt}
      className={`mv-logo mv-logo-${variant} ${className}`}
      style={{ height: size, width: 'auto', display: 'block', ...style }}
      draggable={false}
      {...rest}
    />
  );
}
