// Original artwork from LogoSystem.zip. Icons fit a square; named lockups
// keep their source aspect ratio and shrink to fit narrow sign-in screens.
const ARTWORK = {
  primary: { icon: 'LogoPrimary.png', lockup: 'LogoPrimaryWithName.png', width: 2009, height: 783 },
  secondary: { icon: 'LogoSecondary.png', lockup: 'LogoSecondaryWithName.png', width: 2172, height: 724 },
};

export default function Logo({
  variant = 'icon',
  tone = 'primary',
  size = 32,
  className = '',
  alt = 'MangroVision',
  style,
  ...rest
}) {
  const artwork = ARTWORK[tone] || ARTWORK.primary;
  const lockup = variant === 'lockup';
  const dimensions = lockup ? { width: artwork.width, height: artwork.height } : { width: 1536, height: 1024 };

  return (
    <img
      src={`/brand/${lockup ? artwork.lockup : artwork.icon}`}
      alt={alt}
      {...dimensions}
      className={`mv-logo mv-logo-${variant} mv-logo-${tone} ${className}`}
      style={{ width: lockup ? size * artwork.width / artwork.height : size,
        height: lockup ? 'auto' : size, maxWidth: '100%', objectFit: 'contain', display: 'block', ...style }}
      draggable={false}
      {...rest}
    />
  );
}
