# MangroVision logo assets

The approved artwork comes from the user's `LogoSystem.zip`, supplied on
2026-10-19. All seven original PNG files are stored unchanged in
`MangroVision_New/client/public/brand/`.

- `LogoPrimary.png`: green monogram for light backgrounds, planner and planter
  interfaces, browser icons, and PDF report headers.
- `LogoPrimaryWithName.png`: transparent green wordmark for staff sign-in,
  planter sign-in/onboarding, and the application splash screen.
- `LogoSecondary.png` and `LogoSecondaryWithName.png`: supplied white versions
  for dark backgrounds. The public website footer uses the white monogram.
- The three `WithBG` variants remain available in the same folder. Social
  previews use `LogoPrimaryWithBGAndName.png` for a solid background.

The shared `client/src/components/Logo.jsx` selects primary or secondary artwork
through `tone`. Icons fit a square without stretching; lockups retain their
original proportions. The public LIKE header/footer keep the ecopark's name
and use the supplied MangroVision monogram.

Existing root `logo-icon.png`, `logo-icon-small.png`, `logo-lockup.png`, and
`favicon.png` URLs contain the new artwork too, supporting older open clients.
New screens use the new `/brand/` paths so browser caches do not keep
the previous tree logo. No original image was recolored, cropped, or retouched.
