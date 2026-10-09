import { useCallback, useEffect, useMemo, useRef } from 'react';
import { Joyride, EVENTS } from 'react-joyride';
import { prepareGuideStep } from '../guides/tourRuntime';

function GuideTooltip({ backProps, closeProps, index, isLastStep, primaryProps, size, step, tooltipProps }) {
  return (
    <div {...tooltipProps} className="guide-tooltip">
      <div className="guide-tooltip-heading">
        <span className="guide-eyebrow">MANGROVISION GUIDE</span>
        <button {...closeProps} type="button" className="guide-icon-button" aria-label="End guide" title="End guide">×</button>
      </div>
      <p className="guide-step-count" aria-live="polite">Step {index + 1} of {size}</p>
      <h2>{step.title}</h2>
      <p className="guide-step-content">{step.content}</p>
      {step.data.isUnavailable() && <p className="guide-step-note">This control appears when the required data or previous step is ready. Follow these instructions when it becomes available.</p>}
      <div className="guide-tooltip-footer">
        <button {...closeProps} type="button" className="guide-text-button">End guide</button>
        <div>
          {index > 0 && <button {...backProps} type="button" className="guide-back-button">Back</button>}
          <button {...primaryProps} type="button" className="guide-next-button">{isLastStep ? 'Finish' : 'Next'}</button>
        </div>
      </div>
    </div>
  );
}

export default function GuideTour({ definitions, onDone, pathname, navigate }) {
  const pathRef = useRef(pathname);
  const navigateRef = useRef(navigate);
  const controllerRef = useRef(null);
  const doneRef = useRef(false);

  useEffect(() => { pathRef.current = pathname; }, [pathname]);
  useEffect(() => { navigateRef.current = navigate; }, [navigate]);
  useEffect(() => () => { controllerRef.current?.abort(); }, []);

  const steps = useMemo(() => definitions.map((definition) => {
    const resolved = { element: null, unavailable: false };
    return {
      title: definition.title,
      content: definition.content,
      target: () => resolved.element,
      placement: definition.placement || 'auto',
      // Joyride merges step data before the async hook. Read the resolved
      // state through a function so conditional-control notes stay current.
      data: { isUnavailable: () => resolved.unavailable },
      before: async () => {
        controllerRef.current?.abort();
        const controller = new AbortController();
        controllerRef.current = controller;
        const result = await prepareGuideStep(definition, {
          navigate: (path) => navigateRef.current?.(path), getPath: () => pathRef.current, signal: controller.signal,
        });
        if (result) Object.assign(resolved, result);
      },
    };
  }), [definitions]);

  const finish = useCallback(() => {
    if (doneRef.current) return;
    doneRef.current = true;
    controllerRef.current?.abort();
    onDone();
  }, [onDone]);

  const onEvent = useCallback((event) => {
    if (event.type === EVENTS.TOUR_END || event.type === EVENTS.ERROR) finish();
  }, [finish]);

  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        finish();
      }
    };
    document.addEventListener('keydown', onKeyDown, true);
    return () => document.removeEventListener('keydown', onKeyDown, true);
  }, [finish]);

  return <Joyride continuous run steps={steps} onEvent={onEvent} tooltipComponent={GuideTooltip}
    locale={{ back: 'Back', close: 'End guide', last: 'Finish', next: 'Next' }}
    options={{
      primaryColor: '#166534', textColor: '#173b2a', zIndex: 12000,
      overlayColor: 'rgba(10, 31, 21, 0.55)', width: 'min(380px, calc(100vw - 28px))',
      skipBeacon: true, blockTargetInteraction: true, closeButtonAction: 'skip',
      dismissKeyAction: false, overlayClickAction: false,
      skipScroll: true, beforeTimeout: 4000, targetWaitTimeout: 1000,
      spotlightPadding: 5, spotlightRadius: 10,
    }} />;
}
