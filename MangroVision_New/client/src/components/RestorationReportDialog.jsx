import { useEffect, useId, useRef } from 'react';
import { createPortal } from 'react-dom';
import RestorationReportWorkspace from '../pages/RestorationReports';

export default function RestorationReportDialog({ initialSelection, initialSites, onClose }) {
  const dialogRef = useRef(null);
  const titleId = useId();
  const descriptionId = useId();

  useEffect(() => {
    const dialog = dialogRef.current;
    const previousFocus = document.activeElement;
    dialog.showModal();
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      dialog.close();
      document.body.style.overflow = previousOverflow;
      if (previousFocus?.isConnected) previousFocus.focus();
    };
  }, []);

  return createPortal(
    <dialog ref={dialogRef} className="reports-dialog" aria-labelledby={titleId} aria-describedby={descriptionId}
      onCancel={(event) => { event.preventDefault(); onClose(); }}>
      <header className="reports-dialog-header">
        <div><h2 id={titleId}>Download report</h2><p id={descriptionId}>Review the dates and project site, add remarks, then download a copy.</p></div>
        <button type="button" className="reports-dialog-close" aria-label="Close report dialog" autoFocus onClick={onClose}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><path d="m6 6 12 12M6 18 18 6" /></svg>
        </button>
      </header>
      <div className="reports-dialog-body">
        <RestorationReportWorkspace initialSelection={initialSelection} initialSites={initialSites} />
      </div>
    </dialog>,
    document.body,
  );
}
