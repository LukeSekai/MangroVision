import { useRef, useState } from 'react';
import Modal from './Modal';
import { participantRecoveryCode } from '../utils/participantDevice';

export default function ParticipantRecovery({ planter, onClose }) {
  const input = useRef(null);
  const [message, setMessage] = useState('');
  let code = '', error = '';
  try { code = participantRecoveryCode(planter?.username); }
  catch (problem) { error = problem.message; }

  const copyCode = async () => {
    try {
      if (!navigator.clipboard?.writeText) throw new Error('Clipboard unavailable');
      await navigator.clipboard.writeText(code);
      setMessage('Recovery code copied. Save it somewhere private.');
    } catch {
      input.current?.select();
      setMessage('Code selected. Copy it and save it somewhere private.');
    }
  };

  return <Modal open title="Device recovery code" variant="info" confirmLabel="Done" cancelLabel="" onConfirm={onClose} onCancel={onClose}>
    {error ? <p role="alert">{error}</p> : <>
      <p>Ordinary return visits restore your points automatically on this browser and field link. This code is a backup for Participant {planter?.participant_slot} if the field link or browser changes.</p>
      <p>To use the backup, choose <strong>I have a device recovery code</strong> at sign-in and enter it with your organization's username and password.</p>
      <label className="field-label">Your private recovery code
        <input ref={input} className="field-input" aria-label="Device recovery code" value={code} readOnly onFocus={(event) => event.target.select()} />
      </label>
      <button type="button" className="btn btn-secondary btn-sm" onClick={copyCode}>Copy recovery code</button>
      <p>This restores your same assigned points and progress. Keep it private; another participant needs their own code.</p>
      {message && <p role="status">{message}</p>}
    </>}
  </Modal>;
}
