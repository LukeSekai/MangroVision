import { create } from 'zustand';

const API = import.meta.env.VITE_API_BASE || '';

// AbortController and the active stream Reader are held outside the React
// state because they are not serializable and we never want them to trigger a
// re-render. The store just exposes start / cancel actions and the
// derived state for the UI.
let activeController = null;
let activeReader = null;

const initialState = {
  // True from the moment the user clicks Process until a result or error.
  // Survives page navigation because it lives here, not in the page component.
  processing: false,

  // Latest stage label and percent emitted by the backend SSE stream.
  stage: null,
  progress: 0,

  // Final analysis payload (the same shape the /process endpoint returns).
  result: null,

  // Error message if the stream failed; null while everything is fine.
  error: '',

  // Save-to-database state, also moved off the page so a save in flight
  // survives navigation.
  saving: false,
  saveError: '',

  // File metadata kept in the store so the indicator can show the name
  // and the panel can rehydrate after navigation. The actual File blob is
  // not persisted because once the upload starts the bytes have already
  // been streamed to the backend.
  fileName: '',
  previewUrl: '',

  // Toggle for whether the result overlay should auto-open on the
  // ImageProcessing page once processing completes.
  overlayOpen: false,
};

export const useProcessingStore = create((set, get) => ({
  ...initialState,

  setOverlayOpen: (open) => set({ overlayOpen: Boolean(open) }),

  // Reset back to a clean slate. Used by "Process Another Image" / "Clear".
  reset: () => {
    if (activeController) {
      try {
        activeController.abort();
      } catch (_err) { /* noop */ }
    }
    activeController = null;
    activeReader = null;
    set({ ...initialState });
  },

  // Cancel an in-flight processing job without clearing already-saved state
  // (kept for an eventual cancel button; currently unused).
  cancel: () => {
    if (activeController) {
      try {
        activeController.abort();
      } catch (_err) { /* noop */ }
    }
    activeController = null;
    activeReader = null;
    set({ processing: false, stage: null, progress: 0 });
  },

  // Apply a fresh result from the most recent run, replacing whatever was
  // there before.
  setResult: (result) => set({ result, overlayOpen: Boolean(result) }),

  // Mark a save success — used by the save flow after the API call returns.
  applySaveSuccess: (savedResult) =>
    set({
      result: savedResult,
      saving: false,
      saveError: '',
    }),

  // Start processing. `params` carries the configuration (file, altitude,
  // drone_model, canopy_buffer, hexagon_size, ai_confidence). The promise
  // resolves once processing terminates either way; callers can fire it and
  // forget. Throws nothing — errors land in `state.error`.
  startProcess: async (params) => {
    const { file } = params;
    if (!file) return;

    // Abort any previous run.
    if (activeController) {
      try {
        activeController.abort();
      } catch (_err) { /* noop */ }
    }

    const controller = new AbortController();
    activeController = controller;

    // Build a tiny preview URL for the indicator and the panel without
    // copying the entire blob into Zustand state.
    const previewUrl = await new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (event) => resolve(event.target?.result || '');
      reader.onerror = () => resolve('');
      reader.readAsDataURL(file);
    });

    set({
      processing: true,
      stage: 'Uploading image...',
      progress: 1,
      result: null,
      error: '',
      saveError: '',
      fileName: file.name || 'drone_image.jpg',
      previewUrl,
      overlayOpen: false,
    });

    try {
      const formData = new FormData();
      formData.append('image', file);
      formData.append('altitude', String(params.altitude));
      formData.append('drone_model', params.drone_model);
      formData.append('canopy_buffer', String(params.canopy_buffer));
      formData.append('hexagon_size', String(params.hexagon_size));
      formData.append('ai_confidence', String(params.ai_confidence));
      formData.append('ai_runtime_tuning', JSON.stringify(params.ai_runtime_tuning || {}));

      const response = await fetch(`${API}/api/analyses/process-stream`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `Processing failed (${response.status})`);
      }

      const reader = response.body.getReader();
      activeReader = reader;
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let finalPayload = null;
      let streamError = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let boundary = buffer.indexOf('\n\n');
        while (boundary !== -1) {
          const rawEvent = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);

          const dataLine = rawEvent
            .split('\n')
            .find((line) => line.startsWith('data:'));
          if (dataLine) {
            const jsonText = dataLine.slice(5).trim();
            if (jsonText) {
              let event;
              try {
                event = JSON.parse(jsonText);
              } catch {
                event = null;
              }
              if (event) {
                if (event.type === 'progress') {
                  const next = {};
                  if (typeof event.stage === 'string') next.stage = event.stage;
                  if (typeof event.pct === 'number') next.progress = event.pct;
                  if (Object.keys(next).length) set(next);
                } else if (event.type === 'result') {
                  finalPayload = event.payload;
                } else if (event.type === 'error') {
                  streamError = event.detail || 'Processing failed';
                }
              }
            }
          }
          boundary = buffer.indexOf('\n\n');
        }
      }

      if (streamError) throw new Error(streamError);
      if (!finalPayload) throw new Error('Processing ended without a result');

      set({
        processing: false,
        stage: 'Complete!',
        progress: 100,
        result: finalPayload,
        overlayOpen: true,
      });
    } catch (processError) {
      if (processError.name === 'AbortError') {
        // User-initiated abort: just clear the running flags, no error message.
        set({ processing: false, stage: null, progress: 0 });
      } else {
        set({
          processing: false,
          stage: null,
          progress: 0,
          error: processError.message || 'Processing failed',
        });
      }
    } finally {
      activeController = null;
      activeReader = null;
    }
  },

  // Save the current result to the database. Survives navigation in the same
  // way as processing.
  saveCurrentAnalysis: async (userId) => {
    const result = get().result;
    if (!result?.analysis_key || get().saving || result.saved) return;

    set({ saving: true, saveError: '' });

    try {
      const response = await fetch(`${API}/api/analyses/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          analysis_key: result.analysis_key,
          user_id: userId ?? null,
        }),
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || 'Save failed');
      }

      const payload = await response.json();
      const savedResult = {
        ...result,
        saved: true,
        analysis_id: payload.analysis_id,
        save_summary: {
          analysis_id: payload.analysis_id,
          new_points: payload.new_points,
          skipped_duplicates: payload.skipped_duplicates,
        },
      };
      set({ result: savedResult, saving: false, saveError: '' });
      return { savedResult, savePayload: payload };
    } catch (err) {
      set({ saving: false, saveError: err.message || 'Save failed' });
      return null;
    }
  },
}));
