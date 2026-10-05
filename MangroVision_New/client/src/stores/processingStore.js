import { create } from 'zustand';
import { readAnalysisResponse, startProcessingJob, waitForProcessingJob } from '../utils/processingJobs';

const API = import.meta.env.VITE_API_BASE || '';

// The AbortController is held outside React state because it is not
// serializable and should never trigger a re-render.
let activeController = null;

const initialState = {
  // True from the moment the user clicks Process until a result or error.
  // Survives page navigation because it lives here, not in the page component.
  processing: false,

  // Latest stage label and percent reported by the backend job.
  stage: null,
  progress: 0,

  // Final analysis payload (the same shape the /process endpoint returns).
  result: null,

  // Error message if processing failed; null while everything is fine.
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
  // Keep the original available in the summary after clearing the upload.
  resultPreviewUrl: '',

  // Toggle for whether the result overlay should auto-open on the
  // ImageProcessing page once processing completes.
  overlayOpen: false,
};

export const useProcessingStore = create((set, get) => ({
  ...initialState,

  setOverlayOpen: (open) => set({ overlayOpen: Boolean(open) }),

  // Reset back to a clean slate when clearing or selecting a new image.
  reset: () => {
    if (activeController) {
      try {
        activeController.abort();
      } catch { /* noop */ }
    }
    activeController = null;
    set({ ...initialState });
  },

  // Stop watching progress without clearing already-saved state.
  // The laptop completes its current analysis even if this tab disconnects.
  // (kept for an eventual cancel button; currently unused).
  cancel: () => {
    if (activeController) {
      try {
        activeController.abort();
      } catch { /* noop */ }
    }
    activeController = null;
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
      } catch { /* noop */ }
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

    if (activeController !== controller) return;

    set({
      processing: true,
      stage: 'Uploading image...',
      progress: 1,
      result: null,
      error: '',
      saveError: '',
      fileName: file.name || 'drone_image.jpg',
      previewUrl,
      resultPreviewUrl: '',
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
      formData.append(
        'allow_partial_map_overlap',
        params.allow_partial_map_overlap ? 'true' : 'false',
      );
      if (params.species) {
        formData.append('species', String(params.species));
      }

      const started = await startProcessingJob({
        api: API,
        formData,
        signal: controller.signal,
        onRetry: () => {
          if (activeController === controller) set({ stage: 'Connection interrupted. Retrying upload...' });
        },
      });
      const finalPayload = await waitForProcessingJob({
        api: API,
        jobId: started.job_id,
        signal: controller.signal,
        onProgress: (job) => {
          if (activeController !== controller) return;
          const next = {};
          if (typeof job.stage === 'string') next.stage = job.stage;
          if (typeof job.pct === 'number') next.progress = job.pct;
          set(next);
        },
      });
      if (activeController !== controller) return;

      set({
        processing: false,
        stage: 'Complete!',
        progress: 100,
        result: finalPayload,
        fileName: '',
        previewUrl: '',
        resultPreviewUrl: previewUrl,
        overlayOpen: true,
      });
    } catch (processError) {
      if (activeController !== controller) return;
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
      if (activeController === controller) activeController = null;
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

      const payload = await readAnalysisResponse(response);
      const savedResult = {
        ...result,
        saved: true,
        analysis_id: payload.analysis_id,
        source_image_name: result.source_image_name || result.uploaded_file_name,
        uploaded_file_name: payload.analysis_name || 'Saved analysis',
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
