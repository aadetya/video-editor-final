(() => {
  "use strict";

  const section = document.getElementById("batch-processing-section");
  if (!section) return;

  const batchListEl = document.getElementById("batch-list");
  const templateEl = document.getElementById("batch-item-template");

  const addBtn = document.getElementById("batch-add-btn");
  const startBtn = document.getElementById("batch-start-btn");
  const stopBtn = document.getElementById("batch-stop-btn");
  const clearBtn = document.getElementById("batch-clear-btn");
  const batchAspectSelect = document.getElementById("batch-aspect-ratio");

  const editorBanner = document.getElementById("batch-editor-banner");
  const editorBannerName = document.getElementById("batch-editor-batch-name");
  const editorBannerMeta = document.getElementById("batch-editor-batch-meta");
  const saveCurrentBtn = document.getElementById("batch-save-current-btn");
  const exitEditorBtn = document.getElementById("batch-exit-editor-btn");

  if (
    !batchListEl ||
    !templateEl ||
    !addBtn ||
    !startBtn ||
    !stopBtn ||
    !clearBtn ||
    !batchAspectSelect ||
    !editorBanner ||
    !editorBannerName ||
    !editorBannerMeta ||
    !saveCurrentBtn ||
    !exitEditorBtn
  ) {
    return;
  }

  const STATUS = {
    NEEDS_FILES: { text: "Needs files", className: "batch-badge-needs" },
    NEEDS_HIGHLIGHTS: { text: "Needs highlights", className: "batch-badge-needs" },
    QUEUED: { text: "Queued", className: "batch-badge-queued" },
    READY: { text: "Ready", className: "batch-badge-ready" },
    EDITING: { text: "Editing", className: "batch-badge-uploading" },
    UPLOADING: { text: "Uploading", className: "batch-badge-uploading" },
    PROCESSING: { text: "Processing", className: "batch-badge-processing" },
    DONE: { text: "Done", className: "batch-badge-done" },
    ERROR: { text: "Error", className: "batch-badge-error" },
    SKIPPED: { text: "Skipped", className: "batch-badge-skipped" },
  };

  /** @type {Array<ReturnType<typeof createBatchFromTemplate>>} */
  const batches = [];
  /** @type {string[]} */
  const processingQueue = [];
  const queuedBatchIds = new Set();
  let currentProcessingBatchId = null;
  let queueWorkerRunning = false;
  let activeBatchId = null;

  function deepClone(value) {
    if (typeof structuredClone === "function") return structuredClone(value);
    return JSON.parse(JSON.stringify(value));
  }

  function safeFilename(name) {
    return (name || "file")
      .replace(/[/\\?%*:|"<>]/g, "_")
      .replace(/\s+/g, "_");
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) return "";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let size = bytes;
    let unitIndex = 0;
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex += 1;
    }
    return `${size.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
  }

  function setBadge(batch, statusKey) {
    const status = STATUS[statusKey] || STATUS.NEEDS_FILES;
    batch.badge.textContent = status.text;
    batch.badge.className = `batch-badge ${status.className}`;
  }

  function appendLog(batch, message, kind = "info") {
    const line = document.createElement("div");
    line.className = `batch-log-line batch-log-${kind}`;
    line.textContent = message;
    batch.log.appendChild(line);
  }

  function clearLog(batch) {
    batch.log.innerHTML = "";
  }

  function batchFilesReady(batch) {
    return Boolean(batch.videoInput.files?.[0] && batch.subtitleInput.files?.[0]);
  }

  function batchHasState(batch) {
    return Boolean(
      batch.savedState &&
        Array.isArray(batch.savedState.transcriptData) &&
        batch.savedState.transcriptData.length > 0 &&
        Array.isArray(batch.savedState.subtitles) &&
        batch.savedState.subtitles.length > 0
    );
  }

  function batchHasUploadedVideo(batch) {
    return Boolean(batch.savedState?.currentVideoPath);
  }

  function batchHasMapping(batch) {
    return Boolean(batch.mappingInput?.files?.[0]);
  }

  function batchStateHighlightCount(state) {
    const clips = Array.isArray(state?.highlights) ? state.highlights.length : 0;
    const music = Array.isArray(state?.musicHighlights) ? state.musicHighlights.length : 0;
    return clips + music;
  }

  function batchCanQueue(batch) {
    if (batchHasState(batch)) {
      return batchStateHighlightCount(batch.savedState) > 0;
    }
    if (!batchFilesReady(batch)) return false;
    return batchHasMapping(batch);
  }

  function batchCanProcess(batch) {
    return batchHasState(batch) || batchFilesReady(batch);
  }

  function editorGlobalsAvailable() {
    return (
      typeof displayTranscript === "function" &&
      typeof displayMusicTranscript === "function" &&
      typeof updateHighlightsList === "function" &&
      typeof updateMusicHighlightsList === "function" &&
      typeof updatePreviewHighlights === "function" &&
      typeof updateMusicHighlightsDisplay === "function"
    );
  }

  function showEditorLoading(message) {
    const text = String(message || "Preparing editor…");
    const placeholder = `<p style="color: #666; text-align: center; padding: 18px;">${text}</p>`;

    try {
      if (typeof transcriptPreviewSection !== "undefined" && transcriptPreviewSection) {
        transcriptPreviewSection.style.display = "block";
      }
      if (typeof selectionSection !== "undefined" && selectionSection) {
        selectionSection.style.display = "block";
      }
      if (typeof highlightsSection !== "undefined" && highlightsSection) {
        highlightsSection.style.display = "block";
      }
      if (typeof musicSelectionSection !== "undefined" && musicSelectionSection) {
        musicSelectionSection.style.display = "block";
      }
      if (typeof musicHighlightsSection !== "undefined" && musicHighlightsSection) {
        musicHighlightsSection.style.display = "block";
      }
      if (typeof processSection !== "undefined" && processSection) {
        processSection.style.display = "block";
      }

      if (typeof transcriptPreview !== "undefined" && transcriptPreview) {
        transcriptPreview.innerHTML = placeholder;
      }
      if (typeof transcriptDisplay !== "undefined" && transcriptDisplay) {
        transcriptDisplay.innerHTML = placeholder;
      }
      if (typeof musicTranscriptDisplay !== "undefined" && musicTranscriptDisplay) {
        musicTranscriptDisplay.innerHTML = placeholder;
      }
      if (typeof highlightsList !== "undefined" && highlightsList && !highlightsList.innerHTML.trim()) {
        highlightsList.innerHTML = placeholder;
      }
      if (
        typeof musicHighlightsList !== "undefined" &&
        musicHighlightsList &&
        !musicHighlightsList.innerHTML.trim()
      ) {
        musicHighlightsList.innerHTML = placeholder;
      }

      if (typeof uploadProgress !== "undefined" && uploadProgress) {
        const textEl = uploadProgress.querySelector(".progress-text");
        if (textEl) {
          textEl.textContent = "Preparing editor… (parsing subtitle text)";
        }
        uploadProgress.style.display = "block";
      }
    } catch {}
  }

  function hideEditorLoading() {
    try {
      if (typeof uploadProgress !== "undefined" && uploadProgress) {
        const textEl = uploadProgress.querySelector(".progress-text");
        if (textEl) textEl.textContent = "Uploading & processing…";
        uploadProgress.style.display = "none";
      }
    } catch {}
  }

  function getEditorAspectRatio() {
    try {
      if (typeof aspectRatioSelect !== "undefined" && aspectRatioSelect) {
        return aspectRatioSelect.value || "4:5";
      }
    } catch {}
    return batchAspectSelect.value || "4:5";
  }

  function setEditorAspectRatio(value) {
    try {
      if (typeof aspectRatioSelect !== "undefined" && aspectRatioSelect) {
        aspectRatioSelect.value = value;
      }
    } catch {}
  }

  function getVideoFilenameText() {
    try {
      if (typeof videoFilename !== "undefined" && videoFilename) {
        return videoFilename.textContent || "";
      }
    } catch {}
    return "";
  }

  function getTranscriptFilenameText() {
    try {
      if (typeof transcriptFilename !== "undefined" && transcriptFilename) {
        return transcriptFilename.textContent || "";
      }
    } catch {}
    return "";
  }

  function setVideoFilenameText(value) {
    try {
      if (typeof videoFilename !== "undefined" && videoFilename) {
        videoFilename.textContent = value;
      }
    } catch {}
  }

  function setTranscriptFilenameText(value) {
    try {
      if (typeof transcriptFilename !== "undefined" && transcriptFilename) {
        transcriptFilename.textContent = value;
      }
    } catch {}
  }

  function captureEditorState() {
    const state = {
      currentVideoPath: null,
      transcriptData: [],
      subtitles: [],
      highlights: [],
      musicHighlights: [],
      aspectRatio: getEditorAspectRatio(),
      videoFilenameText: "",
      transcriptFilenameText: "",
    };

    try {
      if (typeof currentVideoPath !== "undefined") state.currentVideoPath = currentVideoPath;
      if (typeof transcriptData !== "undefined") state.transcriptData = deepClone(transcriptData || []);
      if (typeof subtitles !== "undefined") state.subtitles = deepClone(subtitles || []);
      if (typeof highlights !== "undefined") state.highlights = deepClone(highlights || []);
      if (typeof musicHighlights !== "undefined") state.musicHighlights = deepClone(musicHighlights || []);
    } catch {}

    state.videoFilenameText = getVideoFilenameText();
    state.transcriptFilenameText = getTranscriptFilenameText();
    return state;
  }

  function applyEditorState(state) {
    if (!editorGlobalsAvailable()) {
      alert("Editor UI is not available on this page.");
      return;
    }

    try {
      if (typeof currentVideoPath !== "undefined") currentVideoPath = state.currentVideoPath;
      if (typeof transcriptData !== "undefined") transcriptData = state.transcriptData || [];
      if (typeof subtitles !== "undefined") subtitles = state.subtitles || [];
      if (typeof highlights !== "undefined") highlights = state.highlights || [];
      if (typeof musicHighlights !== "undefined") musicHighlights = state.musicHighlights || [];
      if (typeof selectedRange !== "undefined") selectedRange = null;
      if (typeof selectedMusicRange !== "undefined") selectedMusicRange = null;
    } catch (e) {
      console.warn("Failed to apply editor globals:", e);
      return;
    }

    if (state.aspectRatio) setEditorAspectRatio(state.aspectRatio);
    if (state.videoFilenameText) setVideoFilenameText(state.videoFilenameText);
    if (state.transcriptFilenameText) setTranscriptFilenameText(state.transcriptFilenameText);

    if ((state.transcriptData || []).length > 0 && (state.subtitles || []).length > 0) {
      displayTranscript(state.subtitles, state.transcriptData);
      displayMusicTranscript(state.transcriptData);
      updateHighlightsList();
      updateMusicHighlightsList();
      updatePreviewHighlights();
      updateMusicHighlightsDisplay();

      if (typeof transcriptPreviewSection !== "undefined") transcriptPreviewSection.style.display = "block";
      if (typeof selectionSection !== "undefined") selectionSection.style.display = "block";
      if (typeof highlightsSection !== "undefined") highlightsSection.style.display = "block";
      if (typeof musicSelectionSection !== "undefined") musicSelectionSection.style.display = "block";
      if (typeof musicHighlightsSection !== "undefined") musicHighlightsSection.style.display = "block";
      if (typeof processSection !== "undefined") processSection.style.display = "block";
    }

    if (typeof resultSection !== "undefined") resultSection.style.display = "none";
    if (typeof videoPreview !== "undefined" && videoPreview) {
      try {
        videoPreview.pause();
      } catch {}
      videoPreview.removeAttribute("src");
      videoPreview.load();
    }
    if (typeof videoPreviewContainer !== "undefined") {
      videoPreviewContainer.style.display = "none";
    }
  }

  function showEditorBanner(batch) {
    editorBannerName.textContent = batch.name.textContent || "Batch";
    const clipCount = (batch.savedState?.highlights || []).length;
    const musicCount = (batch.savedState?.musicHighlights || []).length;

    const meta = [];
    const vf = batch.videoInput.files?.[0];
    const tf = batch.subtitleInput.files?.[0];
    if (vf?.name) meta.push(`Video: ${vf.name}`);
    if (tf?.name) meta.push(`Subtitle: ${tf.name}`);
    meta.push(`Clips: ${clipCount}`);
    meta.push(`Music: ${musicCount}`);
    editorBannerMeta.textContent = meta.join(" • ");

    editorBanner.style.display = "flex";
    updateControls();
  }

  function hideEditorBanner() {
    editorBanner.style.display = "none";
    editorBannerName.textContent = "";
    editorBannerMeta.textContent = "";
    saveCurrentBtn.disabled = true;
  }

  function updateBatchNameAndIndex() {
    batches.forEach((batch, idx) => {
      batch.name.textContent = `Batch #${idx + 1}`;
    });
  }

  function updateBatchBadge(batch) {
    if (batch.id === currentProcessingBatchId) {
      setBadge(batch, "PROCESSING");
      return;
    }
    if (batch.isUploading) {
      setBadge(batch, "UPLOADING");
      return;
    }
    if (activeBatchId && batch.id === activeBatchId) {
      setBadge(batch, "EDITING");
      return;
    }
    if (queuedBatchIds.has(batch.id)) {
      setBadge(batch, "QUEUED");
      const queueIndex = processingQueue.indexOf(batch.id);
      if (queueIndex >= 0) {
        batch.badge.textContent = `Queued #${queueIndex + 1}`;
      }
      return;
    }
    if (batch.statusKey) {
      setBadge(batch, batch.statusKey);
      return;
    }
    if (batchHasState(batch)) {
      const highlightCount = batchStateHighlightCount(batch.savedState);
      setBadge(batch, highlightCount > 0 ? "READY" : "NEEDS_HIGHLIGHTS");
      return;
    }
    if (batchFilesReady(batch)) {
      setBadge(batch, batchHasMapping(batch) ? "READY" : "NEEDS_HIGHLIGHTS");
      return;
    }
    setBadge(batch, "NEEDS_FILES");
  }

  function updateAllBadges() {
    batches.forEach(updateBatchBadge);
  }

  function updateControls() {
    addBtn.disabled = false;
    batchAspectSelect.disabled = false;

    const anyEnqueueable = batches.some(
      (b) =>
        batchCanQueue(b) &&
        !queuedBatchIds.has(b.id) &&
        b.id !== currentProcessingBatchId &&
        !b.isUploading
    );

    startBtn.disabled = !anyEnqueueable && processingQueue.length === 0;
    stopBtn.disabled = currentProcessingBatchId === null && processingQueue.length === 0;
    clearBtn.disabled =
      currentProcessingBatchId !== null || processingQueue.length > 0 || batches.length === 0;

    batches.forEach((batch) => {
      const isBusy = batch.id === currentProcessingBatchId || batch.isUploading;
      const isQueued = queuedBatchIds.has(batch.id);
      const queueIndex = isQueued ? processingQueue.indexOf(batch.id) : -1;
      const queuePosition = queueIndex >= 0 ? queueIndex + 1 : null;

      batch.root.classList.toggle(
        "batch-item-processing",
        batch.id === currentProcessingBatchId
      );
      batch.root.classList.toggle("batch-item-queued", isQueued);

      batch.removeBtn.disabled = isBusy;
      batch.videoInput.disabled = isBusy;
      batch.subtitleInput.disabled = isBusy;
      batch.mappingInput.disabled = isBusy;
      batch.editBtn.disabled = isBusy || (!batchFilesReady(batch) && !batchHasState(batch));
      batch.processBtn.disabled = isBusy || isQueued || !batchCanProcess(batch);

      if (isQueued) {
        batch.processBtn.textContent = queuePosition ? `Queued #${queuePosition}` : "Queued";
      } else if (batch.id === currentProcessingBatchId) {
        batch.processBtn.textContent = "Processing";
      } else {
        batch.processBtn.textContent = "Process";
      }
    });

    const activeBatch = activeBatchId ? batches.find((b) => b.id === activeBatchId) : null;
    const activeBusy = Boolean(
      activeBatch && (activeBatch.id === currentProcessingBatchId || activeBatch.isUploading)
    );

    saveCurrentBtn.disabled = !activeBatchId || activeBusy;
    exitEditorBtn.disabled = activeBusy;

    // If the user is editing a batch, route processing through the queue and avoid parallel runs.
    try {
      if (typeof processBtn !== "undefined" && processBtn && activeBatchId) {
        const activeIsQueued = queuedBatchIds.has(activeBatchId);
        const activeIsProcessing = activeBatchId === currentProcessingBatchId;
        if (activeBusy || activeIsQueued || activeIsProcessing) {
          processBtn.disabled = true;
        }
      }
    } catch {}
  }

  function saveActiveBatchState() {
    if (!activeBatchId) return;
    if (!editorGlobalsAvailable()) return;
    const batch = batches.find((b) => b.id === activeBatchId);
    if (!batch) return;
    batch.savedState = captureEditorState();
    batch.userEdited = true;
    updateBatchBadge(batch);
    showEditorBanner(batch);
    updateControls();
  }

  function setActiveBatch(batch) {
    const prevActiveId = activeBatchId;
    activeBatchId = batch.id;
    if (prevActiveId && prevActiveId !== activeBatchId) {
      const prevBatch = batches.find((b) => b.id === prevActiveId);
      if (prevBatch) updateBatchBadge(prevBatch);
    }
    updateBatchBadge(batch);
    showEditorBanner(batch);
    updateControls();
  }

  function exitEditorMode() {
    saveActiveBatchState();
    activeBatchId = null;
    hideEditorBanner();
    batches.forEach(updateBatchBadge);
    updateControls();
    section.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function normalizeWordToken(token) {
    if (!token) return "";
    return token.toLowerCase().replace(/[^a-z0-9]+/gi, "");
  }

  function normalizeSegment(text) {
    if (!text) return "";
    return text
      .replace("\ufeff", "")
      .toLowerCase()
      .split(/\s+/)
      .map((token) => normalizeWordToken(token))
      .filter(Boolean)
      .join(" ");
  }

  function parseMappingText(rawText) {
    if (!rawText) return [];
    let text = rawText;

    if (text && text.charCodeAt(0) === 0xfeff) {
      text = text.slice(1);
    }

    text = text.replace(/→/g, "->");

    // JSON format
    try {
      const json = JSON.parse(text);
      if (!Array.isArray(json)) {
        throw new Error("Mapping JSON must be an array of objects.");
      }

      /** @type {Array<{segment: string, clip: string | number}>} */
      const entries = [];
      json.forEach((item) => {
        if (
          !item ||
          typeof item.segment !== "string" ||
          (typeof item.clip !== "number" && typeof item.clip !== "string")
        ) {
          throw new Error(
            "Each mapping entry must have a 'segment' (string) and 'clip' (number or string)."
          );
        }
        entries.push({ segment: item.segment, clip: item.clip });
      });
      return entries;
    } catch {
      // fallthrough to TXT parsing
    }

    // TXT format: "Some phrase" -> 4
    const lines = text
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.length > 0);

    /** @type {Array<{segment: string, clip: number}>} */
    const parsed = [];

    for (const line of lines) {
      const parts = line.split("->");
      if (parts.length !== 2) continue;

      let segment = parts[0].trim();
      let clipPart = parts[1].trim();

      if (
        (segment.startsWith('"') && segment.endsWith('"')) ||
        (segment.startsWith("'") && segment.endsWith("'"))
      ) {
        segment = segment.slice(1, -1);
      }

      clipPart = clipPart.replace(/[^\d]+$/g, "").trim();
      const clipNumber = parseInt(clipPart, 10);
      if (!segment || Number.isNaN(clipNumber)) continue;

      parsed.push({ segment, clip: clipNumber });
    }

    return parsed;
  }

  async function parseMappingFile(file) {
    if (!file) return [];
    const text = await file.text();
    return parseMappingText(text);
  }

  function buildDraftTranscriptFromText(rawText) {
    if (!rawText) return { transcript: [], subtitles: [], fullText: "", wordCount: 0 };

    let text = rawText;
    if (text && text.charCodeAt(0) === 0xfeff) {
      text = text.slice(1);
    }

    const lines = text
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0);

    /** @type {Array<{word: string, start_time: number, end_time: number}>} */
    const transcript = [];
    /** @type {Array<{text: string, start_word: number, end_word: number, word_count: number}>} */
    const subtitles = [];
    let wordCursor = 0;
    const defaultWordDuration = 0.5;

    for (const line of lines) {
      const tokens = line.split(/\s+/).filter(Boolean);
      if (!tokens.length) continue;

      const startWord = wordCursor;
      tokens.forEach((tok, idx) => {
        const t0 = (wordCursor + idx) * defaultWordDuration;
        const t1 = t0 + defaultWordDuration;
        transcript.push({ word: tok, start_time: t0, end_time: t1 });
      });
      wordCursor += tokens.length;
      const endWord = wordCursor - 1;

      subtitles.push({
        text: line,
        start_word: startWord,
        end_word: endWord,
        word_count: tokens.length,
      });
    }

    const fullText = transcript.map((entry) => entry.word).join(" ").trim();
    return { transcript, subtitles, fullText, wordCount: transcript.length };
  }

  async function buildDraftTranscriptFromFile(file) {
    if (!file) return { transcript: [], subtitles: [], fullText: "", wordCount: 0 };
    const text = await file.text();
    return buildDraftTranscriptFromText(text);
  }

  function clipPathFromSpec(spec) {
    if (typeof spec === "number") return `clips/${spec}.mp4`;
    if (typeof spec === "string") {
      const trimmed = spec.trim();
      if (/^\d+$/.test(trimmed)) return `clips/${trimmed}.mp4`;
      if (trimmed.startsWith("clips/") || trimmed.endsWith(".mp4")) return trimmed;
      return `clips/${trimmed}`;
    }
    return null;
  }

  function buildHighlightsFromMapping(mappingEntries, transcriptDataLocal) {
    if (!mappingEntries?.length || !transcriptDataLocal?.length) {
      return { highlights: [], matched: 0, unmatched: 0 };
    }

    const transcriptTokens = transcriptDataLocal.map((w) => normalizeWordToken(w.word));

    /** @type {Array<any>} */
    const highlightsOut = [];
    let matched = 0;
    let unmatched = 0;

    for (const entry of mappingEntries) {
      const rawSegment = entry.segment || "";
      const normalized = normalizeSegment(rawSegment);
      if (!normalized) continue;

      const segmentTokens = normalized.split(" ").filter(Boolean);
      const segLen = segmentTokens.length;
      if (!segLen) continue;

      let startIndex = -1;
      for (let i = 0; i <= transcriptTokens.length - segLen; i++) {
        let ok = true;
        for (let j = 0; j < segLen; j++) {
          if (transcriptTokens[i + j] !== segmentTokens[j]) {
            ok = false;
            break;
          }
        }
        if (ok) {
          startIndex = i;
          break;
        }
      }

      if (startIndex < 0) {
        unmatched += 1;
        continue;
      }

      const endIndex = startIndex + segLen - 1;
      const phrase = transcriptDataLocal
        .slice(startIndex, endIndex + 1)
        .map((w) => w.word)
        .join(" ");

      const clipPath = clipPathFromSpec(entry.clip);

      highlightsOut.push({
        phrase,
        start_word: startIndex,
        end_word: endIndex,
        clip_path: clipPath,
        music_path: null,
        music_volume: 1.0,
        occurrence: 1,
      });
      matched += 1;
    }

    return { highlights: highlightsOut, matched, unmatched };
  }

  async function buildBatchStateFromLocalFiles(batch) {
    const videoFile = batch.videoInput.files?.[0] || null;
    const subtitleFile = batch.subtitleInput.files?.[0] || null;
    if (!videoFile || !subtitleFile) {
      throw new Error("Missing required files.");
    }

    const draft = await buildDraftTranscriptFromFile(subtitleFile);
    if (!draft.transcript.length || !draft.subtitles.length) {
      throw new Error("Subtitle file produced no transcript words.");
    }

    const existingState = batch.savedState || null;

    const mappingFile = batch.mappingInput.files?.[0] || null;
    let mappingEntries = [];
    if (mappingFile) {
      try {
        mappingEntries = await parseMappingFile(mappingFile);
      } catch (e) {
        appendLog(
          batch,
          `Mapping parse failed (continuing without auto highlights): ${e?.message || e}`,
          "warn"
        );
        mappingEntries = [];
      }
    }

    const existingHighlights = Array.isArray(existingState?.highlights)
      ? existingState.highlights
      : [];
    const existingMusicHighlights = Array.isArray(existingState?.musicHighlights)
      ? existingState.musicHighlights
      : [];

    const keepExistingHighlights = existingHighlights.length + existingMusicHighlights.length > 0;
    const { highlights: autoHighlights, matched, unmatched } = buildHighlightsFromMapping(
      mappingEntries,
      draft.transcript
    );

    if (!keepExistingHighlights && mappingEntries.length) {
      appendLog(
        batch,
        `Auto highlights: ${autoHighlights.length} (matched ${matched}, unmatched ${unmatched})`
      );
    }

    const nextState = {
      currentVideoPath: existingState?.currentVideoPath || null,
      transcriptData: deepClone(draft.transcript),
      subtitles: deepClone(draft.subtitles),
      highlights: deepClone(keepExistingHighlights ? existingHighlights : autoHighlights),
      musicHighlights: deepClone(keepExistingHighlights ? existingMusicHighlights : []),
      aspectRatio: existingState?.aspectRatio || null,
      videoFilenameText: `Selected: ${videoFile.name}`,
      transcriptFilenameText: `Selected: ${subtitleFile.name}`,
    };

    batch.savedState = nextState;
    return nextState;
  }

  async function uploadBatchToState(batch) {
    const videoFile = batch.videoInput.files?.[0] || null;
    const subtitleFile = batch.subtitleInput.files?.[0] || null;
    if (!videoFile || !subtitleFile) {
      throw new Error("Missing required files.");
    }

    batch.isUploading = true;
    batch.statusKey = null;
    updateBatchBadge(batch);
    updateControls();

    try {
      clearLog(batch);
      appendLog(batch, "Uploading video + subtitle text…");

      const uniqueVideoName = safeFilename(
        `batch_${Date.now()}_${Math.random().toString(36).slice(2, 8)}_${videoFile.name}`
      );

      const formData = new FormData();
      formData.append("video", videoFile, uniqueVideoName);
      formData.append("transcript_file", subtitleFile);

      const uploadResp = await fetch("/upload-video-with-txt", {
        method: "POST",
        body: formData,
      });
      const uploadData = await uploadResp.json();

      if (!uploadResp.ok || uploadData?.error) {
        throw new Error(uploadData?.error || uploadResp.statusText || "Upload failed");
      }
      if (!uploadData.video_path || !uploadData.transcript || !uploadData.subtitles) {
        throw new Error("Upload succeeded but response is missing transcript/subtitles.");
      }

      appendLog(
        batch,
        `Uploaded. Words: ${uploadData.word_count || uploadData.transcript.length}`
      );

      const mappingFile = batch.mappingInput.files?.[0] || null;
      let mappingEntries = [];
      if (mappingFile) {
        appendLog(batch, "Parsing mapping file…");
        try {
          mappingEntries = await parseMappingFile(mappingFile);
          appendLog(batch, `Mapping entries: ${mappingEntries.length}`);
        } catch (e) {
          appendLog(
            batch,
            `Mapping parse failed (continuing without auto highlights): ${e?.message || e}`,
            "warn"
          );
          mappingEntries = [];
        }
      }

      const { highlights: autoHighlights, matched, unmatched } = buildHighlightsFromMapping(
        mappingEntries,
        uploadData.transcript
      );
      if (mappingEntries.length) {
        appendLog(
          batch,
          `Auto highlights: ${autoHighlights.length} (matched ${matched}, unmatched ${unmatched})`
        );
      }

      const existingState = batch.savedState || null;
      const existingHighlights = Array.isArray(existingState?.highlights)
        ? existingState.highlights
        : [];
      const existingMusicHighlights = Array.isArray(existingState?.musicHighlights)
        ? existingState.musicHighlights
        : [];

      const keepExistingHighlights =
        existingHighlights.length + existingMusicHighlights.length > 0;

      batch.savedState = {
        currentVideoPath: uploadData.video_path,
        transcriptData: deepClone(uploadData.transcript),
        subtitles: deepClone(uploadData.subtitles),
        highlights: deepClone(keepExistingHighlights ? existingHighlights : autoHighlights),
        musicHighlights: deepClone(keepExistingHighlights ? existingMusicHighlights : []),
        aspectRatio: existingState?.aspectRatio || null,
        videoFilenameText: `Selected: ${videoFile.name}`,
        transcriptFilenameText: `Selected: ${subtitleFile.name}`,
      };
      if (!keepExistingHighlights) {
        batch.userEdited = false;
      }

      return batch.savedState;
    } finally {
      batch.isUploading = false;
      updateBatchBadge(batch);
      updateControls();
    }
  }

  async function ensureBatchState(batch) {
    if (batchHasUploadedVideo(batch)) return batch.savedState;
    if (!batchFilesReady(batch)) throw new Error("Batch is missing required files.");
    // Ensure we have a local transcript/subtitle state before upload, so manual edits persist.
    if (!batchHasState(batch)) {
      await buildBatchStateFromLocalFiles(batch);
    }
    return uploadBatchToState(batch);
  }

  async function ensureBatchEditorState(batch) {
    if (batchHasState(batch)) return batch.savedState;
    if (!batchFilesReady(batch)) throw new Error("Batch is missing required files.");
    return buildBatchStateFromLocalFiles(batch);
  }

  function enqueueBatch(batch, source = "manual") {
    if (!batchCanProcess(batch)) {
      appendLog(batch, "Cannot queue: missing required files/state.", "warn");
      updateControls();
      return false;
    }
    if (!batchCanQueue(batch)) {
      appendLog(
        batch,
        "Cannot queue: add a mapping file, or click Edit and add at least one clip/music highlight.",
        "warn"
      );
      updateControls();
      return false;
    }
    if (batch.id === currentProcessingBatchId) return false;
    if (queuedBatchIds.has(batch.id)) return false;

    processingQueue.push(batch.id);
    queuedBatchIds.add(batch.id);
    batch.statusKey = null;

    appendLog(batch, source === "editor" ? "Queued from editor." : "Queued for processing.");
    updateBatchBadge(batch);
    updateControls();

    void pumpQueue();
    return true;
  }

  function removeBatchFromQueue(batchId) {
    if (!queuedBatchIds.has(batchId)) return false;
    queuedBatchIds.delete(batchId);
    const idx = processingQueue.indexOf(batchId);
    if (idx >= 0) processingQueue.splice(idx, 1);
    return true;
  }

  function clearQueue() {
    if (processingQueue.length === 0) return;
    processingQueue.length = 0;
    queuedBatchIds.clear();
  }

  async function pumpQueue() {
    if (queueWorkerRunning) return;
    queueWorkerRunning = true;

    try {
      while (processingQueue.length > 0) {
        const nextId = processingQueue.shift();
        queuedBatchIds.delete(nextId);

        const batch = batches.find((b) => b.id === nextId);
        if (!batch) continue;

        currentProcessingBatchId = nextId;
        batch.statusKey = null;
        updateAllBadges();
        updateControls();

        await processBatchNow(batch);

        currentProcessingBatchId = null;
        updateAllBadges();
        updateControls();
      }
    } finally {
      currentProcessingBatchId = null;
      queueWorkerRunning = false;
      updateAllBadges();
      updateControls();
    }
  }

  async function processBatchNow(batch) {
    clearLog(batch);
    batch.result.style.display = "none";
    batch.s3Btn.style.display = "none";
    batch.statusKey = null;

    try {
      if (activeBatchId === batch.id) {
        saveActiveBatchState();
      }

      const state = await ensureBatchState(batch);
      const allHighlights = [
        ...(state.highlights || []),
        ...(state.musicHighlights || []),
      ];

      if (allHighlights.length === 0) {
        throw new Error("Please add at least one highlight (clip or music) before processing.");
      }

      appendLog(batch, "Processing video… (this can take a while)");

      const aspectRatio = batch.userEdited
        ? state.aspectRatio || batchAspectSelect.value || "4:5"
        : batchAspectSelect.value || state.aspectRatio || "4:5";

      const processResp = await fetch("/process-video", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_path: state.currentVideoPath,
          highlights: allHighlights,
          transcript: state.transcriptData,
          preserve_audio: true,
          subtitle_sentences: state.subtitles,
          aspect_ratio: aspectRatio,
        }),
      });

      const processData = await processResp.json();
      if (!processResp.ok || processData?.error) {
        throw new Error(processData?.error || processResp.statusText || "Processing failed");
      }
      if (!processData.output_filename) {
        throw new Error("Processing succeeded but response is missing output filename.");
      }

      batch.statusKey = "DONE";
      appendLog(batch, processData.message || "Done.");

      const ts = Date.now();
      const outputFilename = processData.output_filename;
      const previewUrl = `/video/${encodeURIComponent(outputFilename)}?t=${ts}`;
      const downloadUrl = `/download/${encodeURIComponent(outputFilename)}`;

      try {
        batch.preview.pause();
      } catch {}
      batch.preview.removeAttribute("src");
      batch.preview.load();
      batch.preview.src = previewUrl;
      batch.preview.load();

      batch.downloadBtn.href = downloadUrl;

      if (processData.s3_video_url) {
        batch.s3Btn.href = processData.s3_video_url;
        batch.s3Btn.style.display = "inline-block";
      }

      batch.result.style.display = "block";
    } catch (e) {
      batch.statusKey = "ERROR";
      appendLog(batch, e?.message || String(e), "error");
    } finally {
      updateBatchBadge(batch);
      updateControls();
    }
  }

  function createBatchFromTemplate() {
    const fragment = templateEl.content.cloneNode(true);
    const root = fragment.querySelector(".batch-item");
    if (!root) throw new Error("Batch template is missing .batch-item root");

    const name = root.querySelector(".batch-item-name");
    const badge = root.querySelector(".batch-badge");
    const removeBtn = root.querySelector(".batch-remove-btn");
    const editBtn = root.querySelector(".batch-edit-btn");
    const processBtn = root.querySelector(".batch-process-btn");

    const videoInput = root.querySelector(".batch-video-input");
    const subtitleInput = root.querySelector(".batch-subtitle-input");
    const mappingInput = root.querySelector(".batch-mapping-input");

    const videoName = root.querySelector(".batch-video-name");
    const subtitleName = root.querySelector(".batch-subtitle-name");
    const mappingName = root.querySelector(".batch-mapping-name");

    const log = root.querySelector(".batch-log");
    const result = root.querySelector(".batch-result");
    const preview = root.querySelector(".batch-preview");
    const downloadBtn = root.querySelector(".batch-download-btn");
    const s3Btn = root.querySelector(".batch-s3-btn");

    if (
      !name ||
      !badge ||
      !removeBtn ||
      !editBtn ||
      !processBtn ||
      !videoInput ||
      !subtitleInput ||
      !mappingInput ||
      !videoName ||
      !subtitleName ||
      !mappingName ||
      !log ||
      !result ||
      !preview ||
      !downloadBtn ||
      !s3Btn
    ) {
      throw new Error("Batch template is missing required elements");
    }

    const batchId = `batch_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    root.dataset.batchId = batchId;

    /** @type {HTMLVideoElement} */
    const previewVideo = preview;

    const batch = {
      id: batchId,
      root,
      name,
      badge,
      removeBtn,
      editBtn,
      processBtn,
      videoInput,
      subtitleInput,
      mappingInput,
      videoName,
      subtitleName,
      mappingName,
      log,
      result,
      preview: previewVideo,
      downloadBtn,
      s3Btn,
      savedState: null,
      userEdited: false,
      isUploading: false,
      statusKey: null,
    };

    updateBatchBadge(batch);

    videoInput.addEventListener("change", () => {
      const f = videoInput.files?.[0];
      videoName.textContent = f ? `${f.name}${f.size ? ` (${formatBytes(f.size)})` : ""}` : "";

      batch.statusKey = null;
      if (batch.savedState) {
        batch.savedState = null;
        batch.userEdited = false;
        batch.result.style.display = "none";
        batch.s3Btn.style.display = "none";
        appendLog(batch, "Video changed — cleared saved batch state.", "warn");
      }
      updateBatchBadge(batch);
      updateControls();
    });

    subtitleInput.addEventListener("change", () => {
      const f = subtitleInput.files?.[0];
      subtitleName.textContent = f ? `${f.name}${f.size ? ` (${formatBytes(f.size)})` : ""}` : "";

      batch.statusKey = null;
      if (batch.savedState) {
        batch.savedState = null;
        batch.userEdited = false;
        batch.result.style.display = "none";
        batch.s3Btn.style.display = "none";
        appendLog(batch, "Subtitle changed — cleared saved batch state.", "warn");
      }
      updateBatchBadge(batch);
      updateControls();
    });

    mappingInput.addEventListener("change", () => {
      const f = mappingInput.files?.[0];
      mappingName.textContent = f ? `${f.name}${f.size ? ` (${formatBytes(f.size)})` : ""}` : "";
      updateControls();
    });

    editBtn.addEventListener("click", async () => {
      if (!editorGlobalsAvailable()) {
        alert("Editor UI is not available on this page.");
        return;
      }

      saveActiveBatchState();
      setActiveBatch(batch);
      updateControls();

      try {
        if (!batchHasState(batch)) {
          showEditorLoading(
            "Preparing editor… Parsing subtitle text locally. (Video upload + audio transcription happens when you click Process.)"
          );
        }

        const targetAnchor =
          document.getElementById("transcript-preview-section") ||
          document.getElementById("selection-section") ||
          document.getElementById("upload-section");
        if (targetAnchor) {
          targetAnchor.scrollIntoView({ behavior: "smooth", block: "start" });
        }

        appendLog(batch, "Preparing editor… (parsing subtitles)");
        await ensureBatchEditorState(batch);
        applyEditorState(batch.savedState);
        showEditorBanner(batch);
        updateControls();
        const target =
          document.getElementById("transcript-preview-section") ||
          document.getElementById("selection-section");
        if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
      } catch (e) {
        setBadge(batch, "ERROR");
        appendLog(batch, `Edit failed: ${e?.message || e}`, "error");
      } finally {
        hideEditorLoading();
        updateControls();
      }
    });

    processBtn.addEventListener("click", () => {
      saveActiveBatchState();

      // Match the editor behaviour: don't upload/queue if highlights are missing.
      const highlightCount = batchStateHighlightCount(batch.savedState);
      const hasMapping = batchHasMapping(batch);
      if (!highlightCount && !hasMapping) {
        appendLog(
          batch,
          "Add a clip mapping file OR click Edit and add at least one highlight, then try again.",
          "warn"
        );
        alert(
          "Please add a mapping file or add at least one highlight (clip/music) before processing."
        );
        updateControls();
        return;
      }

      enqueueBatch(batch, "manual");
    });

    removeBtn.addEventListener("click", () => {
      if (activeBatchId === batch.id) {
        exitEditorMode();
      }
      removeBatchFromQueue(batch.id);
      const idx = batches.findIndex((b) => b.id === batch.id);
      if (idx >= 0) batches.splice(idx, 1);
      batch.root.remove();
      updateBatchNameAndIndex();
      updateAllBadges();
      updateControls();
    });

    return batch;
  }

  function addBatch() {
    const batch = createBatchFromTemplate();
    batches.push(batch);
    batchListEl.appendChild(batch.root);
    updateBatchNameAndIndex();
    updateControls();
  }

  function clearBatches() {
    if (currentProcessingBatchId !== null || processingQueue.length > 0) return;
    exitEditorMode();
    clearQueue();
    batches.length = 0;
    batchListEl.innerHTML = "";
    updateControls();
    addBatch();
  }

  async function startProcessingAll() {
    saveActiveBatchState();
    batches.forEach((batch) => {
      if (!batchCanQueue(batch)) {
        if (batchCanProcess(batch)) {
          appendLog(
            batch,
            "Skipped: add mapping file or highlights before processing.",
            "warn"
          );
        }
        return;
      }
      enqueueBatch(batch, "manual");
    });
    void pumpQueue();
  }

  addBtn.addEventListener("click", addBatch);
  startBtn.addEventListener("click", startProcessingAll);
  stopBtn.addEventListener("click", () => {
    clearQueue();
    updateAllBadges();
    updateControls();
  });
  clearBtn.addEventListener("click", clearBatches);
  saveCurrentBtn.addEventListener("click", saveActiveBatchState);
  exitEditorBtn.addEventListener("click", exitEditorMode);

  // When editing a batch, clicking the editor's "Process" button should enqueue that batch
  // instead of running processing in parallel.
  try {
    if (typeof processBtn !== "undefined" && processBtn) {
      processBtn.addEventListener(
        "click",
        (event) => {
          if (!activeBatchId) return;
          const batch = batches.find((b) => b.id === activeBatchId);
          if (!batch) return;

          event.preventDefault();
          event.stopImmediatePropagation();

          saveActiveBatchState();
          const highlightCount = batchStateHighlightCount(batch.savedState);
          if (!highlightCount) {
            appendLog(
              batch,
              "Cannot process: add at least one clip/music highlight first.",
              "warn"
            );
            alert("Please add at least one highlight (clip/music) before processing.");
            updateControls();
            return;
          }
          enqueueBatch(batch, "editor");
          section.scrollIntoView({ behavior: "smooth", block: "start" });
        },
        true
      );
    }
  } catch (e) {
    console.warn("Failed to hook editor Process button for queueing:", e);
  }

  window.addEventListener("batchVideoProcessed", (evt) => {
    const detail = evt?.detail;
    if (!detail || !activeBatchId) return;
    if (!detail.output_filename) return;

    const batch = batches.find((b) => b.id === activeBatchId);
    if (!batch) return;

    const ts = Date.now();
    const previewUrl = `/video/${encodeURIComponent(detail.output_filename)}?t=${ts}`;
    const downloadUrl = `/download/${encodeURIComponent(detail.output_filename)}`;

    batch.downloadBtn.href = downloadUrl;
    try {
      batch.preview.pause();
    } catch {}
    batch.preview.removeAttribute("src");
    batch.preview.load();
    batch.preview.src = previewUrl;
    batch.preview.load();
    batch.result.style.display = "block";

    if (detail.s3_video_url) {
      batch.s3Btn.href = detail.s3_video_url;
      batch.s3Btn.style.display = "inline-block";
    }

    batch.statusKey = "DONE";
    updateBatchBadge(batch);
    appendLog(batch, detail.message || "Done.");
    updateControls();
  });

  // Start with a single batch row by default
  addBatch();
})();
