// Global state
let currentVideoPath = null;
let transcriptData = [];
let subtitles = []; // Subtitle boxes from line breaks in TXT file (for organization only)
let highlights = [];
let selectedRange = null;
let isHighlightFileDialogOpen = false;
let highlightFileChosen = false;


// DOM Elements
const mainVideoInput = document.getElementById("main-video-input");
const videoFilename = document.getElementById("video-filename");
const transcriptFileInput = document.getElementById("transcript-file-input");
const transcriptFilename = document.getElementById("transcript-filename");
const uploadBtn = document.getElementById("upload-btn");
const uploadProgress = document.getElementById("upload-progress");
const transcriptPreviewSection = document.getElementById(
  "transcript-preview-section"
);
const transcriptPreview = document.getElementById("transcript-preview");
const selectionSection = document.getElementById("selection-section");
const transcriptDisplay = document.getElementById("transcript-display");
const selectionControls = document.getElementById("selection-controls");
const selectedTextSpan = document.getElementById("selected-text");
const clipInput = document.getElementById("clip-input");
const clipFilename = document.getElementById("clip-filename");
const uploadClipBtn = document.getElementById("upload-clip-btn");
const existingClipsSelect = document.getElementById("existing-clips");
const addHighlightBtn = document.getElementById("add-highlight-btn");
const cancelSelectionBtn = document.getElementById("cancel-selection-btn");
const highlightsSection = document.getElementById("highlights-section");
const highlightsList = document.getElementById("highlights-list");
const musicSelectionSection = document.getElementById(
  "music-selection-section"
);
const musicTranscriptDisplay = document.getElementById(
  "music-transcript-display"
);
const musicSelectionControls = document.getElementById(
  "music-selection-controls"
);
const musicSelectedText = document.getElementById("music-selected-text");
const musicInput = document.getElementById("music-input");
const musicFilename = document.getElementById("music-filename");
const uploadMusicBtn = document.getElementById("upload-music-btn");
const existingMusicSelect = document.getElementById("existing-music-select");
const musicVolume = document.getElementById("music-volume");
const musicVolumeDisplay = document.getElementById("music-volume-display");
const addMusicBtn = document.getElementById("add-music-btn");
const cancelMusicSelectionBtn = document.getElementById(
  "cancel-music-selection-btn"
);
const musicHighlightsSection = document.getElementById(
  "music-highlights-section"
);
const musicHighlightsList = document.getElementById("music-highlights-list");
const processSection = document.getElementById("process-section");
const processBtn = document.getElementById("process-btn");
const processProgress = document.getElementById("process-progress");
const aspectRatioSelect = document.getElementById("aspect-ratio-select");
const resultSection = document.getElementById("result-section");
const resultMessage = document.getElementById("result-message");
const downloadBtn = document.getElementById("download-btn");
const goBackBtn = document.getElementById("go-back-btn");
const videoPreview = document.getElementById("video-preview");
const videoPreviewContainer = document.getElementById("video-preview-container");
const loadProjectBtn = document.getElementById("load-project-btn");
const projectListContainer = document.getElementById("project-list-container");
const projectList = document.getElementById("project-list");
const saveProjectBtn = document.getElementById("save-project-btn");
const projectNameInput = document.getElementById("project-name-input");
const saveProjectStatus = document.getElementById("save-project-status");

// Modal elements
const videoUploadModal = document.getElementById("video-upload-modal");
const modalSelectedText = document.getElementById("modal-selected-text");
const modalVideoInput = document.getElementById("modal-video-input");
const modalVideoFilename = document.getElementById("modal-video-filename");
const modalUploadBtn = document.getElementById("modal-upload-btn");
const modalCancelBtn = document.getElementById("modal-cancel-btn");
const closeModalBtn = document.getElementById("close-modal-btn");

// Music state
let musicHighlights = [];
let selectedMusicRange = null;

// Mapping state + DOM
let mappingData = null;
const mappingFileInput = document.getElementById("mapping-file-input");
const mappingFilename = document.getElementById("mapping-filename");

// Event Listeners
mainVideoInput.addEventListener("change", handleVideoSelection);
transcriptFileInput.addEventListener("change", handleTranscriptSelection);
uploadBtn.addEventListener("click", uploadVideo);
clipInput.addEventListener("change", handleClipSelection);
uploadClipBtn.addEventListener("click", uploadClip);
addHighlightBtn.addEventListener("click", addHighlight);
cancelSelectionBtn.addEventListener("click", cancelSelection);
musicInput.addEventListener("change", handleMusicSelection);
uploadMusicBtn.addEventListener("click", uploadMusicFile);
musicVolume.addEventListener("input", (e) => {
  musicVolumeDisplay.textContent = e.target.value;
});
addMusicBtn.addEventListener("click", addMusicHighlight);
cancelMusicSelectionBtn.addEventListener("click", cancelMusicSelection);
processBtn.addEventListener("click", processVideo);
goBackBtn.addEventListener("click", goBackAndEdit);
loadProjectBtn.addEventListener("click", loadProjectList);
saveProjectBtn.addEventListener("click", saveProjectToS3);

// Mapping file listener (guarded so we don't crash if HTML doesn't have it)
if (mappingFileInput) {
  mappingFileInput.addEventListener("change", handleMappingSelection);
}

// Note: Modal functionality removed - file picker opens directly on text selection

// Load existing clips and music on page load
loadExistingClips();
loadExistingMusic();

// Try to restore state on page load if available
window.addEventListener("load", () => {
  const savedState = sessionStorage.getItem("videoEditorState");
  if (savedState) {
    // Don't auto-restore, but keep state available for manual restore
    console.log("Saved state available. Use 'Go Back' button to restore.");
  }
});

function handleVideoSelection(e) {
  const file = e.target.files[0];
  if (file) {
    videoFilename.textContent = `Selected: ${file.name}`;
    checkUploadReady();
  }
}

function handleTranscriptSelection(e) {
  const file = e.target.files[0];
  if (file) {
    transcriptFilename.textContent = `Selected: ${file.name}`;
    checkUploadReady();
  }
}

function handleClipSelection(e) {
  const file = e.target.files[0];
  if (file) {
    clipFilename.textContent = file.name;
  } else {
    clipFilename.textContent = "";
  }
}

function handleMusicSelection(e) {
  const file = e.target.files[0];
  if (file) {
    musicFilename.textContent = file.name;
  } else {
    musicFilename.textContent = "";
  }
}

function checkUploadReady() {
  const videoInputEl = document.getElementById("main-video-input");
  const transcriptInputEl = document.getElementById("transcript-file-input");

  const hasVideo =
    videoInputEl && videoInputEl.files && videoInputEl.files.length > 0;
  const hasTranscript =
    transcriptInputEl &&
    transcriptInputEl.files &&
    transcriptInputEl.files.length > 0;

  uploadBtn.disabled = !(hasVideo && hasTranscript);
}

// ================
// Mapping handlers
// ================

function handleMappingSelection(e) {
  const file = e.target.files[0];
  mappingFilename.textContent = file ? file.name : "";

  if (!file) {
    mappingData = null;
    return;
  }

  const reader = new FileReader();
  reader.onload = (evt) => {
    let text = evt.target.result;

    // Strip BOM if present
    if (text && text.charCodeAt(0) === 0xfeff) {
      text = text.slice(1);
    }

    // Normalize Unicode arrow to ASCII "->"
    text = text.replace(/→/g, "->");

    try {
      // Try JSON first
      const json = JSON.parse(text);
      if (!Array.isArray(json)) {
        throw new Error("Mapping JSON must be an array of objects.");
      }
      json.forEach((item) => {
        if (
          typeof item.segment !== "string" ||
          (typeof item.clip !== "number" && typeof item.clip !== "string")
        ) {
          throw new Error(
            "Each mapping entry must have a 'segment' (string) and 'clip' (number or string)."
          );
        }
      });
      mappingData = json;
      console.log("Loaded mapping as JSON:", mappingData);
    } catch (jsonErr) {
      // Fallback: TXT mapping like `"Some sentence" -> 4`
      try {
        const lines = text
          .split("\n")
          .map((l) => l.trim())
          .filter((l) => l.length > 0);

        const parsed = [];
        for (const line of lines) {
          const parts = line.split("->");
          if (parts.length !== 2) continue;

          let segment = parts[0].trim();
          let clipPart = parts[1].trim();

          // Remove surrounding quotes if present
          if (
            (segment.startsWith('"') && segment.endsWith('"')) ||
            (segment.startsWith("'") && segment.endsWith("'"))
          ) {
            segment = segment.slice(1, -1);
          }

          // Strip trailing non-digits from clip part
          clipPart = clipPart.replace(/[^\d]+$/g, "").trim();
          const clipNumber = parseInt(clipPart, 10);
          if (!segment || Number.isNaN(clipNumber)) continue;

          parsed.push({ segment, clip: clipNumber });
        }

        if (!parsed.length) {
          throw new Error("No valid mapping lines found in TXT file.");
        }

        mappingData = parsed;
        console.log("Loaded mapping as TXT mapping:", mappingData);
      } catch (txtErr) {
        console.error("Failed to parse mapping:", { jsonErr, txtErr });
        mappingData = null;
        alert(
          "Mapping file could not be parsed as JSON or TXT mapping. Please check the format."
        );
      }
    }
  };

  reader.readAsText(file);
}

// Helper: normalize a full segment from the mapping file
// Uses the same token logic as normalizeWordToken so mapping and transcript align.
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


// Helper: normalize a single word token from transcriptData
function normalizeWordToken(token) {
  if (!token) return "";
  return token
    .toLowerCase()
    // strip non-alphanumerics
    .replace(/[^a-z0-9]+/gi, "");
}

// Use mapping to auto-create highlight segments (word-level search across transcript)
function applyAutoHighlightsFromMapping() {
  if (!mappingData || !mappingData.length) return;
  if (!transcriptData || !transcriptData.length) return;

  // Normalize entire transcript into tokens once
  const transcriptTokens = transcriptData.map((w) =>
    normalizeWordToken(w.word)
  );

  let addedCount = 0;

  mappingData.forEach((entry) => {
    const rawSegment = entry.segment || "";
    const clipSpec = entry.clip;

    // Normalize mapping segment
    const normalized = normalizeSegment(rawSegment);
    if (!normalized) return;

    const segmentTokens = normalized.split(" ").filter(Boolean);
    const segLen = segmentTokens.length;
    if (!segLen) return;

    let startIndex = -1;

    // Sliding-window match over the WHOLE transcript (not per-subtitle)
    outer: for (let i = 0; i <= transcriptTokens.length - segLen; i++) {
      for (let j = 0; j < segLen; j++) {
        if (transcriptTokens[i + j] !== segmentTokens[j]) {
          continue outer;
        }
      }
      startIndex = i;
      break;
    }

    if (startIndex === -1) {
      console.warn(
        "Mapping segment not found in transcript (word-level):",
        rawSegment,
        "(normalized:",
        normalized,
        ")"
      );
      return;
    }

    const endIndex = startIndex + segLen - 1;

    // Use original words (with punctuation) for the phrase
    const phrase = transcriptData
      .slice(startIndex, endIndex + 1)
      .map((w) => w.word)
      .join(" ");

    // Derive clip path from clipSpec
    let clipPath = null;
    if (typeof clipSpec === "number") {
      clipPath = `clips/${clipSpec}.mp4`;
    } else if (typeof clipSpec === "string") {
      if (clipSpec.startsWith("clips/") || clipSpec.endsWith(".mp4")) {
        clipPath = clipSpec;
      } else {
        clipPath = `clips/${clipSpec}`;
      }
    }

    const highlight = {
      phrase,
      start_word: startIndex,
      end_word: endIndex,
      clip_path: clipPath,
      music_path: null,
      music_volume: 1.0,
      occurrence: 1,
    };

    highlights.push(highlight);
    addedCount++;
  });

  console.log(
    `Auto-highlights added from mapping (word-level): ${addedCount}`
  );
  if (addedCount > 0) {
    updateHighlightsList();
    updatePreviewHighlights();
  }
}

// =========================
// Upload video + TXT
// =========================

async function uploadVideo() {
  // Re-resolve the inputs in case the DOM was re-rendered
  const videoInputEl = document.getElementById("main-video-input");
  const transcriptInputEl = document.getElementById("transcript-file-input");

  const videoFile =
    videoInputEl && videoInputEl.files && videoInputEl.files[0]
      ? videoInputEl.files[0]
      : null;

  const txtFile =
    transcriptInputEl && transcriptInputEl.files && transcriptInputEl.files[0]
      ? transcriptInputEl.files[0]
      : null;


  if (!videoFile || !txtFile) {
    alert("Please select both video and TXT file");
    return;
  }

  const formData = new FormData();
  formData.append("video", videoFile);
  formData.append("transcript_file", txtFile);

  uploadBtn.disabled = true;
  uploadProgress.style.display = "block";

  try {
    const response = await fetch("/upload-video-with-txt", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.error) {
      alert("Error: " + data.error);
      return;
    }

    if (!data.subtitles || !data.transcript) {
      alert(
        "Error: Invalid response from server. Missing subtitles or transcript data."
      );
      console.error("Server response:", data);
      return;
    }

    currentVideoPath = data.video_path;
    transcriptData = data.transcript;
    subtitles = data.subtitles;

    displayTranscript(data.subtitles, data.transcript);
    displayMusicTranscript(data.transcript);
    transcriptPreviewSection.style.display = "block";
    selectionSection.style.display = "block";
    highlightsSection.style.display = "block";
    musicSelectionSection.style.display = "block";
    musicHighlightsSection.style.display = "block";
    processSection.style.display = "block";

    // If user already chose a mapping file, apply it now
    if (mappingData && mappingData.length) {
      applyAutoHighlightsFromMapping();
    }

    alert(
      `Script loaded! Found ${data.subtitles.length} subtitle boxes from ${data.word_count} words.`
    );
  } catch (error) {
    alert("Error uploading files: " + error.message);
  } finally {
    uploadProgress.style.display = "none";
    uploadBtn.disabled = false;
  }
}

function displayTranscript(subtitles, transcript) {
  // Clear both sections
  transcriptPreview.innerHTML = "";
  transcriptDisplay.innerHTML = "";

  // STEP 2: Read-only preview with line-by-line display and highlights
  const previewContainer = document.createElement("div");
  previewContainer.className = "transcript-preview-container";

  // Display transcript line by line with subtitle labels
  subtitles.forEach((subtitle, index) => {
    const subtitleBlock = document.createElement("div");
    subtitleBlock.className = "subtitle-block";

    // Add subtitle label
    const label = document.createElement("div");
    label.className = "subtitle-label-preview";
    label.textContent = `Subtitle ${index + 1}`;
    subtitleBlock.appendChild(label);

    // Add the line of text
    const lineDiv = document.createElement("div");
    lineDiv.className = "transcript-line";

    for (let i = subtitle.start_word; i <= subtitle.end_word; i++) {
      const wordSpan = document.createElement("span");
      wordSpan.className = "preview-word";
      wordSpan.dataset.index = i;
      wordSpan.textContent = transcript[i].word;
      lineDiv.appendChild(wordSpan);
      lineDiv.appendChild(document.createTextNode(" "));
    }

    subtitleBlock.appendChild(lineDiv);
    previewContainer.appendChild(subtitleBlock);
  });

  transcriptPreview.appendChild(previewContainer);

  // STEP 3: Interactive word selection
  const selectionContainer = document.createElement("div");
  selectionContainer.className = "transcript-text-container";

  // Track if user is currently dragging
  let isDragging = false;
  // Track if Shift key was used in the current selection
  let shiftUsedInSelection = false;

  // Display all words inline for selection
  transcript.forEach((entry, index) => {
    const wordSpan = document.createElement("span");
    wordSpan.className = "word-inline";
    wordSpan.textContent = entry.word;
    wordSpan.dataset.index = index;

    // Mouse events for selection
    wordSpan.addEventListener("mousedown", (e) => {
      isDragging = true;
      if (e.shiftKey && selectedRange) {
        shiftUsedInSelection = true;
        const newIndex = parseInt(wordSpan.dataset.index);
        selectedRange.end = newIndex;
        updateSelection(false); // Don't open file picker during drag
      } else {
        shiftUsedInSelection = false;
        selectedRange = {
          start: parseInt(wordSpan.dataset.index),
          end: parseInt(wordSpan.dataset.index),
        };
        updateSelection(false); // Don't open file picker during drag
      }
      // Auto-scroll to keep selected word visible when starting selection
      autoScrollToElement(wordSpan, transcriptDisplay);
    });

    wordSpan.addEventListener("mouseenter", (e) => {
      if (e.buttons === 1 && selectedRange && isDragging) {
        selectedRange.end = parseInt(wordSpan.dataset.index);
        updateSelection(false); // Don't open file picker during drag

        // Auto-scroll to keep selected word visible
        autoScrollToElement(wordSpan, transcriptDisplay);
      }
    });

    selectionContainer.appendChild(wordSpan);
    selectionContainer.appendChild(document.createTextNode(" "));
  });

  // Open file picker when mouse is released (selection complete)
  // But only if Shift wasn't used (Shift+Click opens picker on Shift release)
  selectionContainer.addEventListener("mouseup", () => {
    if (isDragging && selectedRange) {
      isDragging = false;
      // Only open file picker if Shift wasn't used
      // If Shift was used, wait for Shift key release
      if (!shiftUsedInSelection) {
        updateSelection(true); // Open file picker when selection is complete
      }
    }
  });

  // Open file picker when Shift key is released (for Shift+Click selections)
  document.addEventListener("keyup", (e) => {
    if (e.key === "Shift" && selectedRange && shiftUsedInSelection) {
      shiftUsedInSelection = false;
      updateSelection(true); // Open file picker when Shift is released
    }
  });

  transcriptDisplay.appendChild(selectionContainer);

  // Update preview with existing highlights
  updatePreviewHighlights();
}

function displayMusicTranscript(transcript) {
  // Clear music transcript display
  musicTranscriptDisplay.innerHTML = "";

  // STEP 5: Interactive word selection for music
  const musicContainer = document.createElement("div");
  musicContainer.className = "transcript-text-container";

  // Track if user is currently dragging for music selection
  let isDraggingMusic = false;
  // Track if Shift key was used in the current music selection
  let shiftUsedInMusicSelection = false;

  // Display all words inline for music selection
  transcript.forEach((entry, index) => {
    const wordSpan = document.createElement("span");
    wordSpan.className = "word-inline-music";
    wordSpan.textContent = entry.word;
    wordSpan.dataset.index = index;

    // Mouse events for music selection
    wordSpan.addEventListener("mousedown", (e) => {
      isDraggingMusic = true;
      if (e.shiftKey && selectedMusicRange) {
        shiftUsedInMusicSelection = true;
        const newIndex = parseInt(wordSpan.dataset.index);
        selectedMusicRange.end = newIndex;
        updateMusicSelection(false); // Don't open file picker during drag
      } else {
        shiftUsedInMusicSelection = false;
        selectedMusicRange = {
          start: parseInt(wordSpan.dataset.index),
          end: parseInt(wordSpan.dataset.index),
        };
        updateMusicSelection(false); // Don't open file picker during drag
      }
      // Auto-scroll to keep selected word visible when starting selection
      autoScrollToElement(wordSpan, musicTranscriptDisplay);
    });

    wordSpan.addEventListener("mouseenter", (e) => {
      if (e.buttons === 1 && selectedMusicRange && isDraggingMusic) {
        selectedMusicRange.end = parseInt(wordSpan.dataset.index);
        updateMusicSelection(false); // Don't open file picker during drag

        // Auto-scroll to keep selected word visible
        autoScrollToElement(wordSpan, musicTranscriptDisplay);
      }
    });

    musicContainer.appendChild(wordSpan);
    musicContainer.appendChild(document.createTextNode(" "));
  });

  // Open file picker when mouse is released (selection complete)
  // But only if Shift wasn't used (Shift+Click opens picker on Shift release)
  musicContainer.addEventListener("mouseup", () => {
    if (isDraggingMusic && selectedMusicRange) {
      isDraggingMusic = false;
      // Only open file picker if Shift wasn't used
      // If Shift was used, wait for Shift key release
      if (!shiftUsedInMusicSelection) {
        updateMusicSelection(true); // Open file picker when selection is complete
      }
    }
  });

  // Open file picker when Shift key is released (for Shift+Click music selections)
  // Use a separate listener for music to avoid conflicts
  const musicShiftKeyupHandler = (e) => {
    if (e.key === "Shift" && selectedMusicRange && shiftUsedInMusicSelection) {
      shiftUsedInMusicSelection = false;
      updateMusicSelection(true); // Open file picker when Shift is released
    }
  };
  document.addEventListener("keyup", musicShiftKeyupHandler);

  musicTranscriptDisplay.appendChild(musicContainer);

  // Update with existing music highlights
  updateMusicHighlightsDisplay();
}

function updatePreviewHighlights() {
  // Clear all highlights in Step 2 preview
  document.querySelectorAll(".preview-word").forEach((el) => {
    el.classList.remove("highlighted");
  });

  // Clear all highlights in Step 3 selection
  document.querySelectorAll(".word-inline").forEach((el) => {
    el.classList.remove("highlighted");
  });

  // Apply highlights based on current highlights array to BOTH Step 2 and Step 3
  highlights.forEach((highlight) => {
    const start = Math.min(highlight.start_word, highlight.end_word);
    const end = Math.max(highlight.start_word, highlight.end_word);

    for (let i = start; i <= end; i++) {
      // Highlight in Step 2 preview
      const previewWordEl = document.querySelector(
        `.preview-word[data-index="${i}"]`
      );
      if (previewWordEl) {
        previewWordEl.classList.add("highlighted");
      }

      // Highlight in Step 3 selection
      const selectionWordEl = document.querySelector(
        `.word-inline[data-index="${i}"]`
      );
      if (selectionWordEl) {
        selectionWordEl.classList.add("highlighted");
      }
    }
  });
}

function updateSelection(showFilePicker = false) {
  // Clear previous selection in Step 3
  document.querySelectorAll(".word-inline.selected").forEach((el) => {
    el.classList.remove("selected");
  });

  // Clear previous selection preview in Step 2
  document.querySelectorAll(".preview-word.selecting").forEach((el) => {
    el.classList.remove("selecting");
  });

  if (!selectedRange) return;

  const start = Math.min(selectedRange.start, selectedRange.end);
  const end = Math.max(selectedRange.start, selectedRange.end);

  // Highlight selected words in Step 3
  for (let i = start; i <= end; i++) {
    const wordEl = document.querySelector(`.word-inline[data-index="${i}"]`);
    if (wordEl) {
      wordEl.classList.add("selected");
    }

    // Also highlight in Step 2 preview (real-time)
    const previewWordEl = document.querySelector(
      `.preview-word[data-index="${i}"]`
    );
    if (previewWordEl) {
      previewWordEl.classList.add("selecting");
    }
  }

  // Get selected text
  const selectedWords = transcriptData
    .slice(start, end + 1)
    .map((e) => e.word)
    .join(" ");

  // Update selection controls text
  selectedTextSpan.textContent = selectedWords;

  // Open file picker when selection is complete
  if (showFilePicker) {
    let tempFileInput = document.getElementById("temp-video-input");
    if (!tempFileInput) {
      tempFileInput = document.createElement("input");
      tempFileInput.type = "file";
      tempFileInput.id = "temp-video-input";
      tempFileInput.accept = "video/*";
      tempFileInput.style.display = "none";
      document.body.appendChild(tempFileInput);

      // Handle file selection ONLY when a file is actually chosen
      tempFileInput.addEventListener("change", async (e) => {
        const file =
          e.target.files && e.target.files[0] ? e.target.files[0] : null;
        if (file && selectedRange) {
          await uploadAndAttachVideo(file);
        }
        // Reset so the same file can be picked again later
        tempFileInput.value = "";
      });
    }

    // Trigger file picker
    tempFileInput.click();
  }
}



function autoScrollToElement(element, container) {
  // Auto-scroll container to keep element visible when selecting
  if (!element || !container) return;

  const containerRect = container.getBoundingClientRect();
  const elementRect = element.getBoundingClientRect();

  // Calculate scroll boundaries
  const scrollThreshold = 80; // Start scrolling when within 80px of edge
  const scrollPadding = 30; // Keep 30px padding from edge

  // Check if element is above visible area or near top
  if (elementRect.top < containerRect.top + scrollThreshold) {
    // Scroll element into view at the top with padding
    element.scrollIntoView({
      behavior: "auto", // Instant scroll during drag
      block: "nearest",
      inline: "nearest",
    });
    // Fine-tune scroll position
    if (container.scrollTop > 0) {
      container.scrollTop = Math.max(0, container.scrollTop - scrollPadding);
    }
  }
  // Check if element is below visible area or near bottom
  else if (elementRect.bottom > containerRect.bottom - scrollThreshold) {
    // Scroll element into view at the bottom with padding
    element.scrollIntoView({
      behavior: "auto", // Instant scroll during drag
      block: "nearest",
      inline: "nearest",
    });
    // Fine-tune scroll position
    container.scrollTop = container.scrollTop + scrollPadding;
  }
}

function cancelSelection() {
  selectedRange = null;
  currentSubtitleIndex = null;

  // Clear selection in Step 3
  document.querySelectorAll(".word-inline.selected").forEach((el) => {
    el.classList.remove("selected");
  });

  // Clear real-time preview in Step 2
  document.querySelectorAll(".preview-word.selecting").forEach((el) => {
    el.classList.remove("selecting");
  });

  selectionControls.style.display = "none";
}

async function uploadAndAttachVideo(file) {
  if (!file) {
    return;
  }

  if (!selectedRange) {
    alert("No text selected");
    return;
  }

  try {
    // Upload the video file
    const formData = new FormData();
    formData.append("file", file);

    const uploadResponse = await fetch("/upload-clip", {
      method: "POST",
      body: formData,
    });

    const uploadData = await uploadResponse.json();

    if (uploadData.error) {
      alert("Error: " + uploadData.error);
      return;
    }

    // Get the clip path from server response
    const clipPath = uploadData.file_path;

    // Add to existing clips dropdown and select it
    const option = document.createElement("option");
    option.value = clipPath;
    option.textContent = file.name;
    existingClipsSelect.appendChild(option);
    existingClipsSelect.value = clipPath;

    // Reuse the existing, battle-tested highlight creation flow
    // This will:
    //  - read selectedRange
    //  - push into `highlights`
    //  - update the "Review B-roll highlights" list
    //  - update preview highlights
    //  - cancel/clear the selection
    addHighlight();
  } catch (error) {
    alert("Error uploading video: " + error.message);
  }
}


// Music selection functions
function updateMusicSelection(showFilePicker = false) {
  // Clear previous selection in Step 5
  document.querySelectorAll(".word-inline-music.selected").forEach((el) => {
    el.classList.remove("selected");
  });

  if (!selectedMusicRange) return;

  const start = Math.min(selectedMusicRange.start, selectedMusicRange.end);
  const end = Math.max(selectedMusicRange.start, selectedMusicRange.end);

  // Highlight selected words in Step 5
  for (let i = start; i <= end; i++) {
    const wordEl = document.querySelector(
      `.word-inline-music[data-index="${i}"]`
    );
    if (wordEl) {
      wordEl.classList.add("selected");
    }
  }

  // Show music selection controls
  const selectedWords = transcriptData
    .slice(start, end + 1)
    .map((e) => e.word)
    .join(" ");
  musicSelectedText.textContent = selectedWords;
  musicSelectionControls.style.display = "block";

  // Directly open file picker when selection is complete
  if (showFilePicker) {
    // Create a temporary file input if it doesn't exist
    let tempMusicInput = document.getElementById("temp-music-input");
    if (!tempMusicInput) {
      tempMusicInput = document.createElement("input");
      tempMusicInput.type = "file";
      tempMusicInput.id = "temp-music-input";
      tempMusicInput.accept = "audio/*";
      tempMusicInput.style.display = "none";
      document.body.appendChild(tempMusicInput);

      // Handle file selection
      tempMusicInput.addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (file && selectedMusicRange) {
          await uploadAndAttachMusic(file);
          // Reset the input for next selection
          tempMusicInput.value = "";
        }
      });
    }

    // Trigger file picker
    tempMusicInput.click();
  }
}

function cancelMusicSelection() {
  selectedMusicRange = null;

  // Clear selection in Step 5
  document.querySelectorAll(".word-inline-music.selected").forEach((el) => {
    el.classList.remove("selected");
  });

  musicSelectionControls.style.display = "none";
}

async function uploadClip() {
  const file = clipInput.files[0];
  if (!file) {
    alert("Please select a file to upload");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  uploadClipBtn.disabled = true;

  try {
    const response = await fetch("/upload-clip", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.error) {
      alert("Error: " + data.error);
      return;
    }

    // Add to existing clips dropdown
    const option = document.createElement("option");
    option.value = data.file_path;
    option.textContent = file.name;
    existingClipsSelect.appendChild(option);
    existingClipsSelect.value = data.file_path;

    alert("File uploaded successfully!");
  } catch (error) {
    alert("Error uploading file: " + error.message);
  } finally {
    uploadClipBtn.disabled = false;
  }
}

async function loadExistingClips() {
  try {
    const response = await fetch("/list-clips");
    const data = await response.json();

    // Add clips only (videos)
    data.clips.forEach((clip) => {
      const option = document.createElement("option");
      option.value = `clips/${clip}`;
      option.textContent = `📹 ${clip}`;
      existingClipsSelect.appendChild(option);
    });
  } catch (error) {
    console.error("Error loading clips:", error);
  }
}

async function uploadMusicFile() {
  const file = musicInput.files[0];
  if (!file) {
    alert("Please select a music file to upload");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  uploadMusicBtn.disabled = true;

  try {
    const response = await fetch("/upload-clip", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.error) {
      alert("Error: " + data.error);
      return;
    }

    // Add to existing music dropdown
    const option = document.createElement("option");
    option.value = data.file_path;
    option.textContent = file.name;
    existingMusicSelect.appendChild(option);
    existingMusicSelect.value = data.file_path;

    alert("Music uploaded successfully!");
  } catch (error) {
    alert("Error uploading music: " + error.message);
  } finally {
    uploadMusicBtn.disabled = false;
  }
}

async function uploadAndAttachMusic(file) {
  if (!file) {
    return;
  }

  if (!selectedMusicRange) {
    alert("No text selected");
    return;
  }

  try {
    // Upload the music file
    const formData = new FormData();
    formData.append("file", file);

    const uploadResponse = await fetch("/upload-clip", {
      method: "POST",
      body: formData,
    });

    const uploadData = await uploadResponse.json();

    if (uploadData.error) {
      alert("Error: " + uploadData.error);
      return;
    }

    // Get the music path
    const musicPath = uploadData.file_path;

    // Add music highlight with the uploaded music
    const start = Math.min(selectedMusicRange.start, selectedMusicRange.end);
    const end = Math.max(selectedMusicRange.start, selectedMusicRange.end);
    const phrase = transcriptData
      .slice(start, end + 1)
      .map((e) => e.word)
      .join(" ");

    const musicHighlight = {
      phrase: phrase,
      start_word: start,
      end_word: end,
      music_path: musicPath,
      music_volume: parseFloat(musicVolume.value),
      occurrence: 1,
    };

    musicHighlights.push(musicHighlight);

    // Update the existing music dropdown
    const option = document.createElement("option");
    option.value = musicPath;
    option.textContent = file.name;
    existingMusicSelect.appendChild(option);

    // Update UI
    updateMusicHighlightsList();
    updateMusicHighlightsDisplay();

    // Clear selection
    selectedMusicRange = null;
    document.querySelectorAll(".word-inline-music.selected").forEach((el) => {
      el.classList.remove("selected");
    });
    musicSelectionControls.style.display = "none";
  } catch (error) {
    alert("Error uploading music: " + error.message);
  }
}

async function loadExistingMusic() {
  try {
    const response = await fetch("/list-clips");
    const data = await response.json();

    // Add audio files to music dropdown
    data.audio_files.forEach((audio) => {
      const option = document.createElement("option");
      option.value = `audio_files/${audio}`;
      option.textContent = `🎵 ${audio}`;
      existingMusicSelect.appendChild(option);
    });
  } catch (error) {
    console.error("Error loading music:", error);
  }
}

function addMusicHighlight() {
  if (!selectedMusicRange) return;

  const musicPath = existingMusicSelect.value;
  if (!musicPath) {
    alert("Please select or upload a music/audio file");
    return;
  }

  const start = Math.min(selectedMusicRange.start, selectedMusicRange.end);
  const end = Math.max(selectedMusicRange.start, selectedMusicRange.end);
  const phrase = transcriptData
    .slice(start, end + 1)
    .map((e) => e.word)
    .join(" ");

  const musicHighlight = {
    phrase: phrase,
    start_word: start,
    end_word: end,
    music_path: musicPath,
    music_volume: parseFloat(musicVolume.value),
    occurrence: 1,
  };

  musicHighlights.push(musicHighlight);

  updateMusicHighlightsList();
  updateMusicHighlightsDisplay();
  cancelMusicSelection();
}

function updateMusicHighlightsList() {
  musicHighlightsList.innerHTML = "";

  if (musicHighlights.length === 0) {
    musicHighlightsList.innerHTML =
      '<p style="color: #999; text-align: center; padding: 20px;">No music/audio added yet. Select words in Step 5 to add music.</p>';
    return;
  }

  // Add header
  const header = document.createElement("div");
  header.className = "subtitles-header";
  header.innerHTML = `
    <h3>Your Music/Audio Highlights (${musicHighlights.length})</h3>
    <p class="subtitles-description">Each music/audio will play during the selected words</p>
  `;
  musicHighlightsList.appendChild(header);

  // Add each music highlight
  musicHighlights.forEach((music, index) => {
    const item = document.createElement("div");
    item.className = "highlight-item";

    const badge = document.createElement("div");
    badge.className = "subtitle-badge";
    badge.textContent = `#${index + 1}`;

    const info = document.createElement("div");
    info.className = "highlight-info";

    const phrase = document.createElement("div");
    phrase.className = "highlight-phrase";
    phrase.textContent = `"${music.phrase}"`;

    const details = document.createElement("div");
    details.className = "highlight-details";

    const musicName = music.music_path.split("/").pop();
    details.innerHTML = `
      <span class="detail-item">🎵 ${musicName}</span>
      <span class="detail-item">🔊 Volume: ${music.music_volume.toFixed(
        1
      )}</span>
      <span class="detail-item">📝 Words ${music.start_word + 1}-${
      music.end_word + 1
    }</span>
    `;

    info.appendChild(phrase);
    info.appendChild(details);

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "btn btn-danger";
    deleteBtn.textContent = "Delete";
    deleteBtn.onclick = () => deleteMusicHighlight(index);

    item.appendChild(badge);
    item.appendChild(info);
    item.appendChild(deleteBtn);

    musicHighlightsList.appendChild(item);
  });
}

function updateMusicHighlightsDisplay() {
  // Clear all music highlights in Step 5
  document.querySelectorAll(".word-inline-music").forEach((el) => {
    el.classList.remove("highlighted");
  });

  // Apply music highlights
  musicHighlights.forEach((music) => {
    const start = Math.min(music.start_word, music.end_word);
    const end = Math.max(music.start_word, music.end_word);

    for (let i = start; i <= end; i++) {
      const wordEl = document.querySelector(
        `.word-inline-music[data-index="${i}"]`
      );
      if (wordEl) {
        wordEl.classList.add("highlighted");
      }
    }
  });
}

function deleteMusicHighlight(index) {
  musicHighlights.splice(index, 1);
  updateMusicHighlightsList();
  updateMusicHighlightsDisplay();
}

function addHighlight() {
  if (!selectedRange) return;

  const clipPath = existingClipsSelect.value;
  if (!clipPath) {
    alert("Please select or upload a clip");
    return;
  }

  const start = Math.min(selectedRange.start, selectedRange.end);
  const end = Math.max(selectedRange.start, selectedRange.end);
  const phrase = transcriptData
    .slice(start, end + 1)
    .map((e) => e.word)
    .join(" ");

  const highlight = {
    phrase: phrase,
    start_word: start,
    end_word: end,
    clip_path: clipPath,
    music_path: null,
    music_volume: 1.0,
    occurrence: 1,
  };

  highlights.push(highlight);

  updateHighlightsList();
  updatePreviewHighlights(); // Update Step 2 and Step 3
  cancelSelection();
}

function updateHighlightsList() {
  highlightsList.innerHTML = "";

  if (highlights.length === 0) {
    highlightsList.innerHTML =
      '<p style="color: #999; text-align: center; padding: 20px;">No subtitles added yet. Select text in the transcript above to create your first subtitle.</p>';
    return;
  }

  // Add header
  const header = document.createElement("div");
  header.className = "subtitles-header";
  header.innerHTML = `
    <h3>Your Assigned Highlights (${highlights.length})</h3>
    <p class="subtitles-description">Each highlight will appear during the selected words with the assigned clip/audio</p>
  `;
  highlightsList.appendChild(header);

  highlights.forEach((highlight, index) => {
    const item = document.createElement("div");
    item.className = "highlight-item";

    // Subtitle number badge
    const badge = document.createElement("div");
    badge.className = "subtitle-badge";
    badge.textContent = `#${index + 1}`;

    const info = document.createElement("div");
    info.className = "highlight-info";

    // Subtitle text label
    const label = document.createElement("div");
    label.className = "subtitle-label";
    label.textContent = `Subtitle ${index + 1}`;

    const phrase = document.createElement("div");
    phrase.className = "highlight-phrase";
    phrase.textContent = `"${highlight.phrase}"`;

    const details = document.createElement("div");
    details.className = "highlight-details";
    const fileName = (highlight.clip_path || highlight.music_path)
      .split("/")
      .pop();
    const fileType = highlight.clip_path ? "📹 Video Clip" : "🎵 Audio/Music";
    const wordRange = `Words ${highlight.start_word + 1}-${
      highlight.end_word + 1
    }`;
    details.innerHTML = `
      <span class="detail-item">${fileType}: <strong>${fileName}</strong></span>
      <span class="detail-item">Volume: <strong>${highlight.music_volume}</strong></span>
      <span class="detail-item">${wordRange}</span>
    `;

    info.appendChild(label);
    info.appendChild(phrase);
    info.appendChild(details);

    const removeBtn = document.createElement("button");
    removeBtn.className = "btn btn-danger";
    removeBtn.innerHTML = "🗑️<br>Remove";
    removeBtn.onclick = () => removeHighlight(index);

    item.appendChild(badge);
    item.appendChild(info);
    item.appendChild(removeBtn);
    highlightsList.appendChild(item);
  });
}

function removeHighlight(index) {
  highlights.splice(index, 1);
  updateHighlightsList();
  updatePreviewHighlights(); // Update Step 2 preview
}

// State management functions
function saveState() {
  const state = {
    currentVideoPath: currentVideoPath,
    transcriptData: transcriptData,
    subtitles: subtitles,
    highlights: highlights,
    musicHighlights: musicHighlights,
    aspectRatio: aspectRatioSelect.value || "4:5",
    videoFilename: videoFilename.textContent,
    transcriptFilename: transcriptFilename.textContent,
    timestamp: Date.now(),
  };
  sessionStorage.setItem("videoEditorState", JSON.stringify(state));
  console.log("State saved:", state);
}

function restoreState() {
  const savedState = sessionStorage.getItem("videoEditorState");
  if (!savedState) {
    alert("No saved state found. Cannot restore.");
    return false;
  }

  try {
    const state = JSON.parse(savedState);

    // Restore all state variables
    currentVideoPath = state.currentVideoPath;
    transcriptData = state.transcriptData || [];
    subtitles = state.subtitles || [];
    highlights = state.highlights || [];
    musicHighlights = state.musicHighlights || [];

    // Restore UI elements
    if (state.videoFilename) {
      videoFilename.textContent = state.videoFilename;
    }
    if (state.transcriptFilename) {
      transcriptFilename.textContent = state.transcriptFilename;
    }
    if (state.aspectRatio) {
      aspectRatioSelect.value = state.aspectRatio;
    }

    // Restore transcript displays
    if (transcriptData.length > 0 && subtitles.length > 0) {
      displayTranscript(subtitles, transcriptData);
      displayMusicTranscript(transcriptData);

      // Restore highlights displays
      updateHighlightsList();
      updateMusicHighlightsList();
      updatePreviewHighlights();
      updateMusicHighlightsDisplay();

      // Show all relevant sections
      transcriptPreviewSection.style.display = "block";
      selectionSection.style.display = "block";
      highlightsSection.style.display = "block";
      musicSelectionSection.style.display = "block";
      musicHighlightsSection.style.display = "block";
      processSection.style.display = "block";
    }

    // Hide result section
    resultSection.style.display = "none";

    // Clear any existing video preview when going back to edit
    if (videoPreview) {
      try {
        videoPreview.pause();
      } catch (e) {
        console.warn("Error pausing video preview:", e);
      }
      videoPreview.removeAttribute("src");
      videoPreview.load(); // force the <video> element to reset
    }
    if (videoPreviewContainer) {
      videoPreviewContainer.style.display = "none";
    }

    // Scroll to top
    window.scrollTo({ top: 0, behavior: "smooth" });

    alert(
      `State restored! You have ${highlights.length} clip highlights and ${musicHighlights.length} music highlights. You can now edit them.`
    );
    return true;
  } catch (error) {
    console.error("Error restoring state:", error);
    alert("Error restoring state: " + error.message);
    return false;
  }
}

function goBackAndEdit() {
  if (restoreState()) {
    // State restored successfully
    console.log("Returned to editing mode");
  }
}

// Project loading functions
async function loadProjectList() {
  loadProjectBtn.disabled = true;
  projectList.innerHTML = "<p>Loading projects...</p>";
  projectListContainer.style.display = "block";

  try {
    const response = await fetch("/list-projects");
    const data = await response.json();

    if (data.error) {
      alert("Error: " + data.error);
      return;
    }

    if (!data.projects || data.projects.length === 0) {
      projectList.innerHTML =
        '<p style="color: #999; text-align: center; padding: 20px;">No projects found in S3.</p>';
      return;
    }

    projectList.innerHTML = "";
    data.projects.forEach((project, index) => {
      const projectItem = document.createElement("div");
      projectItem.className = "highlight-item";
      projectItem.style.marginBottom = "10px";
      projectItem.style.cursor = "pointer";
      projectItem.style.border = "1px solid #ddd";
      projectItem.style.borderRadius = "5px";
      projectItem.style.padding = "15px";
      projectItem.style.transition = "background-color 0.2s";

      const date = new Date(project.last_modified);
      const formattedDate = date.toLocaleString();

      projectItem.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <div>
            <strong>${project.filename}</strong>
            <div style="color: #666; font-size: 0.9em; margin-top: 5px;">
              Modified: ${formattedDate} | Size: ${(project.size / 1024).toFixed(
                2
              )} KB
            </div>
          </div>
          <button class="btn btn-primary" onclick="loadProjectFromS3('${
            project.key
          }')">
            Load
          </button>
        </div>
      `;

      projectItem.addEventListener("mouseenter", () => {
        projectItem.style.backgroundColor = "#f5f5f5";
      });
      projectItem.addEventListener("mouseleave", () => {
        projectItem.style.backgroundColor = "white";
      });

      projectList.appendChild(projectItem);
    });
  } catch (error) {
    alert("Error loading projects: " + error.message);
    projectList.innerHTML =
      '<p style="color: red;">Error loading projects.</p>';
  } finally {
    loadProjectBtn.disabled = false;
  }
}

async function saveProjectToS3() {
  if (!currentVideoPath) {
    alert("Please upload a video first");
    return;
  }

  if (highlights.length === 0 && musicHighlights.length === 0) {
    alert("Please add at least one highlight or music before saving");
    return;
  }

  saveProjectBtn.disabled = true;
  saveProjectStatus.style.display = "block";
  saveProjectStatus.innerHTML =
    '<p style="color: #666;">Saving project...</p>';

  try {
    const projectName = projectNameInput.value.trim() || null;
    const aspectRatio = aspectRatioSelect.value || "4:5";

    // Combine highlights and music highlights
    const allHighlights = [...highlights, ...musicHighlights];

    const response = await fetch("/save-project", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        video_path: currentVideoPath,
        highlights: allHighlights,
        transcript: transcriptData,
        subtitle_sentences: subtitles,
        aspect_ratio: aspectRatio,
        project_name: projectName,
      }),
    });

    const data = await response.json();

    if (data.error) {
      saveProjectStatus.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
      return;
    }

    saveProjectStatus.innerHTML = `
      <p style="color: green; font-weight: bold;">✅ ${data.message}</p>
      <p style="color: #666; font-size: 0.9em; margin-top: 5px;">
        Project saved as: <strong>${data.project_filename}</strong>
      </p>
    `;

    // Clear project name input
    projectNameInput.value = "";

    // Auto-hide success message after 5 seconds
    setTimeout(() => {
      saveProjectStatus.style.display = "none";
    }, 5000);
  } catch (error) {
    saveProjectStatus.innerHTML = `<p style="color: red;">Error saving project: ${error.message}</p>`;
  } finally {
    saveProjectBtn.disabled = false;
  }
}

async function loadProjectFromS3(projectKey) {
  try {
    const response = await fetch("/load-project", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        project_key: projectKey,
      }),
    });

    const data = await response.json();

    if (data.error) {
      alert("Error loading project: " + data.error);
      return;
    }

    const project = data.project;

    // Restore state from project
    currentVideoPath = project.project_info.video_path;
    transcriptData = project.transcript || [];
    subtitles = project.subtitle_sentences || [];

    // Separate clip highlights from music highlights
    const allHighlights = project.highlights || [];
    highlights = [];
    musicHighlights = [];

    allHighlights.forEach((highlight) => {
      if (highlight.music_path && !highlight.clip_path) {
        // This is a music-only highlight
        musicHighlights.push(highlight);
      } else {
        // This is a clip highlight (may also have music)
        highlights.push(highlight);
      }
    });

    // Restore aspect ratio
    if (project.project_info.aspect_ratio) {
      aspectRatioSelect.value = project.project_info.aspect_ratio;
    }

    // Restore filenames
    if (project.project_info.video_path) {
      const videoName = project.project_info.video_path.split("/").pop();
      videoFilename.textContent = `Selected: ${videoName}`;

      // Note: We assume the video file still exists locally
      // If it doesn't, the user will need to re-upload it
    }

    // Restore transcript displays
    if (transcriptData.length > 0 && subtitles.length > 0) {
      displayTranscript(subtitles, transcriptData);
      displayMusicTranscript(transcriptData);

      // Restore highlights displays
      updateHighlightsList();
      updateMusicHighlightsList();
      updatePreviewHighlights();
      updateMusicHighlightsDisplay();

      // Show all relevant sections
      transcriptPreviewSection.style.display = "block";
      selectionSection.style.display = "block";
      highlightsSection.style.display = "block";
      musicSelectionSection.style.display = "block";
      musicHighlightsSection.style.display = "block";
      processSection.style.display = "block";
    }

    // Hide project list
    projectListContainer.style.display = "none";

    // Scroll to top
    window.scrollTo({ top: 0, behavior: "smooth" });

    alert(
      `Project loaded successfully! You have ${highlights.length} clip highlights and ${musicHighlights.length} music highlights. You can now edit them.`
    );
  } catch (error) {
    alert("Error loading project: " + error.message);
  }
}

async function processVideo() {
  if (!currentVideoPath) {
    alert("Please upload a video first");
    return;
  }

  if (highlights.length === 0) {
    alert("Please add at least one highlight");
    return;
  }

  // Save state before processing
  saveState();

  processBtn.disabled = true;
  processProgress.style.display = "block";

  // Combine highlights and music highlights
  const allHighlights = [...highlights, ...musicHighlights];

  try {
    const aspectRatio = aspectRatioSelect.value || "4:5";

    const response = await fetch("/process-video", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        video_path: currentVideoPath,
        highlights: allHighlights,
        transcript: transcriptData,
        preserve_audio: true,
        subtitle_sentences: subtitles,
        aspect_ratio: aspectRatio,
      }),
    });

    const data = await response.json();

    if (data.error) {
      alert("Error: " + data.error);
      return;
    }

    resultMessage.textContent = data.message;
    downloadBtn.onclick = () => {
      const url = `/download/${encodeURIComponent(data.output_filename)}`;
      window.location.href = url;
    };
    

    // Set video preview source (cache-busted + hard reload)
    if (data.output_filename) {
      const ts = Date.now();
      const newSrc = `/video/${data.output_filename}?t=${ts}`;

      // Reset the video element first to avoid weird stale states
      try {
        videoPreview.pause();
      } catch (e) {
        console.warn("Error pausing video preview:", e);
      }
      videoPreview.removeAttribute("src");
      videoPreview.load();

      // Now set the fresh URL
      videoPreview.src = newSrc;
      videoPreviewContainer.style.display = "block";
      videoPreview.load(); // actually load the new video file
    } else {
      videoPreviewContainer.style.display = "none";
    }
    

    // Allow batch UI (if loaded) to react to completed renders.
    try {
      window.dispatchEvent(new CustomEvent("batchVideoProcessed", { detail: data }));
    } catch (e) {
      console.warn("Failed to dispatch batchVideoProcessed event:", e);
    }

    resultSection.style.display = "block";
    processProgress.style.display = "none";
  } catch (error) {
    alert("Error processing video: " + error.message);
    processProgress.style.display = "none";
  } finally {
    processBtn.disabled = false;
  }
}

// Hook up volume display initially
if (musicVolume && musicVolumeDisplay) {
  musicVolumeDisplay.textContent = musicVolume.value;
}
