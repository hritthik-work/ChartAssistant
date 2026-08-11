const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const escapeHtml = (value) =>
  String(value).replace(/[&<>'"]/g, (char) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char],
  );

let activeRequest = null;
let requestSequence = 0;
let loadingTimers = [];
let loadingInterval = null;
let selectedFiles = [];
let uploadInterval = null;

const serviceNames = {
  azure_openai_chat: "Answering model",
  azure_openai_embeddings: "Chart embeddings",
  azure_ai_search_query: "Knowledge-base search",
  azure_ai_search_ingestion: "Chart uploads",
};

async function checkSystemStatus() {
  const control = $("#system-status");
  control.className = "system-status checking";
  control.disabled = true;
  $("#status-icon").textContent = "↻";
  $("#status-label").textContent = "Checking system status";
  $("#status-tooltip").textContent = "Running live service checks…";
  try {
    const response = await fetch("/health/services?deep=true");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.message || `Health check returned ${response.status}`);
    const failures = Object.values(payload.services || {}).filter(
      (service) => !["ok", "configured"].includes(service.status),
    );
    if (payload.status === "ok" && failures.length === 0) {
      control.className = "system-status healthy";
      $("#status-icon").textContent = "✓";
      $("#status-label").textContent = "System operational";
      $("#status-tooltip").textContent = "All answering, search, and upload services are available.\nClick to check again.";
    } else {
      control.className = "system-status error";
      $("#status-icon").textContent = "!";
      $("#status-label").textContent = "System issue";
      $("#status-tooltip").textContent = failures.map((service) => {
        const detail = service.error_code || service.status || "unavailable";
        return `${serviceNames[service.service] || service.service}: ${detail}`;
      }).join("\n") || "One or more services are unavailable.\nClick to check again.";
    }
  } catch (error) {
    control.className = "system-status error";
    $("#status-icon").textContent = "!";
    $("#status-label").textContent = "System issue";
    $("#status-tooltip").textContent = `Health check failed: ${error.message}\nClick to check again.`;
  } finally {
    control.disabled = false;
  }
}

async function refreshKnowledgeBase() {
  const [health, documents] = await Promise.all([
    fetch("/health").then((response) => response.json()),
    fetch("/documents").then((response) => response.json()),
  ]);
  const patients = new Set(documents.map((document) => document.patient_id).filter(Boolean));
  $("#knowledge-stats").innerHTML = [
    `<div class="stat"><strong>${patients.size}</strong><span>patients</span></div>`,
    `<div class="stat"><strong>${documents.length}</strong><span>documents</span></div>`,
  ].join("");
}

$("#system-status").addEventListener("click", checkSystemStatus);

$$('[data-query]').forEach((button) =>
  button.addEventListener("click", () => {
    $("#query").value = button.dataset.query;
    $("#query").focus();
  }),
);
$("#submit").addEventListener("click", runQuestion);
$("#cancel").addEventListener("click", () => activeRequest?.abort());
$("#query").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    runQuestion();
  }
});
$("#query").addEventListener("input", () => $("#query-error").classList.add("hidden"));

function resetResult() {
  $("#result").classList.add("hidden");
  ["#status-title", "#status-badge", "#confidence-badge", "#answer", "#trace"].forEach(
    (selector) => { $(selector).textContent = ""; },
  );
  ["#citations", "#controls", "#missing", "#tool-panel"].forEach((selector) =>
    $(selector).replaceChildren(),
  );
  $("#patient-badge").classList.add("hidden");
  $("#tool-panel").classList.add("hidden");
}

function setQuestionBusy(busy) {
  $("#query").disabled = busy;
  $("#submit").disabled = busy;
  $$('[data-query]').forEach((button) => { button.disabled = busy; });
}

function setAnswerStep(index, message) {
  $$('[data-loading-step]').forEach((step, stepIndex) => {
    step.classList.toggle("active", stepIndex === index);
    step.classList.toggle("complete", stepIndex < index);
  });
  $("#loading-status").textContent = message;
}

function startQuestionLoading(query) {
  loadingTimers.forEach(clearTimeout);
  if (loadingInterval) clearInterval(loadingInterval);
  setAnswerStep(0, "Understanding the question.");
  $("#loading-query").textContent = `“${query}”`;
  $("#loading-elapsed").textContent = "0s elapsed";
  $("#loading").classList.remove("hidden");
  const started = performance.now();
  loadingInterval = setInterval(() => {
    $("#loading-elapsed").textContent = `${Math.floor((performance.now() - started) / 1000)}s elapsed`;
  }, 250);
  loadingTimers = [
    setTimeout(() => setAnswerStep(1, "Searching the relevant patient chart."), 900),
    setTimeout(() => setAnswerStep(2, "Preparing the answer and sources."), 4000),
  ];
}

function stopQuestionLoading() {
  loadingTimers.forEach(clearTimeout);
  loadingTimers = [];
  if (loadingInterval) clearInterval(loadingInterval);
  loadingInterval = null;
  $("#loading").classList.add("hidden");
}

async function runQuestion() {
  const query = $("#query").value.trim();
  resetResult();
  if (query.length < 3) {
    $("#query-error").classList.remove("hidden");
    $("#query").focus();
    return;
  }
  activeRequest?.abort();
  const controller = new AbortController();
  const sequence = ++requestSequence;
  activeRequest = controller;
  startQuestionLoading(query);
  setQuestionBusy(true);
  $("#submit").textContent = "Looking through charts…";
  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
      signal: controller.signal,
    });
    const payload = await response.json().catch(() => ({}));
    if (sequence !== requestSequence) return;
    if (!response.ok) throw new Error(payload.message || "The question could not be answered");
    renderAnswer(payload);
  } catch (error) {
    if (error.name === "AbortError" || sequence !== requestSequence) return;
    renderAnswerError(error.message);
  } finally {
    if (sequence !== requestSequence) return;
    activeRequest = null;
    stopQuestionLoading();
    setQuestionBusy(false);
    $("#submit").innerHTML = "Ask Chart Assistant <span>→</span>";
    if (!$("#result").classList.contains("hidden")) {
      $("#result").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }
}

function renderAnswerError(message) {
  $("#result").classList.remove("hidden");
  $("#status-title").textContent = "Unable to answer";
  $("#status-badge").textContent = "Error";
  $("#status-badge").className = "badge stop";
  $("#confidence-badge").textContent = "Try again";
  $("#confidence-badge").className = "badge warn";
  $("#answer").textContent = message;
  $("#citations").innerHTML = "<p>No sources were returned.</p>";
  $("#missing").innerHTML = "<li>Retry after checking the service status.</li>";
}

function renderAnswer(data) {
  $("#result").classList.remove("hidden");
  const needsPatient = data.status === "refused" && data.trace.model_calls === 0 && data.intent === "clinical_evidence";
  $("#status-title").textContent = needsPatient ? "Which patient?" : data.status === "answered" ? "Answer" : data.status === "partial" ? "Partial answer" : "No answer found";
  $("#status-badge").textContent = data.status;
  $("#status-badge").className = `badge ${data.status === "refused" ? "stop" : data.status === "partial" ? "warn" : ""}`;
  $("#confidence-badge").textContent = `${data.confidence} confidence`;
  $("#confidence-badge").className = `badge ${data.confidence !== "high" ? "warn" : ""}`;
  $("#answer").textContent = data.answer.replace(/\s*\[[^\]]+-chunk-\d+\]/g, "");

  if (data.resolved_patient_reference) {
    $("#patient-badge").textContent = `Patient · ${data.resolved_patient_reference}`;
    $("#patient-badge").classList.remove("hidden");
  }

  const retrieved = Object.fromEntries(data.trace.retrieved.map((item) => [item.chunk_id, item]));
  $("#citations").innerHTML = data.citations.length
    ? data.citations.map((citation) => {
        const source = retrieved[citation.chunk_id];
        const section = source?.section ? ` · ${escapeHtml(source.section)}` : "";
        return `<article class="source-card"><blockquote>“${escapeHtml(citation.quote)}”</blockquote><p>${escapeHtml(citation.document_id)}${section}</p></article>`;
      }).join("")
    : "<p>No chart sources supported an answer.</p>";

  $("#missing").innerHTML = data.missing_information.length
    ? data.missing_information.map((item) => `<li>${escapeHtml(item)}</li>`).join("")
    : "<li>Nothing identified for this question.</li>";
  $("#controls").innerHTML = [
    `<dt>Question type</dt><dd>${escapeHtml(data.intent.replaceAll("_", " "))}</dd>`,
    `<dt>Confidence</dt><dd>${escapeHtml(data.confidence_reason)}</dd>`,
    `<dt>Search mode</dt><dd>${escapeHtml(data.trace.retrieval_backend.replaceAll("_", " "))}</dd>`,
    `<dt>Human check</dt><dd>Required for any real-world use</dd>`,
  ].join("");
  if (data.tool_result) {
    const tool = data.tool_result;
    $("#tool-panel").classList.remove("hidden");
    $("#tool-panel").innerHTML = `<strong>ICD terminology lookup</strong><p>${tool.code ? `${escapeHtml(tool.code)} — ${escapeHtml(tool.description)}` : "No matching code was returned."}</p><p>${escapeHtml(tool.warning)}</p>`;
  }
  $("#trace").textContent = JSON.stringify(data.trace, null, 2);
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function addFiles(files) {
  const allowed = new Set(["pdf", "docx", "txt", "md"]);
  const additions = [...files].filter((file) => allowed.has(file.name.split(".").pop().toLowerCase()));
  additions.forEach((file) => {
    const key = `${file.name}:${file.size}:${file.lastModified}`;
    if (!selectedFiles.some((item) => `${item.name}:${item.size}:${item.lastModified}` === key)) selectedFiles.push(file);
  });
  renderFiles();
}

function renderFiles() {
  const list = $("#file-list");
  list.classList.toggle("hidden", selectedFiles.length === 0);
  list.innerHTML = selectedFiles.map((file, index) => `
    <div class="file-row">
      <div><strong>${escapeHtml(file.name)}</strong><small>${formatBytes(file.size)}</small></div>
      <button class="remove-file" type="button" data-remove-file="${index}" aria-label="Remove ${escapeHtml(file.name)}">Remove</button>
    </div>`).join("");
  $$('[data-remove-file]').forEach((button) => button.addEventListener("click", () => {
    selectedFiles.splice(Number(button.dataset.removeFile), 1);
    renderFiles();
  }));
}

$("#chart-files").addEventListener("change", (event) => {
  addFiles(event.target.files);
  event.target.value = "";
});
const dropzone = $("#dropzone");
["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => {
  event.preventDefault();
  dropzone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragging");
}));
dropzone.addEventListener("drop", (event) => addFiles(event.dataTransfer.files));

function setUploadBusy(busy) {
  $("#upload-submit").disabled = busy;
  $("#patient-reference").disabled = busy;
  $("#chart-files").disabled = busy;
  $("#synthetic-confirmed").disabled = busy;
}

function stageIndex(stage) {
  if (["queued"].includes(stage)) return 0;
  if (["validating", "parsing"].includes(stage)) return 1;
  if (["chunking", "checking_changes"].includes(stage)) return 2;
  if (["embedding", "updating_index", "saving"].includes(stage)) return 3;
  return 4;
}

function updateUploadProgress(percent, title, message, step, failed = false) {
  $("#upload-progress").classList.remove("hidden");
  $("#upload-percent").textContent = `${percent}%`;
  $("#upload-progress-bar").style.width = `${percent}%`;
  $("#upload-progress-title").textContent = title;
  $("#upload-message").textContent = message;
  $$('[data-upload-step]').forEach((item, index) => {
    item.classList.toggle("active", !failed && index === step);
    item.classList.toggle("complete", index < step || (!failed && step === 4 && index === 4));
    item.classList.toggle("failed", failed && index === step);
  });
}

function uploadRequest(formData) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/ingestion/jobs");
    xhr.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) {
        const uploadPercent = Math.max(1, Math.round((event.loaded / event.total) * 10));
        updateUploadProgress(uploadPercent, "Uploading files", "Sending selected chart files…", 0);
      }
    });
    xhr.addEventListener("load", () => {
      let payload = {};
      try { payload = JSON.parse(xhr.responseText); } catch { /* handled below */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(payload);
      else reject(new Error(payload.message || "The upload could not be started"));
    });
    xhr.addEventListener("error", () => reject(new Error("The upload connection failed")));
    xhr.send(formData);
  });
}

async function pollUploadJob(statusUrl) {
  while (true) {
    const response = await fetch(statusUrl);
    const job = await response.json();
    if (!response.ok) throw new Error(job.message || "Upload status is unavailable");
    const step = stageIndex(job.stage);
    const progress = job.status === "completed" ? 100 : Math.max(10, job.progress);
    updateUploadProgress(progress, step === 4 ? "Charts are ready" : "Processing patient charts", job.message, step, job.status === "failed");
    if (job.status === "completed") return job;
    if (job.status === "failed") throw new Error(job.error?.message || job.message);
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
}

$("#upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const patientReference = $("#patient-reference").value.trim();
  const confirmed = $("#synthetic-confirmed").checked;
  const error = $("#upload-error");
  error.classList.add("hidden");
  if (patientReference.length < 2 || selectedFiles.length === 0 || !confirmed) {
    error.textContent = patientReference.length < 2
      ? "Enter the patient ID, name, or chart number."
      : selectedFiles.length === 0
        ? "Choose at least one chart file."
        : "Confirm that the files contain synthetic demo data.";
    error.classList.remove("hidden");
    return;
  }
  const formData = new FormData();
  formData.append("patient_reference", patientReference);
  formData.append("synthetic_confirmed", "true");
  selectedFiles.forEach((file) => formData.append("files", file));
  setUploadBusy(true);
  $("#upload-submit").textContent = "Adding charts…";
  const started = performance.now();
  if (uploadInterval) clearInterval(uploadInterval);
  uploadInterval = setInterval(() => {
    $("#upload-elapsed").textContent = `${Math.floor((performance.now() - started) / 1000)}s elapsed`;
  }, 250);
  updateUploadProgress(1, "Uploading files", "Starting upload…", 0);
  try {
    const created = await uploadRequest(formData);
    const completed = await pollUploadJob(created.status_url);
    const result = completed.result;
    updateUploadProgress(100, "Charts are ready", `${result.documents_indexed} updated, ${result.documents_unchanged} unchanged · ${result.chunks_indexed} chart sections added`, 4);
    selectedFiles = [];
    renderFiles();
    $("#upload-form").reset();
    await refreshKnowledgeBase();
  } catch (uploadError) {
    const currentStep = $$('[data-upload-step]').findIndex((item) => item.classList.contains("active"));
    updateUploadProgress(Number($("#upload-percent").textContent.replace("%", "")) || 0, "Upload needs attention", uploadError.message, Math.max(0, currentStep), true);
    error.textContent = uploadError.message;
    error.classList.remove("hidden");
  } finally {
    if (uploadInterval) clearInterval(uploadInterval);
    uploadInterval = null;
    setUploadBusy(false);
    $("#upload-submit").textContent = "Add charts";
  }
});

refreshKnowledgeBase().catch(() => {});
checkSystemStatus();
