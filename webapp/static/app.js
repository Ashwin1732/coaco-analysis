const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const previewWrap = document.getElementById("previewWrap");
const preview = document.getElementById("preview");
const results = document.getElementById("results");
const statusEl = document.getElementById("status");

function fmt(n, digits = 4) {
  if (!Number.isFinite(n)) return "—";
  if (Math.abs(n) >= 100) return n.toFixed(1);
  if (Math.abs(n) >= 1) return n.toFixed(3);
  return n.toFixed(digits);
}

function setStatus(msg, isError = false) {
  statusEl.textContent = msg || "";
  statusEl.classList.toggle("error", Boolean(isError));
}

function showPreview(file) {
  if (file.type.startsWith("image/")) {
    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.classList.remove("hidden");
    if(document.getElementById("csvPreview")) document.getElementById("csvPreview").classList.add("hidden");
  } else {
    preview.classList.add("hidden");
    let csvPrev = document.getElementById("csvPreview");
    if (!csvPrev) {
        csvPrev = document.createElement("div");
        csvPrev.id = "csvPreview";
        Object.assign(csvPrev.style, {
            position: "absolute", inset: "0", display: "flex", flexDirection: "column",
            justifyContent: "center", alignItems: "center", background: "var(--bg2)", zIndex: "0"
        });
        csvPrev.innerHTML = `<div style="font-size:3rem">📊</div><div id="csvName" style="margin-top:1rem;color:var(--ink);word-break:break-all;padding:0 2rem;text-align:center;"></div>`;
        previewWrap.appendChild(csvPrev);
    }
    csvPrev.classList.remove("hidden");
    document.getElementById("csvName").textContent = file.name;
  }
  previewWrap.classList.remove("hidden");
  dropzone.classList.add("has-image");
}

function renderResult(data) {
  document.getElementById("qualityCode").textContent = data.quality;
  document.getElementById("qualityLabel").textContent = data.quality_label;
  document.getElementById("confidence").textContent =
    `${(data.confidence * 100).toFixed(1)}%`;

  document.getElementById("amplitude").textContent = fmt(data.amplitude.value);
  document.getElementById("amplitudeUnit").textContent =
    data.amplitude.unit || "RMS";
  document.getElementById("frequency").textContent = fmt(data.frequency.value, 1);
  document.getElementById("frequencyUnit").textContent =
    data.frequency.unit || "Hz";
  document.getElementById("power").textContent = fmt(data.power.value);
  document.getElementById("powerUnit").textContent =
    data.power.unit || "mean-square";

  const probs = document.getElementById("probs");
  probs.innerHTML = "";
  Object.entries(data.class_probabilities || {}).forEach(([cls, p]) => {
    const div = document.createElement("div");
    div.className = "prob";
    div.innerHTML = `<div><strong>${cls}</strong> ${(p * 100).toFixed(1)}%</div>
      <div class="bar"><span style="width:${(p * 100).toFixed(1)}%"></span></div>`;
    probs.appendChild(div);
  });

  results.classList.remove("hidden");
}

async function predict(file) {
  showPreview(file);
  results.classList.add("hidden"); // Hide previous results while loading
  dropzone.classList.add("loading");
  setStatus(`Analyzing ${file.name.endsWith('.csv') ? 'waveform' : 'thermal image'}…`);

  const body = new FormData();
  body.append("file", file);

  try {
    const res = await fetch("/api/predict", { method: "POST", body });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Prediction failed");
    }
    renderResult(data);
    setStatus(`Done · ${data.filename || file.name}`);
  } catch (err) {
    setStatus(err.message || String(err), true);
  } finally {
    dropzone.classList.remove("loading");
  }
}

function onFiles(files) {
  const file = files && files[0];
  if (!file) return;
  const isImage = file.type.startsWith("image/");
  const isCsv = file.name.toLowerCase().endsWith(".csv") || file.type.includes("csv");
  if (!isImage && !isCsv) {
    setStatus("Please upload an image or CSV file.", true);
    results.classList.remove("hidden");
    return;
  }
  predict(file);
}

browseBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener("change", () => onFiles(fileInput.files));

["dragenter", "dragover"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (e) => onFiles(e.dataTransfer.files));
