"use strict";

const form = document.querySelector("#detect-form");
const submitButton = document.querySelector("#submit-button");
const formError = document.querySelector("#form-error");
const warningBox = document.querySelector("#warning-box");
const textArea = document.querySelector("#user-text");
const textCount = document.querySelector("#text-count");
const imageInput = document.querySelector("#image-input");
const imagePreview = document.querySelector("#image-preview");
const previewWrap = document.querySelector("#preview-wrap");
const dropZone = document.querySelector("#drop-zone");

function scoreText(value) {
  return typeof value === "number" ? value.toFixed(2) : "--";
}

function setText(selector, value) {
  document.querySelector(selector).textContent = value;
}

function firstEvidence(items, source, fallback) {
  const item = (items || []).find((entry) => entry.source === source);
  return item ? item.description : fallback;
}

function showImage(file) {
  if (!file) {
    previewWrap.hidden = true;
    imagePreview.removeAttribute("src");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showError("图片不能超过 10 MB");
    imageInput.value = "";
    return;
  }
  imagePreview.src = URL.createObjectURL(file);
  previewWrap.hidden = false;
}

function showError(message) {
  formError.textContent = message;
  formError.hidden = !message;
}

function renderResult(report) {
  const fusion = report.fusion;
  const image = report.image_evidence;
  const text = report.text_evidence;
  const summary = document.querySelector("#result-summary");
  const labels = { risk: "风险", normal: "正常", insufficient_evidence: "证据不足" };
  summary.className = `result-summary ${fusion.label_name === "risk" ? "risk" : fusion.label_name === "insufficient_evidence" ? "insufficient" : "normal"}`;
  setText("#verdict", labels[fusion.label_name] || fusion.label_name);
  setText("#risk-score", scoreText(fusion.risk_score));
  setText("#threshold-label", `阈值 ${fusion.threshold.toFixed(2)}`);
  document.querySelector("#risk-meter").style.width = `${Math.round((fusion.risk_score || 0) * 100)}%`;
  setText("#score-user", scoreText(fusion.component_scores.user_text_score));
  setText("#score-ocr", scoreText(fusion.component_scores.ocr_text_score));
  setText("#score-visual", scoreText(fusion.component_scores.visual_score));
  setText("#score-cross", scoreText(fusion.component_scores.cross_modal_score));
  setText("#detail-user", firstEvidence(text.evidence, "user_text", report.request.user_text || "未提供用户文字"));
  setText("#detail-ocr", image.ocr_text || (image.ocr_status === "no_text" ? "图片中未识别到文字" : "OCR 文字不可用"));
  setText("#detail-visual", firstEvidence(image.evidence, "image", image.available ? "B 未提供图像证据描述" : "B 图像模块不可用"));
  const overlap = fusion.component_scores.text_overlap;
  setText("#detail-cross", fusion.component_scores.cross_modal_score == null ? "缺少计算跨模态关联所需的文字分数" : `字符关联度 ${scoreText(overlap)}；融合器已计算风险协同与可疑差异。`);
  warningBox.textContent = fusion.warnings.join("；");
  warningBox.hidden = fusion.warnings.length === 0;
  document.querySelector("#result-title").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();
    for (const name of ["B", "C", "D"]) {
      const row = document.querySelector(`[data-module="${name}"]`);
      const item = status[name];
      row.classList.add(item.available ? "available" : "unavailable");
      row.querySelector(".status-value").textContent = item.available ? "正常" : "未接入";
      row.querySelector("small").textContent = item.config_id || item.version;
    }
  } catch (error) {
    document.querySelectorAll(".module-state").forEach((row) => {
      row.classList.add("unavailable");
      row.querySelector(".status-value").textContent = "检查失败";
    });
  }
}

textArea.addEventListener("input", () => { textCount.textContent = String(textArea.value.length); });
imageInput.addEventListener("change", () => showImage(imageInput.files[0]));
for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.add("dragging"); });
}
for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.remove("dragging"); });
}
dropZone.addEventListener("drop", (event) => {
  const files = event.dataTransfer.files;
  if (files.length) {
    imageInput.files = files;
    showImage(files[0]);
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  showError("");
  warningBox.hidden = true;
  submitButton.disabled = true;
  submitButton.querySelector("span").textContent = "检测中";
  try {
    const response = await fetch("/api/detect", { method: "POST", body: new FormData(form) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "检测请求失败");
    renderResult(payload);
  } catch (error) {
    showError(error.message || String(error));
  } finally {
    submitButton.disabled = false;
    submitButton.querySelector("span").textContent = "开始检测";
  }
});

const recordsPanel = document.querySelector("#records-panel");
document.querySelector("#records-button").addEventListener("click", async () => {
  const response = await fetch("/api/records");
  const payload = await response.json();
  const list = document.querySelector("#records-list");
  list.replaceChildren();
  if (!payload.records.length) {
    const empty = document.createElement("p");
    empty.textContent = "尚无检测记录。";
    list.append(empty);
  }
  for (const record of payload.records) {
    const row = document.createElement("div");
    row.className = "record-row";
    const time = document.createElement("span");
    time.textContent = new Date(record.recorded_at_utc).toLocaleString("zh-CN");
    const sample = document.createElement("strong");
    sample.textContent = record.sample_id;
    const label = document.createElement("span");
    label.textContent = record.result.label_name;
    const score = document.createElement("span");
    score.textContent = scoreText(record.result.risk_score);
    row.append(time, sample, label, score);
    list.append(row);
  }
  recordsPanel.hidden = false;
});
document.querySelector("#records-close").addEventListener("click", () => { recordsPanel.hidden = true; });

textCount.textContent = String(textArea.value.length);
loadStatus();
