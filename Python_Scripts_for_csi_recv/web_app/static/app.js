"use strict";

const $ = (id) => document.getElementById(id);
let eventSocket;
let currentSessionId = null;
let selectedSessionId = null;
let sessionRefreshTimer;
let detailRefreshTimer;

async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed (${response.status})`);
  }
  return response.status === 204 ? null : response.json();
}

function showDashboard() {
  $("login").classList.add("hidden");
  $("dashboard").classList.remove("hidden");
  connectEvents();
  loadHistory().catch(showPageError);
  loadSessions().catch(showPageError);
}

function showPageError(error) {
  $("device-state").textContent = error.message;
}

function connectEvents() {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  eventSocket = new WebSocket(`${scheme}//${location.host}/api/events`);
  eventSocket.onmessage = ({ data }) => {
    const event = JSON.parse(data);
    if (event.status) renderStatus(event.status);
    if (event.type === "settings_changed") {
      scheduleSessionRefresh();
      if (selectedSessionId !== null) scheduleDetailRefresh();
    }
    if (event.type === "session_closed") {
      scheduleSessionRefresh();
      if (selectedSessionId === event.session_id) scheduleDetailRefresh();
    }
    if (event.status?.session_id !== currentSessionId) {
      currentSessionId = event.status?.session_id ?? null;
      scheduleSessionRefresh();
    }
    for (const prediction of event.predictions || []) {
      addPrediction(prediction, true);
      scheduleSessionRefresh();
      if (selectedSessionId === prediction.session_id) scheduleDetailRefresh();
    }
  };
  eventSocket.onclose = (event) => {
    if (event.code === 4401) {
      $("dashboard").classList.add("hidden");
      $("login").classList.remove("hidden");
      return;
    }
    $("device-state").textContent = "Live feed disconnected";
    $("device-dot").parentElement.className = "pill offline";
    if (!$("dashboard").classList.contains("hidden")) setTimeout(connectEvents, 2000);
  };
  eventSocket.onerror = () => eventSocket.close();
}

function renderStatus(status) {
  $("model").textContent = (status.model || "—").toUpperCase();
  $("mode").textContent = status.mode === "motion" ? "Super Motion" : "HAR";
  $("frames").textContent = Number(status.frames_received || 0).toLocaleString();
  $("fps").textContent = `${Number(status.fps || 0).toFixed(1)} Hz`;
  $("buffer").textContent = `${status.buffered_frames || 0} / ${status.buffer_capacity || 0}`;
  const settings = status.settings || {};
  $("inference-settings").textContent = [
    `Window ${settings.window_size ?? status.window_size ?? "—"}`,
    `step ${settings.step_size ?? "—"}`,
    `warm-up ${settings.warmup_frames ?? "—"}`,
    `cutoff ${settings.cutoff ?? "—"} Hz`,
    `EMA ${settings.ema_alpha ?? "—"}`,
    `confidence threshold ${settings.conf_thresh ?? "—"}%`,
    `hysteresis ${settings.hyst_count ?? "—"}`,
    `motion trigger ${settings.motion_threshold ?? "—"}`,
  ].join(" · ");
  $("device-state").textContent = status.device_connected ? "Device online" : "Waiting for device";
  $("device-dot").parentElement.className = `pill ${status.device_connected ? "online" : "offline"}`;
  if (status.latest_prediction) {
    renderPrediction(status.latest_prediction);
  } else {
    $("activity").textContent = "Waiting for data";
    $("confidence").textContent = "Waiting for a prediction from the selected model.";
    $("probabilities").replaceChildren();
  }
}

function renderPrediction(prediction) {
  $("activity").textContent = prediction.label;
  $("confidence").textContent = `${Number(prediction.confidence).toFixed(1)}% confidence`;
  const bars = $("probabilities");
  bars.replaceChildren();
  for (const [label, probability] of Object.entries(prediction.probabilities || {})) {
    const row = document.createElement("div");
    row.className = "bar-row";
    const name = document.createElement("span");
    name.textContent = label;
    const progress = document.createElement("progress");
    progress.className = "bar-progress";
    progress.max = 100;
    progress.value = Math.max(0, Math.min(100, Number(probability) || 0));
    const percentage = document.createElement("span");
    percentage.textContent = `${Number(probability).toFixed(0)}%`;
    row.append(name, progress, percentage);
    bars.append(row);
  }
}

function addPrediction(prediction, prepend = false) {
  renderPrediction(prediction);
  const row = document.createElement("tr");
  const values = [
    new Date(prediction.created_at).toLocaleTimeString(),
    String(prediction.label),
    String(prediction.model || "—").toUpperCase(),
    `${Number(prediction.confidence).toFixed(1)}%`,
    Number(prediction.frame_count).toLocaleString(),
  ];
  for (const value of values) {
    const cell = document.createElement("td");
    cell.textContent = value;
    row.append(cell);
  }
  if (prepend) $("history").prepend(row);
  else $("history").append(row);
  while ($("history").children.length > 50) $("history").lastElementChild.remove();
  $("history-table").classList.remove("hidden");
  $("empty-history").classList.add("hidden");
}

async function loadHistory() {
  const rows = await api("/api/predictions?limit=50");
  $("history").replaceChildren();
  $("history-table").classList.toggle("hidden", rows.length === 0);
  $("empty-history").classList.toggle("hidden", rows.length > 0);
  rows.reverse().forEach((prediction) => addPrediction(prediction));
}

function scheduleSessionRefresh() {
  clearTimeout(sessionRefreshTimer);
  sessionRefreshTimer = setTimeout(() => loadSessions().catch(showPageError), 1500);
}

async function loadSessions() {
  const sessions = await api("/api/activity-sessions?limit=100");
  const body = $("sessions");
  body.replaceChildren();
  for (const session of sessions) {
    const row = document.createElement("tr");
    const values = [
      new Date(session.started_at).toLocaleString(),
      formatDuration(session.duration_seconds),
      Number(session.total_frames).toLocaleString(),
      Number(session.prediction_count).toLocaleString(),
      session.active ? "Active" : "Ended",
    ];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    const action = document.createElement("td");
    const open = document.createElement("button");
    open.type = "button";
    open.className = "secondary small-button";
    open.textContent = "Details";
    open.addEventListener("click", () => openSession(session.id));
    action.append(open);
    row.append(action);
    body.append(row);
  }
  const hasSessions = sessions.length > 0;
  $("sessions-table").classList.toggle("hidden", !hasSessions);
  $("empty-sessions").classList.toggle("hidden", hasSessions);
}

function formatDuration(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const remainder = total % 60;
  return hours ? `${hours}h ${minutes}m` : minutes ? `${minutes}m ${remainder}s` : `${remainder}s`;
}

function appendCells(row, values) {
  for (const value of values) {
    const cell = document.createElement("td");
    cell.textContent = String(value);
    row.append(cell);
  }
}

async function openSession(sessionId) {
  selectedSessionId = Number(sessionId);
  $("session-detail").classList.remove("hidden");
  $("session-title").textContent = `Session #${selectedSessionId}`;
  $("session-period").textContent = "Loading session details…";
  $("session-list-panel").classList.add("hidden");
  try {
    await refreshSessionDetail();
  } catch (error) {
    $("session-period").textContent = error.message;
  }
}

async function refreshSessionDetail() {
  if (selectedSessionId === null) return;
  const session = await api(`/api/activity-sessions/${selectedSessionId}`);
  $("session-title").textContent = `Session #${session.id}${session.active ? " · Active" : ""}`;
  $("session-period").textContent =
    `${new Date(session.started_at).toLocaleString()} – ` +
    `${session.active ? "in progress" : new Date(session.ended_at).toLocaleString()}`;
  const metrics = [
    ["Duration", formatDuration(session.duration_seconds)],
    ["CSI frames", Number(session.total_frames).toLocaleString()],
    ["Predictions", session.statistics.reduce((sum, item) => sum + item.prediction_count, 0).toLocaleString()],
  ];
  const metricContainer = $("session-metrics");
  metricContainer.replaceChildren();
  for (const [label, value] of metrics) {
    const card = document.createElement("article");
    card.className = "detail-metric";
    const name = document.createElement("span");
    name.className = "label";
    name.textContent = label;
    const count = document.createElement("strong");
    count.textContent = value;
    card.append(name, count);
    metricContainer.append(card);
  }

  const statistics = [...session.statistics];
  for (const label of ["walk_activity", "no_activity"]) {
    if (!statistics.some((item) => item.label === label)) {
      statistics.push({
        label,
        model: "—",
        prediction_count: 0,
        average_confidence: 0,
        average_probability: 0,
      });
    }
  }
  const statisticsBody = $("statistics");
  statisticsBody.replaceChildren();
  for (const stat of statistics) {
    const row = document.createElement("tr");
    appendCells(row, [
      stat.label,
      stat.model === "—" ? stat.model : stat.model.toUpperCase(),
      Number(stat.prediction_count).toLocaleString(),
      `${Number(stat.average_confidence).toFixed(1)}%`,
      `${Number(stat.average_probability).toFixed(1)}%`,
    ]);
    statisticsBody.append(row);
  }
  const hasStatistics = session.statistics.length > 0;
  $("statistics-table").classList.toggle("hidden", !hasStatistics);
  $("empty-statistics").classList.toggle("hidden", hasStatistics);

  const changesBody = $("settings-changes");
  changesBody.replaceChildren();
  for (const change of session.settings_changes) {
    const row = document.createElement("tr");
    appendCells(row, [
      new Date(change.changed_at).toLocaleTimeString(),
      change.setting,
      change.old_value.toUpperCase(),
      change.new_value.toUpperCase(),
    ]);
    changesBody.append(row);
  }
  const hasChanges = session.settings_changes.length > 0;
  $("settings-table").classList.toggle("hidden", !hasChanges);
  $("empty-settings").classList.toggle("hidden", hasChanges);

  const timeline = $("timeline");
  timeline.replaceChildren();
  const predictions = [...session.predictions].reverse();
  for (const prediction of predictions) {
    const row = document.createElement("tr");
    appendCells(row, [
      new Date(prediction.created_at).toLocaleString(),
      prediction.label,
      prediction.model.toUpperCase(),
      `${Number(prediction.confidence).toFixed(1)}%`,
      Number(prediction.frame_count).toLocaleString(),
    ]);
    timeline.append(row);
  }
  const hasPredictions = predictions.length > 0;
  $("timeline-table").classList.toggle("hidden", !hasPredictions);
  $("empty-timeline").classList.toggle("hidden", hasPredictions);
}

function scheduleDetailRefresh() {
  clearTimeout(detailRefreshTimer);
  detailRefreshTimer = setTimeout(() => {
    refreshSessionDetail().catch((error) => {
      $("session-period").textContent = error.message;
    });
  }, 1500);
}

$("sessions-back").addEventListener("click", () => {
  selectedSessionId = null;
  $("session-detail").classList.add("hidden");
  $("session-list-panel").classList.remove("hidden");
});

$("login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("login-error").textContent = "";
  try {
    await api("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ username: $("username").value, password: $("password").value }),
    });
    $("password").value = "";
    showDashboard();
  } catch (error) {
    $("login-error").textContent = error.message;
  }
});

$("logout").addEventListener("click", async () => {
  if (eventSocket) eventSocket.close();
  try {
    await api("/api/auth/logout", { method: "POST" });
    $("dashboard").classList.add("hidden");
    $("login").classList.remove("hidden");
  } catch (error) {
    $("device-state").textContent = error.message;
  }
});

api("/api/auth/me").then(showDashboard).catch(() => {});
