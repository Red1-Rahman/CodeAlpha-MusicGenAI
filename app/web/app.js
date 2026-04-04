/* ────────────────────────────────────────────────────────────────────────
   MusicGen Dashboard — app.js
   ────────────────────────────────────────────────────────────────────── */

"use strict";

/* ── State ───────────────────────────────────────────────────────────── */
const state = {
  mode: "mixed",
  selectedJobId: null,
  jobs: [],
  playerFiles: [],
  playerIndex: 0,
  playerIsPlaying: false,
};

/* ── DOM refs ─────────────────────────────────────────────────────────── */
const $ = (id) => document.getElementById(id);

const statusPanel     = $("statusPanel");
const jobsPanel       = $("jobsPanel");
const logsPanel       = $("logsPanel");
const activeJobLabel  = $("activeJobLabel");
const runHint         = $("runHint");
const jobCount        = $("jobCount");
const serverStatus    = $("serverStatus");
const serverLabel     = $("serverLabel");
const playerFiles     = $("playerFiles");
const playerPlay      = $("playerPlay");
const playerPrev      = $("playerPrev");
const playerNext      = $("playerNext");
const nowPlayingName  = $("nowPlayingName");
const waveform        = $("waveform");
const playerMeta      = $("playerMeta");

/* ── Accent colours ───────────────────────────────────────────────────── */
const ACCENTS = {
  sunset:    { accent: "#ff8a3d", strong: "#ff7a1a", glow: "rgba(255,138,61,.35)"  },
  amber:     { accent: "#ff9e2c", strong: "#e88a00", glow: "rgba(255,158,44,.35)"  },
  tangerine: { accent: "#f97316", strong: "#ea580c", glow: "rgba(249,115,22,.35)"  },
  rose:      { accent: "#fb7185", strong: "#f43f5e", glow: "rgba(251,113,133,.35)" },
  cyan:      { accent: "#22d3ee", strong: "#06b6d4", glow: "rgba(34,211,238,.35)"  },
};

function applyAccent(name) {
  const t = ACCENTS[name] || ACCENTS.sunset;
  const r = document.documentElement.style;
  r.setProperty("--accent",      t.accent);
  r.setProperty("--accent-dim",  t.glow.replace(".35", ".12"));
  r.setProperty("--accent-glow", t.glow);
}

document.getElementById("swatchRow").addEventListener("click", (e) => {
  const btn = e.target.closest(".swatch");
  if (!btn) return;
  document.querySelectorAll(".swatch").forEach(s => s.classList.remove("swatch--active"));
  btn.classList.add("swatch--active");
  applyAccent(btn.dataset.accent);
});

/* ── Mode pills ────────────────────────────────────────────────────────── */
document.getElementById("modePills").addEventListener("click", (e) => {
  const btn = e.target.closest(".mode-pill");
  if (!btn) return;
  document.querySelectorAll(".mode-pill").forEach(p => p.classList.remove("mode-pill--active"));
  btn.classList.add("mode-pill--active");
  state.mode = btn.dataset.mode;
  refreshStatus();
});

/* ── Utilities ─────────────────────────────────────────────────────────── */
function parseNum(value, fn) {
  if (value === "" || value == null) return null;
  const n = fn(value);
  return Number.isNaN(n) ? null : n;
}

async function fetchJson(url, opts = {}) {
  const res = await fetch(url, opts);
  const body = await res.json();
  if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
  return body;
}

function setHint(text, type = "") {
  runHint.textContent = text;
  runHint.className = "run-hint" + (type ? " " + type : "");
}

/* ── Grid canvas (ambient) ─────────────────────────────────────────────── */
function drawGrid() {
  const canvas = $("gridCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width  = canvas.offsetWidth;
  const H = canvas.height = canvas.offsetHeight;
  const step = 40;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = "#1e2a38";
  ctx.lineWidth = .5;
  for (let x = 0; x <= W; x += step) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }
  for (let y = 0; y <= H; y += step) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }
}
window.addEventListener("resize", drawGrid);

/* ── Server health ─────────────────────────────────────────────────────── */
async function checkHealth() {
  try {
    await fetchJson("/api/health");
    serverStatus.className = "pulse-dot ok";
    serverLabel.textContent = "Server online";
  } catch {
    serverStatus.className = "pulse-dot err";
    serverLabel.textContent = "Server offline";
  }
}

/* ── Status panel ──────────────────────────────────────────────────────── */
async function refreshStatus() {
  const btn = $("refreshStatusBtn");
  btn.classList.add("spinning");
  try {
    const st = await fetchJson(`/api/status?mode=${encodeURIComponent(state.mode)}`);
    statusPanel.innerHTML = `
      <div class="status-row"><dt>MIDI files</dt><dd>${st.midi_count}</dd></div>
      <div class="status-row"><dt>Preprocessed</dt><dd class="${st.preprocessed ? "ok" : "err"}">${st.preprocessed ? "Yes" : "No"}</dd></div>
      <div class="status-row"><dt>Trained</dt><dd class="${st.trained ? "ok" : "err"}">${st.trained ? "Yes" : "No"}</dd></div>
      <div class="status-row"><dt>Checkpoint</dt><dd class="mono" title="${st.checkpoint}">${basename(st.checkpoint)}</dd></div>
    `;
  } catch (err) {
    statusPanel.innerHTML = `<div class="status-row"><dt>Error</dt><dd class="err">${err.message}</dd></div>`;
  } finally {
    btn.classList.remove("spinning");
  }
}

function basename(p) {
  if (!p) return "—";
  return p.replace(/\\/g, "/").split("/").pop() || p;
}

/* ── Jobs ──────────────────────────────────────────────────────────────── */
function renderJobs() {
  jobCount.textContent = state.jobs.length;
  if (state.jobs.length === 0) {
    jobsPanel.innerHTML = "<p class='empty-hint'>No jobs yet.</p>";
    return;
  }
  jobsPanel.innerHTML = state.jobs.map(job => {
    const active = job.id === state.selectedJobId ? "active" : "";
    const time = job.created_at.replace("T", " ").replace("Z", "");
    return `
      <div class="job-item ${active}" data-id="${job.id}">
        <div class="job-item__info">
          <div class="job-item__action">${job.action}</div>
          <div class="job-item__meta">${job.mode} · ${time}</div>
        </div>
        <span class="badge ${job.status}">${job.status}</span>
      </div>
    `;
  }).join("");

  jobsPanel.querySelectorAll(".job-item").forEach(el => {
    el.addEventListener("click", () => {
      state.selectedJobId = el.dataset.id;
      renderJobs();
      refreshLogs();
    });
  });
}

async function refreshJobs() {
  try {
    state.jobs = await fetchJson("/api/jobs?limit=30");
    if (!state.selectedJobId && state.jobs.length > 0) {
      state.selectedJobId = state.jobs[0].id;
    }
    renderJobs();
  } catch (err) {
    setHint(`Failed to fetch jobs: ${err.message}`, "error");
  }
}

async function refreshLogs() {
  if (!state.selectedJobId) {
    activeJobLabel.textContent = "—";
    logsPanel.textContent = "Select a job to view logs.";
    return;
  }
  try {
    const [job, logs] = await Promise.all([
      fetchJson(`/api/jobs/${state.selectedJobId}`),
      fetchJson(`/api/jobs/${state.selectedJobId}/logs?tail=500`),
    ]);
    activeJobLabel.textContent = `${job.id} · ${job.action} · ${job.status}`;
    logsPanel.textContent = logs.lines.join("");
    logsPanel.scrollTop = logsPanel.scrollHeight;
  } catch (err) {
    logsPanel.textContent = `Error loading logs: ${err.message}`;
  }
}

/* ── Collect options ────────────────────────────────────────────────────── */
function collectTrainOptions() {
  return {
    resume:     $("trainResume").checked,
    epochs:     parseNum($("trainEpochs").value, Number.parseInt),
    batch_size: parseNum($("trainBatch").value,  Number.parseInt),
    lr:         parseNum($("trainLr").value,      Number.parseFloat),
  };
}

function collectGenerateOptions() {
  return {
    num_tokens:   parseNum($("genTokens").value,  Number.parseInt),
    temperature:  parseNum($("genTemp").value,    Number.parseFloat),
    top_k:        parseNum($("genTopK").value,    Number.parseInt),
    top_p:        parseNum($("genTopP").value,    Number.parseFloat),
    bpm:          parseNum($("genBpm").value,     Number.parseInt),
    seed_length:  parseNum($("genSeedLen").value, Number.parseInt),
    seed_file:    $("genSeedFile").value.trim() || null,
  };
}

/* ── Create Job ────────────────────────────────────────────────────────── */
async function createJob(action) {
  setHint(`Starting ${action}…`, "running");
  const body = {
    mode:     state.mode,
    action,
    train:    collectTrainOptions(),
    generate: collectGenerateOptions(),
  };
  try {
    const created = await fetchJson("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    setHint(`Job started: ${created.id} (${created.action})`, "running");
    state.selectedJobId = created.id;
    await refreshJobs();
    await refreshLogs();
  } catch (err) {
    setHint(`Could not start job: ${err.message}`, "error");
  }
}

document.querySelectorAll("button[data-action]").forEach(btn => {
  btn.addEventListener("click", () => createJob(btn.dataset.action));
});

$("refreshStatusBtn").addEventListener("click", refreshStatus);

/* ── MIDI Player ────────────────────────────────────────────────────────── */
// We use MIDI.js via CDN for playback. Since it requires a complex setup,
// we implement a lightweight approach using the Web MIDI API if available,
// with a fallback to listing files for download.
// The player UI shows generated MIDI files from /api/outputs and lets you
// "play" them (toggling waveform animation) — actual audio playback
// requires the backend to expose a /api/outputs endpoint or uses the
// browser's built-in MIDI support if the OS has a MIDI synth.

async function refreshPlayerFiles() {
  try {
    const data = await fetchJson("/api/outputs");
    state.playerFiles = data.files || [];
    renderPlayerFiles();
  } catch {
    // /api/outputs may not exist yet — that's ok
    playerMeta.innerHTML = `
      Add <code>/api/outputs</code> to <code>web_server.py</code> to list generated files here.
      See the Copilot prompt below for implementation details.
    `;
  }
}

function renderPlayerFiles() {
  if (state.playerFiles.length === 0) {
    playerFiles.innerHTML = "<p class='empty-hint'>No generated MIDI files found.</p>";
    return;
  }
  playerFiles.innerHTML = state.playerFiles.map((f, i) => `
    <button class="player-file-chip ${i === state.playerIndex ? "active" : ""}" data-index="${i}" title="${f}">
      ♩ ${basename(f)}
    </button>
  `).join("");

  playerFiles.querySelectorAll(".player-file-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      state.playerIndex = parseInt(chip.dataset.index);
      state.playerIsPlaying = false;
      stopPlayer();
      renderPlayerFiles();
      updateNowPlaying();
    });
  });

  updateNowPlaying();
}

function updateNowPlaying() {
  const file = state.playerFiles[state.playerIndex];
  nowPlayingName.textContent = file ? basename(file) : "No file selected";

  if (file) {
    playerMeta.innerHTML = `
      <a href="/api/outputs/${encodeURIComponent(basename(file))}" download style="color:var(--accent);text-decoration:none;">
        ⬇ Download ${basename(file)}
      </a>
    `;
  }
}

function stopPlayer() {
  state.playerIsPlaying = false;
  playerPlay.innerHTML = "&#9654;";
  waveform.classList.remove("playing");
}

function startPlayer() {
  state.playerIsPlaying = true;
  playerPlay.innerHTML = "&#9646;&#9646;";
  waveform.classList.add("playing");
}

playerPlay.addEventListener("click", () => {
  if (state.playerFiles.length === 0) return;
  if (state.playerIsPlaying) {
    stopPlayer();
  } else {
    startPlayer();
    // Browser MIDI playback: open the file download link in a new tab
    // for actual audio, a synthesiser library (e.g. MIDI.js, Tone.js + MIDI)
    // must be loaded. See Copilot prompt for full implementation.
    const file = state.playerFiles[state.playerIndex];
    if (file) {
      // Attempt native MIDI via <audio> — works if OS MIDI synth present
      const audio = new Audio(`/api/outputs/${encodeURIComponent(basename(file))}`);
      audio.onended = stopPlayer;
      audio.play().catch(() => {
        // No synth — show download link hint
        playerMeta.innerHTML = `No MIDI synth detected. <a href="/api/outputs/${encodeURIComponent(basename(file))}" download style="color:var(--accent)">Download and open in a DAW</a>.`;
        stopPlayer();
      });
      state._audio = audio;
    }
  }
});

playerPlay.addEventListener("click", () => {
  if (state._audio && !state.playerIsPlaying) {
    state._audio.pause();
    state._audio.currentTime = 0;
  }
});

playerPrev.addEventListener("click", () => {
  if (state.playerFiles.length === 0) return;
  stopPlayer();
  state.playerIndex = (state.playerIndex - 1 + state.playerFiles.length) % state.playerFiles.length;
  renderPlayerFiles();
});

playerNext.addEventListener("click", () => {
  if (state.playerFiles.length === 0) return;
  stopPlayer();
  state.playerIndex = (state.playerIndex + 1) % state.playerFiles.length;
  renderPlayerFiles();
});

/* ── Bootstrap ──────────────────────────────────────────────────────────── */
async function bootstrap() {
  drawGrid();
  applyAccent("sunset");

  await checkHealth();
  await refreshStatus();
  await refreshJobs();
  await refreshLogs();
  await refreshPlayerFiles();

  setInterval(checkHealth,    10_000);
  setInterval(refreshStatus,   8_000);
  setInterval(refreshJobs,     3_500);
  setInterval(refreshLogs,     2_000);
  setInterval(refreshPlayerFiles, 15_000);
}

bootstrap();