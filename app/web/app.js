const state = {
  selectedJobId: null,
  jobs: [],
};

const modeSelect = document.getElementById("modeSelect");
const statusPanel = document.getElementById("statusPanel");
const jobsPanel = document.getElementById("jobsPanel");
const logsPanel = document.getElementById("logsPanel");
const activeJob = document.getElementById("activeJob");
const runHint = document.getElementById("runHint");
const accentSelect = document.getElementById("accentSelect");

function toneByName(name) {
  if (name === "amber") {
    return { accent: "#ff9e2c", strong: "#ff8600" };
  }
  if (name === "tangerine") {
    return { accent: "#f97316", strong: "#ea580c" };
  }
  return { accent: "#ff8a3d", strong: "#ff7a1a" };
}

function applyAccent(name) {
  const tone = toneByName(name);
  document.documentElement.style.setProperty("--accent", tone.accent);
  document.documentElement.style.setProperty("--accent-strong", tone.strong);
}

accentSelect.addEventListener("change", () => applyAccent(accentSelect.value));

function parseNum(value, parseFn) {
  if (value === "" || value === null || value === undefined) {
    return null;
  }
  const parsed = parseFn(value);
  return Number.isNaN(parsed) ? null : parsed;
}

function collectTrainOptions() {
  return {
    resume: document.getElementById("trainResume").checked,
    epochs: parseNum(document.getElementById("trainEpochs").value, Number.parseInt),
    batch_size: parseNum(document.getElementById("trainBatch").value, Number.parseInt),
    lr: parseNum(document.getElementById("trainLr").value, Number.parseFloat),
  };
}

function collectGenerateOptions() {
  return {
    num_tokens: parseNum(document.getElementById("genTokens").value, Number.parseInt),
    temperature: parseNum(document.getElementById("genTemp").value, Number.parseFloat),
    top_k: parseNum(document.getElementById("genTopK").value, Number.parseInt),
    top_p: parseNum(document.getElementById("genTopP").value, Number.parseFloat),
    bpm: parseNum(document.getElementById("genBpm").value, Number.parseInt),
    seed_length: parseNum(document.getElementById("genSeedLen").value, Number.parseInt),
    seed_file: document.getElementById("genSeedFile").value.trim() || null,
  };
}

async function fetchJson(url, options = {}) {
  const res = await fetch(url, options);
  const body = await res.json();
  if (!res.ok) {
    throw new Error(body.detail || "Request failed");
  }
  return body;
}

function statusRow(label, value) {
  return `<div><dt>${label}</dt><dd>${value}</dd></div>`;
}

async function refreshStatus() {
  const mode = modeSelect.value;
  try {
    const st = await fetchJson(`/api/status?mode=${encodeURIComponent(mode)}`);
    statusPanel.innerHTML = [
      statusRow("MIDI files", st.midi_count),
      statusRow("Preprocessed", st.preprocessed ? "Yes" : "No"),
      statusRow("Model trained", st.trained ? "Yes" : "No"),
      statusRow("Processed", st.processed_path),
      statusRow("Vocab", st.vocab_path),
      statusRow("Checkpoint", st.checkpoint),
    ].join("");
  } catch (err) {
    statusPanel.innerHTML = statusRow("Error", err.message);
  }
}

function renderJobs() {
  if (state.jobs.length === 0) {
    jobsPanel.innerHTML = "<p class='hint'>No jobs yet.</p>";
    return;
  }

  jobsPanel.innerHTML = state.jobs
    .map((job) => {
      const activeClass = job.id === state.selectedJobId ? "active" : "";
      return `
        <article class="job-item ${activeClass}" data-id="${job.id}">
          <div><strong>${job.action}</strong> <small>(${job.mode})</small></div>
          <span class="badge ${job.status}">${job.status}</span>
          <small>Created: ${job.created_at}</small>
        </article>
      `;
    })
    .join("");

  jobsPanel.querySelectorAll(".job-item").forEach((el) => {
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
    runHint.textContent = `Failed to fetch jobs: ${err.message}`;
  }
}

async function refreshLogs() {
  if (!state.selectedJobId) {
    activeJob.textContent = "Select a job to stream logs.";
    logsPanel.textContent = "No logs yet.";
    return;
  }

  try {
    const [job, logs] = await Promise.all([
      fetchJson(`/api/jobs/${state.selectedJobId}`),
      fetchJson(`/api/jobs/${state.selectedJobId}/logs?tail=500`),
    ]);
    activeJob.textContent = `Job ${job.id} | ${job.action} | ${job.status}`;
    logsPanel.textContent = logs.lines.join("");
    logsPanel.scrollTop = logsPanel.scrollHeight;
  } catch (err) {
    logsPanel.textContent = `Error loading logs: ${err.message}`;
  }
}

async function createJob(action) {
  const body = {
    mode: modeSelect.value,
    action,
    train: collectTrainOptions(),
    generate: collectGenerateOptions(),
  };

  try {
    const created = await fetchJson("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    runHint.textContent = `Started ${created.action} job (${created.id}).`;
    state.selectedJobId = created.id;
    await refreshJobs();
    await refreshLogs();
  } catch (err) {
    runHint.textContent = `Could not start job: ${err.message}`;
  }
}

document.querySelectorAll("button[data-action]").forEach((btn) => {
  btn.addEventListener("click", () => createJob(btn.dataset.action));
});

document.getElementById("refreshStatusBtn").addEventListener("click", refreshStatus);
modeSelect.addEventListener("change", refreshStatus);

async function bootstrap() {
  applyAccent("sunset");
  await refreshStatus();
  await refreshJobs();
  await refreshLogs();

  setInterval(refreshStatus, 7000);
  setInterval(refreshJobs, 3500);
  setInterval(refreshLogs, 1800);
}

bootstrap();
