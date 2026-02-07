function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",").map(h => h.trim());
  const rows = [];

  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(",");
    const obj = {};
    headers.forEach((h, idx) => {
      obj[h] = (cols[idx] ?? "").trim();
    });
    rows.push(obj);
  }
  return { headers, rows };
}

function toNum(v) {
  const x = parseFloat(v);
  return Number.isFinite(x) ? x : 0;
}

function buildTable(el, headers, rows) {
  const showCols = [
    "region","zone","facility_type","predicted_demand_mw","priority_level","outage_risk",
    "decision_x","allocation_level","allocated_mw","unmet_mw","score"
  ].filter(c => headers.includes(c));

  const thead = `
    <thead>
      <tr>${showCols.map(h => `<th>${h}</th>`).join("")}</tr>
    </thead>
  `;

  const tbody = `
    <tbody>
      ${rows.map(r => `
        <tr>
          ${showCols.map(h => `<td>${r[h] ?? ""}</td>`).join("")}
        </tr>
      `).join("")}
    </tbody>
  `;

  el.innerHTML = thead + tbody;
}

function metricsFrom(rows) {
  const first = rows[0] || {};

  const supplyLimit =
    toNum(first["supply_mw_limit"]) ||
    toNum(first["supply_limit_mw"]) ||
    toNum(first["supply_mw"]) ||
    toNum(first["total_supply_mw"]) ||
    toNum(first["supply_limit"]) ||
    0;

  const used = rows.reduce((s, r) => s + toNum(r["allocated_mw"]), 0);
  const served = rows.filter(r => toNum(r["allocation_level"]) > 0).length;
  const totalScore = rows.reduce((s, r) => s + toNum(r["score"]) * toNum(r["allocation_level"]), 0);

  return { supplyLimit, used, served, totalScore };
}

async function loadOne(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load: ${path}`);
  const text = await res.text();
  return parseCSV(text);
}

function setText(id, v) {
  const el = document.getElementById(id);
  if (el) el.textContent = v; // safe if element was removed from HTML
}

async function loadDashboard(hour) {
  const qPath = `results/quantum_allocation_hour_${hour}.csv`;

  const q = await loadOne(qPath);

  // Sort like your screenshot: priority desc then score desc
  const qRows = [...q.rows].sort(
    (a, b) =>
      (toNum(b.priority_level) - toNum(a.priority_level)) ||
      (toNum(b.score) - toNum(a.score))
  );

  // Quantum table only
  const qTable = document.getElementById("qTable");
  if (qTable) buildTable(qTable, q.headers, qRows);

  // Quantum metrics only
  const qm = metricsFrom(qRows);
  const supply = qm.used; // since CSV has no supply limit
  setText("supplyLimit", supply ? supply.toFixed(2) : "—");
  

  //setText("supplyLimit", supply ? supply.toFixed(2) : "—");
  setText("qUsed", qm.used.toFixed(2));
  setText("qServed", `Facilities served: ${qm.served}`);
  setText("qScore", `Total score: ${qm.totalScore.toFixed(2)}`);
}

document.getElementById("loadBtn").addEventListener("click", async () => {
  const hour = document.getElementById("hourInput").value;
  try {
    await loadDashboard(hour);
  } catch (e) {
    alert(e.message + "\n\n.");
  }
});

// auto-load default
loadDashboard(document.getElementById("hourInput").value).catch(() => {});

// ===== CHAT TOGGLE =====
const chatToggle = document.getElementById("chatToggle");
const chatModal = document.getElementById("chatModal");
const closeChat = document.getElementById("closeChat");

chatToggle.onclick = () => {
  chatModal.classList.remove("hidden");
};

closeChat.onclick = () => {
  chatModal.classList.add("hidden");
};

// ===== CHAT LOGIC =====
const chatWindow = document.getElementById("chatWindow");
const chatInput = document.getElementById("chatInput");
const sendChat = document.getElementById("sendChat");
function addMessage(text, sender) {
  const div = document.createElement("div");
  div.className = `msg ${sender}`;
  div.textContent = text;
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

sendChat.onclick = sendQuestion;

chatInput.addEventListener("keydown", e => {
  if (e.key === "Enter") sendQuestion();
});

function sendQuestion() {
  const question = chatInput.value.trim();
  if (!question) return;

  addMessage(question, "user");
  chatInput.value = "";

  fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      hour: document.getElementById("hourInput").value,
      question: question
    })
  })
    .then(res => res.json())
    .then(data => {
      addMessage(data.answer, "bot");
    })
    .catch(() => {
      addMessage("Error connecting to backend.", "bot");
    });
}

// ===== SUGGESTED QUESTIONS =====
document.querySelectorAll(".suggested button").forEach(btn => {
  btn.addEventListener("click", () => {
    chatInput.value = btn.dataset.q;
    chatInput.focus();
  });
});