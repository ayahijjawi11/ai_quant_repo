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
  const supplyLimit = rows.length ? toNum(rows[0]["supply_mw_limit"]) : 0;
  const used = rows.reduce((s, r) => s + toNum(r["allocated_mw"]), 0);
  const served = rows.filter(r => toNum(r["allocation_level"]) > 0).length;

  // total score = sum(score * allocation_level) 
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
  document.getElementById(id).textContent = v;
}

async function loadDashboard(hour) {
  const qPath = `results/quantum_allocation_hour_${hour}.csv`;
  const cPath = `results/classical_greedy_hour_${hour}.csv`;

  const q = await loadOne(qPath);
  const c = await loadOne(cPath);

  // Sort like your screenshot: priority desc then score desc
  const qRows = [...q.rows].sort((a,b) => (toNum(b.priority_level)-toNum(a.priority_level)) || (toNum(b.score)-toNum(a.score)));
  const cRows = [...c.rows].sort((a,b) => (toNum(b.priority_level)-toNum(a.priority_level)) || (toNum(b.score)-toNum(a.score)));

  buildTable(document.getElementById("qTable"), q.headers, qRows);
  buildTable(document.getElementById("cTable"), c.headers, cRows);

  const qm = metricsFrom(qRows);
  const cm = metricsFrom(cRows);

  const supply = qm.supplyLimit || cm.supplyLimit;

  setText("supplyLimit", supply ? supply.toFixed(2) : "—");
  setText("qUsed", qm.used.toFixed(2));
  setText("qServed", `Facilities served: ${qm.served}`);
  setText("qScore", `Total score: ${qm.totalScore.toFixed(2)}`);

  setText("cUsed", cm.used.toFixed(2));
  setText("cServed", `Facilities served: ${cm.served}`);
  setText("cScore", `Total score: ${cm.totalScore.toFixed(2)}`);

  // Winner decision
  let winner = "Quantum";
  let reason = "Higher total score";
  if (cm.totalScore > qm.totalScore + 1e-9) {
    winner = "Classic";
    reason = "Higher total score";
  } else if (Math.abs(cm.totalScore - qm.totalScore) <= 1e-9) {
    // tie-breaker by served
    if (cm.served > qm.served) {
      winner = "Classic";
      reason = "Tie score, more facilities served";
    } else if (qm.served > cm.served) {
      winner = "Quantum";
      reason = "Tie score, more facilities served";
    } else {
      winner = "Tie";
      reason = "Same score & served count";
    }
  }

  setText("winner", winner);
  setText("winnerReason", reason);
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
loadDashboard(document.getElementById("hourInput").value).catch(()=>{});
