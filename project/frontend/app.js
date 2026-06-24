/* =====================================================================
   Smart Career Navigator — minimal single-chat UI
     source "dataset" -> /api/chat           (gold US dataset, 119K)
     source "scrape"  -> /api/scrape-search   (live LinkedIn / Israel)
   The live-scan field editor lives in a drawer (gear button).

   CV PERSONALIZATION ADDITIONS:
     cvState — module-level CV state (cv_text, profile, history, roles)
     Feature 1: CV upload via composer "+" button, conversational greeting (POST /api/cv)
     Feature 2: action-driven rendering — history+roles sent per turn
     Feature 3: per-job "התאם קו״ח למשרה" button
     Feature 4: collaborative editor side-panel (POST /api/cv-tailor)
   ===================================================================== */

const SALARY_LABELS = ["לא צוין","מתחת ל-$40K","$40K–$60K","$60K–$80K",
  "$80K–$100K","$100K–$150K","$150K–$200K","$200K+"];
const EXP_LABELS = ["לא צוין","Internship","Entry level","Associate",
  "Mid-Senior level","Director","Executive"];

const HINTS = {
  dataset: "תאר תפקיד · רמת ניסיון · מיקום — הסוכן יאתר משרות וייתן לך קישור ישיר להגשה.",
  scrape:  "משרות חיות מ-LinkedIn ישראל · לחץ ⚙ לסריקה נוספת · קישור ישיר להגשה.",
};
const PLACEHOLDERS = {
  dataset: "איזו משרה לאתר ולהגיש עבורך? לדוגמה: Senior Data Analyst in NY",
  scrape:  "איזו משרה בישראל לאתר עבורך? לדוגמה: דאטה אנליסט בתל אביב",
};

/* =====================================================================
   CV STATE — module-level; browser is the single source of truth.
   cv_text  : raw text extracted from the uploaded file (or pasted text)
   profile  : {skills, titles_held, seniority, years, city, state, domains}
   history  : [{role:'user'|'assistant', content}] — the CV conversation
   roles    : roles from the last CV search (for pagination)
   ===================================================================== */
const cvState = {
  cv_text: null,
  profile: null,
  history: [],   // [{role:'user'|'assistant', content}] — the CV conversation
  roles: [],     // roles from the last CV search (for pagination)
  searchCriteria: null,  // amendable search levers (Flow B): roles/city/state/seniority/years/remote
};

/* =====================================================================
   SAVED-CV LIBRARY — browser-side only (localStorage, no server).
   Entries: { jobTitle, company, cv_text, savedAt } — most-recent-first.
   Cap: 20 entries, each cv_text truncated at 50 000 chars.
   ===================================================================== */
const savedCvLib = (() => {
  const KEY      = "savedCvVersions";
  const MAX_ENTRIES  = 20;
  const MAX_CV_LEN   = 50000;

  function load() {
    try { return JSON.parse(localStorage.getItem(KEY) || "[]"); }
    catch { return []; }
  }
  function save(list) {
    try { localStorage.setItem(KEY, JSON.stringify(list)); } catch {}
  }

  /** Persist a new tailored CV version; deduplicates by jobTitle (most-recent wins). */
  function addEntry(jobTitle, company, cv_text) {
    const list = load().filter((e) => e.jobTitle !== jobTitle);   // remove prior version for same job
    list.unshift({
      jobTitle: String(jobTitle).slice(0, 200),
      company:  String(company  || "").slice(0, 200),
      cv_text:  String(cv_text  || "").slice(0, MAX_CV_LEN),
      savedAt:  new Date().toISOString(),
    });
    save(list.slice(0, MAX_ENTRIES));
  }

  /** Return all saved entries (most-recent-first). */
  function getAll() { return load(); }

  return { addEntry, getAll };
})();

/**
 * Build a safe Blob object-URL download link for a CV version.
 * The CV text goes into a Blob — never into innerHTML.
 * Returns an <a> element (not a string) to be appended to a bubble.
 */
function makeCvDownloadLink(jobTitle, cv_text, linkText) {
  const blob = new Blob([cv_text], { type: "text/plain;charset=utf-8" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href     = url;
  a.download = `CV - ${jobTitle}.txt`;
  a.textContent = linkText;
  a.className   = "cv-download-link";
  // Revoke the object URL after the click so memory is freed
  a.addEventListener("click", () => setTimeout(() => URL.revokeObjectURL(url), 30000));
  return a;
}

marked.setOptions({ breaks: true });

/** Sanitize markdown before injecting into innerHTML — prevents stored XSS */
function mdSafe(s) { return DOMPurify.sanitize(marked.parse(s || "")); }

const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[c]));

let activeSrc = "dataset";
// per-tab conversation state for "show me more" pagination
const convo = { dataset: { lastQuery: "", shown: 0 }, scrape: { lastQuery: "", shown: 0 } };

/* ---------- elements ---------- */
const stage       = document.getElementById("stage");
const hero        = document.getElementById("hero");
const promptEl    = document.getElementById("prompt");
const sendBtn     = document.getElementById("sendBtn");
const composer    = document.getElementById("composer");
const composerHint= document.getElementById("composerHint");
const gearBtn     = document.getElementById("gearBtn");
const threads = {
  dataset: document.getElementById("thread-dataset"),
  scrape:  document.getElementById("thread-scrape"),
};

const scrollDown = () => { stage.scrollTop = stage.scrollHeight; };
const threadHasMsgs = (src) => threads[src].childElementCount > 0;

async function postJSON(url, body, signal) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
                              body: JSON.stringify(body), signal });
  return r.json();
}

/* ---------- in-flight request control (stop button) ---------- */
let inflightController = null;
const isBusy  = () => inflightController !== null;
const isAbort = (err) => !!err && err.name === "AbortError";
/** Start a cancellable turn: returns an AbortSignal and flips the send button to "stop". */
function beginRequest() {
  inflightController = new AbortController();
  composer.classList.add("busy");
  sendBtn.setAttribute("aria-label", "עצור");
  sendBtn.title = "עצור";
  return inflightController.signal;
}
function endRequest() {
  inflightController = null;
  composer.classList.remove("busy");
  sendBtn.setAttribute("aria-label", "שלח");
  sendBtn.removeAttribute("title");
}
function abortRequest() { if (inflightController) inflightController.abort(); }

/* ---------- composer textarea: auto-grow to fit, then scroll ---------- */
function autoGrow() {
  promptEl.style.height = "auto";
  promptEl.style.height = Math.min(promptEl.scrollHeight, 154) + "px";
}
function clearPrompt() { promptEl.value = ""; autoGrow(); }

/* ---------- chat rendering ---------- */
function addUser(src, text) {
  const w = document.createElement("div");
  w.className = "msg user";
  w.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  threads[src].appendChild(w); scrollDown();
}
const THINKING = {
  dataset: ["מאתר משרות עבורך…", "מנתח התאמות ובודק סיכונים…", "מכין המלצות להגשה…"],
  scrape:  ["מאתר משרות חיות בישראל…", "בודק רמת תחרותיות…", "מכין המלצות להגשה…"],
};
function addThinking(src) {
  const w = document.createElement("div");
  w.className = "msg agent thinking";
  w.innerHTML = `<div class="bubble"><span class="ttext"></span><span class="tdots"><span></span><span></span><span></span></span></div>`;
  const phrases = THINKING[src] || THINKING.dataset;
  const textEl = w.querySelector(".ttext");
  textEl.textContent = phrases[0];
  let i = 0;
  const timer = setInterval(() => { i = (i + 1) % phrases.length; textEl.textContent = phrases[i]; }, 1600);
  threads[src].appendChild(w); scrollDown();
  return { stop: () => clearInterval(timer), remove: () => { clearInterval(timer); w.remove(); } };
}
function addAgent(src, reply, cardsHtml) {
  const w = document.createElement("div");
  w.className = "msg agent";
  w.innerHTML = `<div class="bubble"><div class="md">${mdSafe(reply || "")}</div>${cardsHtml || ""}</div>`;
  threads[src].appendChild(w); scrollDown();
}

/* ---------- job cards ---------- */
/** Only allow http(s) job links into href — blocks javascript:/data: from scraped 3rd-party URLs. */
function safeHref(url) {
  try { return ["https:", "http:"].includes(new URL(url).protocol) ? url : "#"; }
  catch { return "#"; }
}
/** "Improve CV for this job" — only shown once a CV is loaded. Opens the IDE improver. */
function improveBtn(m) {
  if (!cvState.cv_text) return "";
  const jobData = JSON.stringify({
    title: m.title || "",
    company: m.company && m.company !== "nan" ? m.company : "",
    description: m.description || m.blurb || "",
    skills: m.skills || [],
  });
  return `<button type="button" class="btn improve-btn" data-job="${escapeHtml(jobData)}">✨ שפר קו״ח למשרה זו</button>`;
}
function jobActions(m) {
  const hasApply = m.application_url && m.application_url !== "nan" && safeHref(m.application_url) !== "#";
  const hasView  = m.job_posting_url && m.job_posting_url !== "nan" && safeHref(m.job_posting_url) !== "#";
  return `
    <div class="job-actions">
      <a class="btn primary" href="${hasApply ? safeHref(m.application_url) : "#"}" ${hasApply ? 'target="_blank" rel="noopener"' : 'aria-disabled="true"'}>הגש מועמדות ↗</a>
      <a class="btn" href="${hasView ? safeHref(m.job_posting_url) : "#"}" ${hasView ? 'target="_blank" rel="noopener"' : 'aria-disabled="true"'}>צפה במשרה</a>
      ${improveBtn(m)}
    </div>`;
}
function cardShell(m, chipsHtml, anomaly) {
  const company = m.company && m.company !== "nan" ? m.company : "מעסיק לא ידוע";
  return `
    <div class="job-card ${anomaly ? "anomaly" : ""}">
      <div class="job-top">
        <div>
          <div class="job-title">${escapeHtml(m.title || "")}</div>
          <div class="job-company">${escapeHtml(company)} · ${escapeHtml(m.location || "")}</div>
        </div>
        <span class="rank">#${m.rank} · ${(m.score || 0).toFixed(2)}</span>
      </div>
      <div class="job-chips">${chipsHtml}</div>
      ${jobActions(m)}
    </div>`;
}
function datasetCard(m) {
  const anomaly = m.is_anomaly === 1;
  const cvFitTag = (typeof m.cv_fit === "number")
    ? `<span class="tag tag-cvfit">🎯 התאמה לקו״ח: <span class="v">${m.cv_fit}%</span></span>` : "";
  const chips =
    `<span class="tag">שכר: <span class="v">${SALARY_LABELS[m.salary_band]}</span></span>` +
    `<span class="tag">ניסיון: <span class="v">${EXP_LABELS[m.experience_rank]}</span></span>` +
    (anomaly ? `<span class="tag warn">⚠ אנומליית שוק</span>` : "") +
    cvFitTag;
  return cardShell(m, chips, anomaly);
}
function scrapeCard(m) {
  const anomaly = m.is_anomaly === 1;
  const exp = m.experience_level ? `<span class="tag">ניסיון: <span class="v">${escapeHtml(m.experience_level)}</span></span>` : "";
  const contract = m.contract_type ? `<span class="tag">${escapeHtml(m.contract_type)}</span>` : "";
  const salary = m.salary ? `<span class="tag">שכר: <span class="v">${escapeHtml(m.salary)}</span></span>` : "";
  const apps = m.applications_count ? `<span class="tag">${escapeHtml(m.applications_count)}</span>` : "";
  const seg = (m.cluster_id != null && m.cluster_id >= 0) ? `<span class="tag">סגמנט: <span class="v">#${m.cluster_id}</span></span>` : "";
  const anom = anomaly ? `<span class="tag warn">⚠ אנומליית שוק</span>` : "";
  const cvFitTag = (typeof m.cv_fit === "number")
    ? `<span class="tag tag-cvfit">🎯 התאמה לקו״ח: <span class="v">${m.cv_fit}%</span></span>` : "";
  return cardShell(m, exp + contract + salary + apps + seg + anom + cvFitTag, anomaly);
}

/* interleave: intro, then for each job [blurb -> its card], then a closing tip */
function interleaved(intro, matches, tip, cardFn) {
  let html = intro ? `<div class="md intro">${mdSafe(intro)}</div>` : "";
  html += matches.map((m) => `
    <div class="job-block">
      ${m.blurb ? `<div class="md blurb">${mdSafe(m.blurb)}</div>` : ""}
      ${cardFn(m)}
    </div>`).join("");
  if (tip) html += `<div class="md tip">${mdSafe(tip)}</div>`;
  return html;
}
function addAgentRich(src, innerHtml) {
  const w = document.createElement("div");
  w.className = "msg agent";
  w.innerHTML = `<div class="bubble rich">${innerHtml}</div>`;
  threads[src].appendChild(w); scrollDown();
}

/* ---------- source toggle ---------- */
function setSource(src) {
  activeSrc = src;
  document.querySelectorAll(".src").forEach((b) => b.classList.toggle("active", b.dataset.src === src));
  threads.dataset.classList.toggle("hidden", src !== "dataset");
  threads.scrape.classList.toggle("hidden", src !== "scrape");
  gearBtn.classList.toggle("inactive", src !== "scrape");   // reserve its space — no topbar shift
  promptEl.placeholder = PLACEHOLDERS[src];
  composerHint.textContent = HINTS[src];
  hero.style.display = threadHasMsgs(src) ? "none" : "";
  if (src === "scrape") refreshScrapeConfig();
  scrollDown();
}
document.querySelectorAll(".src").forEach((b) =>
  b.addEventListener("click", () => setSource(b.dataset.src)));

/* ---------- status pill ---------- */
const statusEl = document.getElementById("status");
const statusTextEl = document.getElementById("statusText");
const setStatus = (state, text) => { if (!statusEl || !statusTextEl) return; statusEl.className = "status " + state; statusTextEl.textContent = text; };

/* ---------- submit (routes to active source) ---------- */
async function submitPrompt() {
  const prompt = promptEl.value.trim();
  if (!prompt) return;
  const src = activeSrc;
  hero.style.display = "none";
  // In CV-coaching mode the composer talks to the tailoring workspace, not search.
  if (tailorState.active) { submitTailorTurn(src, prompt); return; }
  // In no-CV interview mode the composer answers clarifying questions.
  if (onboardState.active) { handleInterviewAnswer(src, prompt); return; }

  // ── Saved-CV recall intercept ──
  // Detect Hebrew/English phrases that mean "show me my saved CV versions".
  const RECALL_RE = /קורות[\s ]*חיים[\s ]*שמור|גרסאות[\s ]*שמור|saved[\s ]*cv|my[\s ]*saved[\s ]*cv/i;
  if (RECALL_RE.test(prompt)) {
    addUser(src, prompt);
    clearPrompt();
    const saved = savedCvLib.getAll();
    if (!saved.length) {
      addAgent(src, "אין לך עדיין גרסאות קורות חיים שמורות. התאם קו״ח למשרה ותישמר אוטומטית.", "");
    } else {
      // Build the list safely: titles + dates rendered through mdSafe,
      // each entry gets its own Blob download link appended as a real DOM node.
      const w = document.createElement("div");
      w.className = "msg agent";
      const bubble = document.createElement("div");
      bubble.className = "bubble";

      const mdLines = ["**גרסאות קורות חיים שמורות:**", ""];
      saved.forEach((e, i) => {
        const date = e.savedAt ? new Date(e.savedAt).toLocaleDateString("he-IL") : "";
        mdLines.push(`${i + 1}. **${e.jobTitle}**${e.company ? " · " + e.company : ""}${date ? " · " + date : ""}`);
      });
      const mdDiv = document.createElement("div");
      mdDiv.className = "md";
      mdDiv.innerHTML = mdSafe(mdLines.join("\n"));   // job titles go through DOMPurify
      bubble.appendChild(mdDiv);

      // Append one download link per entry — CV text never touches innerHTML.
      saved.forEach((e) => {
        const row = document.createElement("div");
        row.className = "cv-save-row";
        const lbl = document.createElement("span");
        lbl.className = "cv-save-label";
        lbl.textContent = e.jobTitle;           // textContent — safe
        const dl = makeCvDownloadLink(e.jobTitle, e.cv_text, "הורדה ↓");
        row.appendChild(lbl);
        row.appendChild(dl);
        bubble.appendChild(row);
      });

      w.appendChild(bubble);
      threads[src].appendChild(w);
      scrollDown();
    }
    promptEl.focus();
    return;   // skip normal search
  }

  clearPrompt();
  runSearch(src, prompt, true);
}

/** Send one search/chat turn and render the result. When showUser is false the
 *  prompt is run on the user's behalf (auto-search) with no user bubble. Also the
 *  single place that enters a seeded interview when the agent infers job-search intent. */
async function runSearch(src, prompt, showUser = true) {
  if (showUser) addUser(src, prompt);
  const signal = beginRequest();
  const t = addThinking(src);
  const isDataset = src === "dataset";
  if (isDataset) setStatus("busy", "הסוכן חושב…");
  try {
    const searchBody = { prompt, lastQuery: convo[src].lastQuery, offset: convo[src].shown };
    if (showUser) cvState.history.push({ role: "user", content: prompt });
    searchBody.history = cvState.history.slice(-8);
    searchBody.lastMatches = (lastMatches[src] || []).map(m => ({
      rank: m.rank, title: m.title, company: m.company, location: m.location,
      experience_level: m.experience_level, salary_band: m.salary_band,
    }));
    if (cvState.cv_text) {
      searchBody.cv_text = cvState.cv_text;
      searchBody.profile = cvState.profile;
      searchBody.cv_roles = cvState.roles || [];
    }
    if (cvState.searchCriteria) searchBody.searchCriteria = cvState.searchCriteria;
    const data = await postJSON(isDataset ? "/api/chat" : "/api/scrape-search", searchBody, signal);
    t.remove();
    if (data.error) addAgent(src, "אירעה שגיאה: " + data.error, "");
    else if (data.mode === "interview") {     // agent inferred job-search intent (no CV yet)
      enterInterview(src, data);
      if (isDataset) setStatus("live", "מחובר");
      return;                                  // interview owns the turn
    }
    else if (data.mode === "tailor") {
      addAgent(src, data.intro || data.reply || "", "");
      const job = resolveJob(data.tailor_query, src);
      if (cvState.cv_text && job) enterTailoring(src, jobForTailor(job));
      else addAgent(src, "לא הצלחתי לזהות לאיזו משרה התכוונת — נסה לציין מספר (#) או כותרת.", "");
    }
    else if (!data.matches || !data.matches.length)
      addAgent(src, data.intro || data.reply || "לא נמצאו משרות מתאימות. נסו לנסח מחדש או לשנות מיקום.", "");
    else {
      lastMatches[src] = data.matches;
      addAgentRich(src, interleaved(data.intro || data.reply, data.matches, data.tip, isDataset ? datasetCard : scrapeCard));
    }
    const got = (data.matches || []).length;
    if (data.mode === "more") convo[src].shown += got;
    else if (data.mode === "search") { convo[src].lastQuery = data.query || prompt; convo[src].shown = got; }
    const agentText = data.reply || data.intro || "";
    if (agentText) cvState.history.push({ role: "assistant", content: agentText });
    cvState.history = cvState.history.slice(-16);
    if (cvState.cv_text && Array.isArray(data.roles)) cvState.roles = data.roles;
    if (data.searchCriteria) cvState.searchCriteria = data.searchCriteria;
    if (isDataset) setStatus("live", "מחובר");
  } catch (err) {
    t.remove();
    if (isAbort(err)) {                        // user pressed stop — drop the turn quietly
      if (isDataset) setStatus("live", "מחובר");
    } else {
      addAgent(src, "לא ניתן להתחבר לשרת. ודאו ש-`python app.py` רץ.", "");
      if (isDataset) setStatus("", "מנותק");
    }
  } finally {
    endRequest(); promptEl.focus();
  }
}
composer.addEventListener("submit", (e) => {
  e.preventDefault();
  if (isBusy()) { abortRequest(); return; }    // composer is in "stop" mode
  submitPrompt();
});
// Enter sends · Shift+Enter inserts a line break (handled natively by the textarea).
promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    if (!isBusy()) composer.requestSubmit();
  }
});
promptEl.addEventListener("input", autoGrow);
autoGrow();

/* =====================================================================
   LIVE-SCAN DRAWER (field editor) — same API + field IDs as before
   ===================================================================== */
const drawer        = document.getElementById("drawer");
const drawerBackdrop= document.getElementById("drawerBackdrop");
const openDrawer  = () => { drawerBackdrop.hidden = false; refreshScrapeConfig(); };
const closeDrawer = () => { drawerBackdrop.hidden = true; };
gearBtn.addEventListener("click", openDrawer);
document.getElementById("drawerClose").addEventListener("click", closeDrawer);
drawerBackdrop.addEventListener("click", (e) => { if (e.target === drawerBackdrop) closeDrawer(); });  // click outside the card
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !drawerBackdrop.hidden) closeDrawer(); });

let pollTimer = null;
const EXP_LABEL_HE = { "Internship":"התמחות", "Entry level":"ג'וניור", "Associate":"מיומן",
  "Mid-Senior level":"בכיר", "Director":"ניהול", "Executive":"הנהלה" };
const CONTRACT_LABEL_HE = { "Full-time":"מלאה", "Part-time":"חלקית", "Contract":"קבלן",
  "Temporary":"זמני", "Internship":"התמחות", "Volunteer":"התנדבות", "Other":"אחר" };
const DATE_LABEL_HE = { "Past day":"24 שעות", "Past week":"שבוע", "Past month":"חודש" };
const REMOTE_LABEL_HE = { "On-site":"במשרד", "Remote":"מרחוק", "Hybrid":"היברידי" };

const fTitle = document.getElementById("fTitle");
const fLocation = document.getElementById("fLocation");
const fDate = document.getElementById("fDate");
const fRemote = document.getElementById("fRemote");
const fLimit = document.getElementById("fLimit");
const fExp = document.getElementById("fExp");
const fContract = document.getElementById("fContract");

function renderSelect(sel, options, labels, anyLabel, value) {
  sel.innerHTML = `<option value="">${anyLabel}</option>` +
    options.map((o) => `<option value="${o}">${labels[o] || o}</option>`).join("");
  sel.value = value || "";
}
function renderChips(container, options, labels, selected) {
  container.innerHTML = options.map((o) =>
    `<button type="button" class="chip ${selected.includes(o) ? "on" : ""}" data-val="${o}">${labels[o] || o}</button>`).join("");
  container.querySelectorAll(".chip").forEach((ch) => ch.addEventListener("click", () => ch.classList.toggle("on")));
}
const chipValues = (container) => Array.from(container.querySelectorAll(".chip.on")).map((c) => c.dataset.val);

function currentFilters() {
  return {
    title: fTitle.value.trim(),
    location: fLocation.value.trim() || "Israel",
    experienceLevel: chipValues(fExp),
    datePosted: fDate.value,
    contractType: chipValues(fContract),
    remote: fRemote.value,
    limit: parseInt(fLimit.value, 10) || 100,
  };
}
function fillForm(c) {
  const f = c.filters || {}, opt = c.options || {};
  if (document.activeElement !== fTitle) fTitle.value = f.title || "";
  if (document.activeElement !== fLocation) fLocation.value = f.location || "Israel";
  if (document.activeElement !== fLimit) fLimit.value = f.limit || 100;
  renderSelect(fDate, opt.datePosted || [], DATE_LABEL_HE, "בכל זמן", f.datePosted);
  renderSelect(fRemote, opt.remote || [], REMOTE_LABEL_HE, "הכל", f.remote);
  renderChips(fExp, opt.experienceLevel || [], EXP_LABEL_HE, f.experienceLevel || []);
  renderChips(fContract, opt.contractType || [], CONTRACT_LABEL_HE, f.contractType || []);
}
function scrapeStatusText(c) {
  const last = c.last_scraped ? String(c.last_scraped).replace("T", " ") : "—";
  let msg = `נסרק לאחרונה: ${last} · ${c.jobs_stored} משרות במאגר`;
  if (c.last_result && c.last_result.ok === false) msg += ` · ⚠ ${c.last_result.error}`;
  else if (c.last_result && c.last_result.ok) msg += ` · ✓ נוספו ${c.last_result.scraped}`;
  return msg;
}
function updateStatus(c) {
  document.getElementById("scrapeStatus").textContent = scrapeStatusText(c);
  const btn = document.getElementById("scrapeNowBtn");
  btn.disabled = c.running;
  btn.textContent = c.running ? "סורק…" : "סרוק עכשיו";
  clearTimeout(pollTimer);
  if (c.running) pollTimer = setTimeout(pollStatus, 3000);
}
async function refreshScrapeConfig() {
  try { const c = await (await fetch("/api/scrape-config")).json(); fillForm(c); updateStatus(c); } catch {}
}
async function pollStatus() {
  try { updateStatus(await (await fetch("/api/scrape-config")).json()); } catch {}
}

document.getElementById("saveFiltersBtn").addEventListener("click", async () => {
  const msg = document.getElementById("urlMsg");
  msg.textContent = "שומר…";
  const c = await postJSON("/api/scrape-config", { filters: currentFilters() });
  if (c.ok) { msg.textContent = "נשמר ✓"; fillForm(c); updateStatus(c); }
  else msg.textContent = c.error || "שגיאה";
  setTimeout(() => (msg.textContent = ""), 2500);
});

document.getElementById("scrapeNowBtn").addEventListener("click", async () => {
  const msg = document.getElementById("urlMsg");
  await postJSON("/api/scrape-config", { filters: currentFilters() });
  const data = await postJSON("/api/scrape-now", { limit: parseInt(fLimit.value, 10) || undefined });
  if (!data.ok) { msg.textContent = data.error || "שגיאה"; setTimeout(() => (msg.textContent = ""), 3500); return; }
  msg.textContent = "הסריקה החלה…"; setTimeout(() => (msg.textContent = ""), 2800);
  pollStatus();
});

/* =====================================================================
   "HOW IT WORKS" EXPLAINER — generic renderer driven by explain.json
   ===================================================================== */
const explainBackdrop = document.getElementById("explainBackdrop");
const STAT_LABELS = { jobs: "משרות", clusters: "אשכולות", anomalies: "אנומליות",
                      states: "מדינות", districts: "מחוזות" };
const COL_LABELS = { title: "כותרת", company: "חברה", state: "מדינה", district: "מחוז",
                     experience: "ניסיון", salary: "שכר", applicants: "מועמדים",
                     cluster: "אשכול", anomaly: "חריג" };
const FLOW_DIAGRAM = `
  <div class="flow-diagram">
    <div class="fd-node fd-user">בקשה חופשית של המשתמש<small>"Senior Data Analyst in NY"</small></div>
    <div class="fd-arrow">↓</div>
    <div class="fd-node fd-llm">ה-LLM מבין את הבקשה<small>תפקיד · רמת ניסיון · מיקום</small></div>
    <div class="fd-arrow">↓</div>
    <div class="fd-split">
      <div class="fd-node">סינון גאוגרפי קשיח</div>
      <div class="fd-node">TF-IDF Cosine<small>כותרת 0.6 + ניסיון 0.4</small></div>
    </div>
    <div class="fd-arrow">↓</div>
    <div class="fd-node fd-models">המשרות המדורגות נושאות<small>סגמנט אשכול + דגל אנומליה</small></div>
    <div class="fd-arrow">↓</div>
    <div class="fd-node fd-out">סוכן Claude → המלצה אישית + ⚠️ התראות סיכון</div>
  </div>`;

let EXPLAIN = null;
async function loadExplain() {
  if (!EXPLAIN) {
    const r = await fetch("assets/explain/explain.json");
    if (!r.ok) throw new Error("explain.json " + r.status);
    EXPLAIN = await r.json();
  }
  return EXPLAIN;
}

function explainTable(tab) {
  const cols = tab.sample_cols, rows = tab.sample;
  const head = cols.map((c) => `<th>${COL_LABELS[c] || c}</th>`).join("");
  const body = rows.map((r) =>
    `<tr>${cols.map((c) => `<td>${escapeHtml(String(r[c] ?? ""))}</td>`).join("")}</tr>`).join("");
  return `<div class="explain-table-wrap"><table class="explain-table">
    <thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function renderStep(tab, i) {
  const s = tab.steps[i];
  let html = `<div class="step-summary">${escapeHtml(s.summary)}</div>`;
  if (s.points && s.points.length)
    html += `<ul class="step-points">${s.points.map((p) => `<li>${escapeHtml(p)}</li>`).join("")}</ul>`;
  if (s.diagram === "flow") html += FLOW_DIAGRAM;
  if (s.charts && s.charts.length)
    html += `<div class="step-charts">${s.charts.map((c) =>
      `<img loading="lazy" src="assets/explain/${c}" alt="">`).join("")}</div>`;
  if (s.table) html += explainTable(tab);
  document.getElementById("explainBody").innerHTML = html;
  document.getElementById("explainBody").scrollTop = 0;
}

let curTab = null, curIdx = 0;
function goStep(i) {
  if (!curTab) return;
  curIdx = Math.max(0, Math.min(curTab.steps.length - 1, i));
  const flow = document.getElementById("explainFlow");
  flow.querySelectorAll(".flow-step").forEach((x) => x.classList.toggle("active", +x.dataset.i === curIdx));
  const active = flow.querySelector(".flow-step.active");
  if (active) active.scrollIntoView({ inline: "center", block: "nearest" });
  renderStep(curTab, curIdx);
  document.getElementById("explainCounter").textContent = `שלב ${curIdx + 1} / ${curTab.steps.length}`;
  document.getElementById("explainPrev").disabled = curIdx === 0;
  document.getElementById("explainNext").disabled = curIdx === curTab.steps.length - 1;
}

async function openExplain(src) {
  explainBackdrop.hidden = false;                 // open immediately so the click always responds
  const body = document.getElementById("explainBody");
  let data;
  try {
    data = await loadExplain();
  } catch (e) {
    document.getElementById("explainTitle").textContent = "איך זה עובד";
    document.getElementById("explainStats").textContent = "";
    document.getElementById("explainFlow").innerHTML = "";
    body.innerHTML = '<div class="step-summary">לא ניתן לטעון את ההסבר כרגע. נסו לרענן את הדף.</div>';
    return;
  }
  curTab = data[src];
  if (!curTab) return;
  document.getElementById("explainTitle").textContent = "איך זה עובד · " + curTab.title;
  document.getElementById("explainStats").textContent =
    Object.entries(curTab.stats).map(([k, v]) => `${Number(v).toLocaleString()} ${STAT_LABELS[k] || k}`).join(" · ");
  const flow = document.getElementById("explainFlow");
  flow.innerHTML = curTab.steps.map((s, i) =>
    `<button type="button" class="flow-step" data-i="${i}">
       <span class="flow-num">${i + 1}</span><span class="flow-label">${escapeHtml(s.label.replace(/^\s*\d+\s*·\s*/, ""))}</span>
     </button>${i < curTab.steps.length - 1 ? '<span class="flow-arrow">›</span>' : ""}`).join("");
  flow.querySelectorAll(".flow-step").forEach((b) => b.addEventListener("click", () => goStep(+b.dataset.i)));
  goStep(0);
}

const closeExplain = () => { explainBackdrop.hidden = true; };
document.getElementById("howLink").addEventListener("click", () => openExplain(activeSrc));
document.getElementById("explainClose").addEventListener("click", closeExplain);
document.getElementById("explainPrev").addEventListener("click", () => goStep(curIdx - 1));
document.getElementById("explainNext").addEventListener("click", () => goStep(curIdx + 1));
explainBackdrop.addEventListener("click", (e) => { if (e.target === explainBackdrop) closeExplain(); });
document.addEventListener("keydown", (e) => {
  if (explainBackdrop.hidden) return;
  if (e.key === "Escape") closeExplain();
  else if (e.key === "ArrowLeft") goStep(curIdx + 1);    // RTL: left = forward
  else if (e.key === "ArrowRight") goStep(curIdx - 1);
});

/* =====================================================================
   FEATURE 1 & 2 — CV UPLOAD (composer clip button + conversational flow)
   ===================================================================== */
const cvFileInput    = document.getElementById("cvFileInput");
const cvUploadLabel  = document.getElementById("cvUploadLabel");
const cvClipText     = document.getElementById("cvClipText");
const heroCvBtn      = document.getElementById("heroCvBtn");
const cvLoading      = document.getElementById("cvLoading");
const cvError        = document.getElementById("cvError");
const cvPasteWrap    = document.getElementById("cvPasteWrap");
const cvPasteArea    = document.getElementById("cvPasteArea");
const cvPasteSubmitBtn = document.getElementById("cvPasteSubmitBtn");

/* Bundled sample CV (fictional) — lets a user test the whole flow without a file.
   Fed through the same /api/cv text path as a manual paste, so the profile pipeline,
   title matching and scorecards all run for real. */
const MOCK_CV_TEXT = `Daniel Mercer
Austin, Texas, United States | daniel.mercer.demo@email.com | +1 (512) 555-0194 | linkedin.com/in/danielmercer-demo | github.com/danielmercer-demo

SUMMARY
Analytical and business-oriented AI Product Analyst with 4+ years of experience across product operations, customer analytics, workflow automation, and data-driven decision-making. Skilled at translating business problems into technical requirements, building dashboards, analyzing user behavior, and supporting AI-powered product initiatives from discovery through implementation.
Experienced in working with cross-functional teams including product managers, engineers, designers, sales, and customer success. Strong background in SQL, Python, BI tools, CRM systems, and process optimization. Passionate about using AI, automation, and data products to improve operational efficiency, customer experience, and business performance.

EDUCATION
University of Texas at Dallas — Richardson, TX
B.S. in Information Systems (2015 – 2019)
Relevant coursework: Data Analysis, Database Systems, Business Intelligence, Software Project Management, Human-Computer Interaction, Information Systems Strategy, Digital Transformation.

EXPERIENCE
AI Product Analyst — NovaWorks Technologies, Austin, TX (March 2023 – Present)
- Supported the development of an AI assistant designed to help operations teams automate repetitive customer support and internal knowledge tasks.
- Conducted user research sessions with customer success managers and end users to identify pain points in onboarding, ticket handling, and knowledge retrieval.
- Defined product requirements for AI-powered features, including smart search, automated ticket classification, and suggested response generation.
- Built SQL dashboards to track adoption, feature usage, retention, and support ticket reduction after AI feature launches.
- Analyzed more than 120,000 support interactions to identify recurring customer issues and opportunities for automation.
- Worked with engineering teams to define data requirements for model evaluation, feedback loops, and performance monitoring.
- Helped reduce average support resolution time by 18% through improved ticket tagging and AI-assisted routing.

Business Operations Analyst — Brightlane Retail Group, Chicago, IL (January 2021 – February 2023)
- Analyzed sales, inventory, and customer feedback data across 18 retail locations to identify operational inefficiencies.
- Built Power BI dashboards for management teams, covering revenue trends, stock movement, employee performance, and customer satisfaction.
- Automated weekly reporting workflows using Excel, SQL exports, and Python scripts, saving approximately 8 hours of manual work per week.
- Identified underperforming product categories and presented recommendations that contributed to a 9% improvement in quarterly sell-through rate.
- Supported CRM segmentation initiatives by analyzing customer purchase frequency, average order value, and loyalty program behavior.

Customer Success & Data Coordinator — CloudDesk Solutions, Remote, United States (July 2019 – December 2020)
- Managed customer onboarding processes for small business clients, including account setup, workflow configuration, and training sessions.
- Monitored product usage data to identify customers at risk of churn and suggested proactive engagement actions.
- Created monthly customer health reports using CRM data, support history, and product usage metrics.
- Improved onboarding checklist completion rates by 22% by redesigning follow-up workflows and email templates.

PROJECTS
AI Support Assistant Evaluation — Designed a demo evaluation framework for an AI support assistant: test scenarios, expected responses, quality scoring criteria, and analysis of common failure cases (hallucinations, irrelevant answers, incomplete escalation). Tools: Excel, Python, SQL, Notion.
Customer Feedback Trend Analysis — Analyzed customer feedback data before and after a product update to identify changes in feedback volume, sentiment, and issue categories. Tools: Python, Pandas, Power BI.
Internal Knowledge Base Chatbot Demo — Prototype concept for an internal knowledge chatbot using a RAG architecture and prompt engineering. Tools: OpenAI API concept, Google Sheets, Notion.

TECHNICAL SKILLS
Product: product requirements, user stories, KPI definition, roadmap support, customer journey mapping, stakeholder interviews, feature prioritization, competitive analysis.
Data & Analytics: SQL, Python, Excel, data cleaning, exploratory data analysis, cohort analysis, funnel analysis, A/B testing, KPI dashboards, reporting automation.
AI: LLM-based workflows, prompt design, RAG concepts, chatbot evaluation, process automation, AI product research, customer support automation.
Programming & Querying: Python, SQL, basic JavaScript.
Data Tools: Excel, Google Sheets, Pandas, NumPy.
BI & Visualization: Power BI, Tableau, Looker Studio.
Product & Project Tools: Jira, Trello, Monday.com, Notion, Miro.
CRM & Support: HubSpot, Salesforce, Zendesk, Intercom.
Databases: PostgreSQL, MySQL, BigQuery basics.

CERTIFICATIONS
Google Data Analytics Certificate — 2022
Microsoft Power BI Data Analyst Fundamentals — 2022
Product Analytics Micro-Certification — 2023
Introduction to Generative AI for Business — 2024

LANGUAGES
English — Native | Spanish — Professional working proficiency

ADDITIONAL INFORMATION
Authorized to work in the United States. Comfortable in fast-paced startup environments. Strong ability to bridge business, product, and technical teams. Interested in AI products, automation, customer experience, and data-driven operations.

(This CV is entirely fictional and was created for demonstration and testing purposes only.)`;

/* CV source picker — the upload buttons open this so the user can either pick a file
   or load the bundled sample CV for a one-click end-to-end test. */
const cvSrcBackdrop  = document.getElementById("cvSrcBackdrop");
const cvSrcUploadBtn = document.getElementById("cvSrcUploadBtn");
const cvSrcSampleBtn = document.getElementById("cvSrcSampleBtn");
const cvSrcClose     = document.getElementById("cvSrcClose");
function openCvSrcModal() { cvSrcBackdrop.hidden = false; cvSrcUploadBtn.focus(); }
function hideCvSrcModal() { cvSrcBackdrop.hidden = true; }
cvSrcUploadBtn.addEventListener("click", () => { hideCvSrcModal(); cvFileInput.click(); });
cvSrcSampleBtn.addEventListener("click", () => { hideCvSrcModal(); uploadCvText(MOCK_CV_TEXT); });
cvSrcClose.addEventListener("click", hideCvSrcModal);
cvSrcBackdrop.addEventListener("click", (e) => { if (e.target === cvSrcBackdrop) hideCvSrcModal(); });
document.addEventListener("keydown", (e) => { if (!cvSrcBackdrop.hidden && e.key === "Escape") hideCvSrcModal(); });

// Both upload entry points (welcome CTA + composer clip) open the source picker.
heroCvBtn.addEventListener("click", openCvSrcModal);
cvUploadLabel.addEventListener("click", (e) => { e.preventDefault(); openCvSrcModal(); });

/* CV processing modal — on-brand radar + sequential step reveal */
const cvProcBackdrop = document.getElementById("cvProcBackdrop");
let cvProcTimer = null;
function showCvProcessing() {
  const steps = cvProcBackdrop.querySelectorAll(".cvproc-steps li");
  steps.forEach((s, i) => { s.classList.remove("done"); s.classList.toggle("active", i === 0); });
  cvProcBackdrop.hidden = false;
  let i = 0;
  clearInterval(cvProcTimer);
  // Advance the lit step on a timer (the API has no progress events); hold the last one
  // active until the response arrives, so it always feels like work is happening.
  cvProcTimer = setInterval(() => {
    if (i >= steps.length - 1) { clearInterval(cvProcTimer); return; }
    steps[i].classList.replace("active", "done");
    steps[++i].classList.add("active");
  }, 1100);
}
function hideCvProcessing() {
  clearInterval(cvProcTimer);
  cvProcBackdrop.hidden = true;
}

/* "This isn't a CV" modal — shown when the upload parsed but wasn't a résumé.
   Two ways out: build a profile via the few-question interview, or pick another file. */
const notCvBackdrop  = document.getElementById("notCvBackdrop");
const notCvBuildBtn  = document.getElementById("notCvBuildBtn");
const notCvUploadBtn = document.getElementById("notCvUploadBtn");
function hideNotCvModal() { notCvBackdrop.hidden = true; }
function showNotCvModal() {
  hideCvProcessing();
  notCvBackdrop.hidden = false;
  notCvBuildBtn.focus();
}
notCvBuildBtn.addEventListener("click", () => { hideNotCvModal(); startInterview(); });
notCvUploadBtn.addEventListener("click", () => { hideNotCvModal(); cvFileInput.click(); });

/** Called on successful /api/cv response */
function onCvLoaded(data) {
  // A CV upload supersedes any in-progress interview.
  onboardState.active = false;
  cvStepper.hidden = true;
  cvState.cv_text = data.cv_text;
  cvState.profile = data.profile;
  cvState.history = [];
  cvState.roles   = [];
  cvState.searchCriteria = null;
  cvState.hasRealCv = true;          // a real CV was uploaded -> tailoring enhances it

  hideCvProcessing();
  cvLoading.hidden   = true;
  cvError.hidden     = true;
  cvPasteWrap.hidden = true;

  // Reflect the loaded CV on the composer clip so the user has a clear confirmation.
  cvUploadLabel.classList.add("loaded");
  cvClipText.textContent = "✓ קו״ח";

  // Build the identity card and open the profile panel on the right.
  renderIdentity(cvState.profile);
  openProfile();

  const greeting = data.greeting || "נעים להכיר! אילו תפקידים תרצה שאחפש עבורך?";
  hero.style.display = "none";
  addAgent(activeSrc, greeting, "");
  cvState.history.push({ role: "assistant", content: greeting });
}

/** Shows a Hebrew error; optionally shows the paste fallback */
function onCvError(msg, allowPaste) {
  hideCvProcessing();
  cvLoading.hidden = true;
  cvError.hidden   = false;
  cvError.textContent = msg;
  if (allowPaste) cvPasteWrap.hidden = false;
}

/** POST /api/cv with FormData (file upload) */
async function uploadCvFile(file) {
  showCvProcessing();
  cvError.hidden   = true;
  cvPasteWrap.hidden = true;

  try {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch("/api/cv", { method: "POST", body: fd });
    const data = await r.json();
    if (data.ok) {
      onCvLoaded(data);
    } else if (data.not_cv) {
      showNotCvModal();
    } else {
      onCvError(data.error || "שגיאה בעיבוד הקובץ.", data.allow_paste);
    }
  } catch {
    onCvError("לא ניתן להתחבר לשרת. ודאו ש-`python app.py` רץ.", false);
    cvLoading.hidden = true;
  }
}

/** POST /api/cv with JSON body (manual paste) */
async function uploadCvText(text) {
  showCvProcessing();
  cvError.hidden   = true;

  try {
    const data = await postJSON("/api/cv", { cv_text: text });
    if (data.ok) {
      onCvLoaded(data);
    } else if (data.not_cv) {
      showNotCvModal();
    } else {
      onCvError(data.error || "שגיאה בעיבוד הטקסט.", data.allow_paste);
    }
  } catch {
    onCvError("לא ניתן להתחבר לשרת.", false);
    cvLoading.hidden = true;
  }
}

// File input change → upload
cvFileInput.addEventListener("change", () => {
  const file = cvFileInput.files[0];
  if (!file) return;
  uploadCvFile(file);
  cvFileInput.value = ""; // reset so same file can be re-selected
});

// Paste submit button
cvPasteSubmitBtn.addEventListener("click", () => {
  const text = cvPasteArea.value.trim();
  if (!text) return;
  uploadCvText(text);
});

/* =====================================================================
   CV-COACHING WORKSPACE — docked panel + tailoring conversation in the MAIN chat.
   Fully conversational: enter/answer/accept/exit all happen by replying. No buttons.
   ===================================================================== */
const appEl         = document.querySelector(".app");
const profilePanel  = document.getElementById("profilePanel");
const profileClose  = document.getElementById("profileClose");
const profileHandle = document.getElementById("profileHandle");
const editorHandle  = document.getElementById("editorHandle");
const cvDockClose   = document.getElementById("cvDockClose");
const cvDock        = document.getElementById("cvDock");
const cvDockArea    = document.getElementById("cvDockArea");      // contenteditable div
const cvDockJob     = document.getElementById("cvDockJob");
const cvDockBanner  = document.getElementById("cvDockBanner");
const cvIdentityCard= document.getElementById("cvIdentityCard");
const cvStepper     = document.getElementById("cvStepper");

/* Interview (no-CV profile build) state — mirrors tailorState; while active the
   composer routes answers to handleInterviewAnswer() instead of search. */
const onboardState = { active: false, nextField: "role", skipped: [] };
const reduceMotion  = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ---- Collapsible side panels: edge handles open them, header × closes them.
   A handle shows only when its panel exists (opened once) AND is currently collapsed. ---- */
let profileAvail = false, editorAvail = false;
function updateHandles() {
  profileHandle.hidden = !(profileAvail && profilePanel.hidden);
  editorHandle.hidden  = !(editorAvail  && cvDock.hidden);
}
function openProfile() { profileAvail = true; profilePanel.hidden = false; updateHandles(); }
function showProfile() { profilePanel.hidden = false; updateHandles(); }
function hideProfile() { profilePanel.hidden = true;  updateHandles(); }
profileClose.addEventListener("click", hideProfile);
profileHandle.addEventListener("click", showProfile);

/* ---- CV editor (left, contenteditable) ---- */
function getDockCv()      { return cvDockArea.innerText.replace(/ /g, " "); }
function setDockCv(text)  { cvDockArea.textContent = text || ""; }
function openEditor()     { editorAvail = true; cvDock.hidden = false; updateHandles(); }
function closeEditor()    { cvDock.hidden = true; updateHandles(); }
editorHandle.addEventListener("click", openEditor);
cvDockClose.addEventListener("click", closeEditor);

/* ---- Animated CV editing: diff -> highlight the changed region -> typewriter it in ---- */
let typeTimer = null, pendingMark = null, typingTarget = null;

/** Longest common prefix/suffix -> the single changed middle region. */
function diffRegion(oldStr, newStr) {
  oldStr = oldStr || ""; newStr = newStr || "";
  let p = 0;
  while (p < oldStr.length && p < newStr.length && oldStr[p] === newStr[p]) p++;
  let s = 0;
  while (s < oldStr.length - p && s < newStr.length - p &&
         oldStr[oldStr.length - 1 - s] === newStr[newStr.length - 1 - s]) s++;
  return { before: newStr.slice(0, p), changed: newStr.slice(p, newStr.length - s), after: newStr.slice(newStr.length - s) };
}

/** The previous proposed change is now confirmed (the user replied) — un-highlight it. */
function settlePending() {
  if (pendingMark) { pendingMark.classList.add("settled"); pendingMark.classList.remove("typing"); pendingMark = null; }
}

/** Render newCv into the editor; type the changed region into a live highlight. */
function renderCvChange(oldCv, newCv) {
  clearInterval(typeTimer);
  typingTarget = newCv;                          // remember the full target until the anim finishes
  const { before, changed, after } = diffRegion(oldCv, newCv);
  if (!changed) { setDockCv(newCv); typingTarget = null; return; }
  cvDockArea.textContent = "";
  cvDockArea.appendChild(document.createTextNode(before));
  const mk = document.createElement("mark"); mk.className = "cv-change typing";
  cvDockArea.appendChild(mk);
  cvDockArea.appendChild(document.createTextNode(after));
  pendingMark = mk;
  if (reduceMotion) { mk.textContent = changed; mk.classList.remove("typing"); typingTarget = null; return; }
  let i = 0;
  const step = Math.max(1, Math.round(changed.length / 110));   // cap total frames
  typeTimer = setInterval(() => {
    i = Math.min(changed.length, i + step);
    mk.textContent = changed.slice(0, i);
    mk.scrollIntoView({ block: "nearest" });
    if (i >= changed.length) { clearInterval(typeTimer); mk.classList.remove("typing"); typingTarget = null; }
  }, 16);
}

/** Complete any in-flight typewriter immediately so the editor holds the FULL composed CV.
 *  Without this, code that reads the editor mid-animation (e.g. exitTailoring) would capture
 *  only the un-typed common prefix (the name). No-op when nothing is animating, so genuine
 *  manual editor edits are preserved. */
function flushTypewriter() {
  if (typingTarget != null) {
    clearInterval(typeTimer);
    setDockCv(typingTarget);                      // render the complete target into the editor
    typingTarget = null;
  }
}

// Hebrew labels for the seniority levels coming from the profile.
const SENIORITY_HE = { intern: "מתמחה", entry: "ג'וניור", associate: "מיומן",
  "mid-senior": "בכיר", director: "ניהול", executive: "הנהלה" };

// Small element factory shared by the identity view + edit modes.
const cvEl = (tag, cls, txt) => { const n = document.createElement(tag); if (cls) n.className = cls; if (txt != null) n.textContent = txt; return n; };
// Build a stroked SVG icon from path data (safe DOM, no innerHTML).
function cvSvg(paths, size = 15) {
  const NS = "http://www.w3.org/2000/svg";
  const s = document.createElementNS(NS, "svg");
  s.setAttribute("viewBox", "0 0 24 24"); s.setAttribute("width", size); s.setAttribute("height", size); s.setAttribute("aria-hidden", "true");
  paths.forEach((d) => { const p = document.createElementNS(NS, "path"); p.setAttribute("d", d); s.appendChild(p); });
  return s;
}

// Whether the profile is being manually edited (toggled by the pencil button).
let editingProfile = false;

/** Build the candidate "identity card" — cover hero + role + skills + experience.
 *  Safe DOM only (textContent / createElement), never raw innerHTML. */
function renderIdentity(profile) {
  if (editingProfile) { renderIdentityEdit(profile || {}); return; }
  const p = profile || {};
  cvIdentityCard.textContent = "";
  const el = cvEl;

  // ── Hero: cover banner + overlapping monogram avatar + name / role / meta ──
  const hero = el("div", "cvid-hero");
  hero.appendChild(el("div", "cvid-cover"));

  // edit affordance — let the user correct anything the agent inferred (not mid-interview)
  const hasProfile = !!(p.name || (p.skills || []).length || (p.titles_held || []).length ||
                        (p.work_experience || []).length || p.location);
  if (hasProfile && !onboardState.active) {
    const edit = el("button", "cvid-edit");
    edit.type = "button";
    edit.title = "ערוך פרופיל";
    edit.setAttribute("aria-label", "ערוך פרופיל");
    edit.appendChild(cvSvg(["M12 20h9", "M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4 12.5-12.5Z"]));
    edit.addEventListener("click", () => { editingProfile = true; renderIdentity(cvState.profile); });
    hero.appendChild(edit);
  }

  const initials = (p.name || "").trim().split(/\s+/).map((w) => w[0]).slice(0, 2).join("").toUpperCase();
  hero.appendChild(el("div", "cvid-avatar", initials || "?"));
  const htext = el("div", "cvid-hero-txt");
  htext.appendChild(el("div", "cvid-name", p.name || "המועמד/ת"));
  // show ALL roles the candidate is targeting, not just the first
  const titles = (p.titles_held && p.titles_held.length) ? p.titles_held
                 : (p.title ? [p.title] : []);
  if (titles.length) htext.appendChild(el("div", "cvid-role", titles.join(", ")));
  const meta = el("div", "cvid-meta");
  const loc = p.location || [p.city, p.state].filter(Boolean).join(", ");
  const lvl = SENIORITY_HE[p.seniority] || "";
  const yrs = (p.years != null) ? `${p.years} שנ׳ ניסיון` : "";
  [loc, lvl, yrs].filter(Boolean).forEach((bit, i, arr) => {
    meta.appendChild(el("span", "cvid-meta-bit", bit));
    if (i < arr.length - 1) meta.appendChild(el("span", "cvid-dot", "·"));
  });
  htext.appendChild(meta);
  hero.appendChild(htext);
  cvIdentityCard.appendChild(hero);

  const block = (title) => { const b = el("section", "cvid-block"); b.appendChild(el("h3", "cvid-h", title)); return b; };

  // ── Skills ──
  if ((p.skills || []).length) {
    const b = block("כישורים");
    const wrap = el("div", "cvid-skills");
    p.skills.slice(0, 16).forEach((sk) => wrap.appendChild(el("span", "cvid-skill", String(sk))));
    b.appendChild(wrap); cvIdentityCard.appendChild(b);
  }
  // ── Experience (timeline) ──
  if ((p.work_experience || []).length) {
    const b = block("ניסיון");
    const tl = el("ol", "cvid-timeline");
    p.work_experience.slice(0, 5).forEach((w) => {
      const li = el("li", "cvid-tl");
      li.appendChild(el("div", "cvid-tl-role", w.title || ""));
      const sub = [w.company, w.period].filter(Boolean).join(" · ");
      if (sub) li.appendChild(el("div", "cvid-tl-sub", sub));
      (w.highlights || []).slice(0, 2).forEach((h) => li.appendChild(el("div", "cvid-tl-point", h)));
      tl.appendChild(li);
    });
    b.appendChild(tl); cvIdentityCard.appendChild(b);
  }
  // ── Projects ──
  if ((p.projects || []).length) {
    const b = block("פרויקטים");
    const list = el("ul", "cvid-projects");
    p.projects.slice(0, 5).forEach((pr) => {
      const li = el("li", "cvid-project");
      li.appendChild(el("span", "cvid-project-name", pr.name || ""));
      if (pr.description) li.appendChild(el("span", "cvid-project-desc", pr.description));
      list.appendChild(li);
    });
    b.appendChild(list); cvIdentityCard.appendChild(b);
  }
}

/** Inline editor for the interview-captured fields: name, role, experience, location, skills.
 *  Writes straight back into cvState.profile so the next search uses the corrected values. */
function renderIdentityEdit(p) {
  cvIdentityCard.textContent = "";
  const el = cvEl;
  const draft = {
    name:      p.name || "",
    role:      (p.titles_held || [])[0] || p.title || "",
    seniority: p.seniority || "",
    years:     (p.years != null) ? String(p.years) : "",
    location:  p.location || [p.city, p.state].filter(Boolean).join(", "),
    skills:    (p.skills || []).map(String),
    // _src keeps the original entry so unknown fields survive a round-trip
    experience: (p.work_experience || []).map((w) => ({
      _src: w, title: w.title || "", company: w.company || "", period: w.period || "",
      highlights: (w.highlights || []).map(String),
    })),
    projects: (p.projects || []).map((pr) => ({ _src: pr, name: pr.name || "", description: pr.description || "" })),
  };

  // snapshots taken before editing — drive the proactive follow-up on save
  const beforeFull   = JSON.stringify(profileSnapshot(p, false));
  const beforeSearch = JSON.stringify(profileSnapshot(p, true));

  const card = el("div", "cvid-edit-card");
  card.appendChild(el("h3", "cvid-edit-title", "עריכת פרופיל"));

  const textInput = (val, ph) => { const i = el("input", "cvid-input"); i.type = "text"; i.value = val; if (ph) i.placeholder = ph; return i; };
  const field = (labelTxt, control) => {
    const f = el("label", "cvid-field");
    f.appendChild(el("span", "cvid-field-label", labelTxt));
    f.appendChild(control);
    return f;
  };

  const nameI = textInput(draft.name, "שם מלא");
  const roleI = textInput(draft.role, "תפקיד מבוקש");
  const locI  = textInput(draft.location, "מיקום");
  const yearsI = el("input", "cvid-input"); yearsI.type = "number"; yearsI.min = "0"; yearsI.max = "60"; yearsI.value = draft.years; yearsI.placeholder = "שנים";
  const senS = el("select", "cvid-input");
  [["", "—"]].concat(Object.entries(SENIORITY_HE)).forEach(([k, v]) => {
    const o = el("option", null, v); o.value = k; if (k === draft.seniority) o.selected = true; senS.appendChild(o);
  });

  card.appendChild(field("שם", nameI));
  card.appendChild(field("תפקיד", roleI));
  const expRow = el("div", "cvid-field-row");
  expRow.appendChild(field("רמת ניסיון", senS));
  expRow.appendChild(field("שנות ניסיון", yearsI));
  card.appendChild(expRow);
  card.appendChild(field("מיקום", locI));

  // skills: removable chips + add row
  const skF = el("div", "cvid-field");
  skF.appendChild(el("span", "cvid-field-label", "כישורים"));
  const chips = el("div", "cvid-skill-edit");
  skF.appendChild(chips);
  const renderChips = () => {
    chips.textContent = "";
    draft.skills.forEach((sk, idx) => {
      const chip = el("span", "cvid-skill-chip", sk);
      const x = el("button", "cvid-skill-x", "×");
      x.type = "button"; x.setAttribute("aria-label", `הסר ${sk}`);
      x.addEventListener("click", () => { draft.skills.splice(idx, 1); renderChips(); });
      chip.appendChild(x);
      chips.appendChild(chip);
    });
  };
  renderChips();
  const addRow = el("div", "cvid-skill-add");
  const skInput = textInput("", "הוסף כישור…");
  const addBtn = el("button", "cvid-skill-addbtn", "הוסף"); addBtn.type = "button";
  const addSkill = () => {
    const v = skInput.value.trim();
    if (v && !draft.skills.some((s) => s.toLowerCase() === v.toLowerCase())) { draft.skills.push(v); renderChips(); }
    skInput.value = ""; skInput.focus();
  };
  addBtn.addEventListener("click", addSkill);
  skInput.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); addSkill(); } });
  addRow.appendChild(skInput); addRow.appendChild(addBtn);
  skF.appendChild(addRow);
  card.appendChild(skF);

  // textarea control (shares .cvid-input styling)
  const textArea = (val, ph) => { const t = el("textarea", "cvid-input cvid-textarea"); t.rows = 2; t.value = val; if (ph) t.placeholder = ph; return t; };

  // generic repeatable-entry editor (experience, projects)
  const entryEditor = (labelTxt, list, addLabel, makeEntry, makeBlank) => {
    const f = el("div", "cvid-field");
    f.appendChild(el("span", "cvid-field-label", labelTxt));
    const box = el("div", "cvid-entries");
    f.appendChild(box);
    const render = () => {
      box.textContent = "";
      list.forEach((item, idx) => box.appendChild(makeEntry(item, () => { list.splice(idx, 1); render(); })));
    };
    render();
    const add = el("button", "cvid-entry-add", addLabel); add.type = "button";
    add.addEventListener("click", () => { list.push(makeBlank()); render(); });
    f.appendChild(add);
    card.appendChild(f);
  };

  // ── Experience entries ──
  entryEditor("ניסיון", draft.experience, "+ הוסף ניסיון",
    (w, remove) => {
      const entry = el("div", "cvid-entry");
      const head = el("div", "cvid-entry-head");
      const titleI = textInput(w.title, "תפקיד"); titleI.classList.add("cvid-entry-title");
      titleI.addEventListener("input", () => { w.title = titleI.value; });
      const del = el("button", "cvid-entry-del", "×"); del.type = "button"; del.setAttribute("aria-label", "הסר ניסיון");
      del.addEventListener("click", remove);
      head.appendChild(titleI); head.appendChild(del); entry.appendChild(head);
      const row = el("div", "cvid-field-row");
      const compI = textInput(w.company, "חברה"); compI.addEventListener("input", () => { w.company = compI.value; });
      const perI  = textInput(w.period, "תקופה");  perI.addEventListener("input", () => { w.period = perI.value; });
      row.appendChild(compI); row.appendChild(perI); entry.appendChild(row);
      const hl = textArea(w.highlights.join("\n"), "דגשים — שורה לכל אחד");
      hl.addEventListener("input", () => { w.highlights = hl.value.split("\n").map((s) => s.trim()).filter(Boolean); });
      entry.appendChild(hl);
      return entry;
    },
    () => ({ title: "", company: "", period: "", highlights: [] }));

  // ── Project entries ──
  entryEditor("פרויקטים", draft.projects, "+ הוסף פרויקט",
    (pr, remove) => {
      const entry = el("div", "cvid-entry");
      const head = el("div", "cvid-entry-head");
      const nameI = textInput(pr.name, "שם הפרויקט"); nameI.classList.add("cvid-entry-title");
      nameI.addEventListener("input", () => { pr.name = nameI.value; });
      const del = el("button", "cvid-entry-del", "×"); del.type = "button"; del.setAttribute("aria-label", "הסר פרויקט");
      del.addEventListener("click", remove);
      head.appendChild(nameI); head.appendChild(del); entry.appendChild(head);
      const desc = textArea(pr.description, "תיאור קצר");
      desc.addEventListener("input", () => { pr.description = desc.value; });
      entry.appendChild(desc);
      return entry;
    },
    () => ({ name: "", description: "" }));

  // actions
  const actions = el("div", "cvid-edit-actions");
  const cancel = el("button", "btn ghost", "ביטול"); cancel.type = "button";
  const save = el("button", "btn primary", "שמירה"); save.type = "button";
  cancel.addEventListener("click", () => { editingProfile = false; renderIdentity(cvState.profile); });
  save.addEventListener("click", () => {
    const prof = cvState.profile || (cvState.profile = {});
    prof.name = nameI.value.trim();
    const role = roleI.value.trim();
    prof.titles_held = role ? [role].concat((prof.titles_held || []).slice(1)) : [];
    prof.seniority = senS.value || undefined;
    const y = parseInt(yearsI.value, 10);
    prof.years = Number.isFinite(y) ? y : null;
    prof.location = locI.value.trim();
    prof.skills = draft.skills.slice();
    prof.work_experience = draft.experience
      .map((w) => ({ ...(w._src || {}), title: w.title.trim(), company: w.company.trim(), period: w.period.trim(), highlights: w.highlights.slice() }))
      .filter((w) => w.title || w.company || w.period || w.highlights.length);
    prof.projects = draft.projects
      .map((pr) => ({ ...(pr._src || {}), name: pr.name.trim(), description: pr.description.trim() }))
      .filter((pr) => pr.name || pr.description);
    const changed       = JSON.stringify(profileSnapshot(prof, false)) !== beforeFull;
    const searchChanged = JSON.stringify(profileSnapshot(prof, true))  !== beforeSearch;
    editingProfile = false;
    renderIdentity(cvState.profile);
    proactiveAfterEdit(changed, searchChanged);
  });
  actions.appendChild(cancel); actions.appendChild(save);
  card.appendChild(actions);

  cvIdentityCard.appendChild(card);
}

/** Comparable snapshot of a profile. searchOnly => just the fields that affect job search. */
function profileSnapshot(x, searchOnly) {
  const base = {
    role:      (x.titles_held || [])[0] || x.title || "",
    seniority: x.seniority || "",
    years:     (x.years != null) ? String(x.years) : "",
    location:  x.location || [x.city, x.state].filter(Boolean).join(", "),
    skills:    (x.skills || []).map((s) => String(s).toLowerCase()).sort(),
  };
  if (searchOnly) return base;
  return Object.assign(base, {
    name:       x.name || "",
    experience: (x.work_experience || []).map((w) => [w.title || "", w.company || "", w.period || "", (w.highlights || []).join("|")]),
    projects:   (x.projects || []).map((pr) => [pr.name || "", pr.description || ""]),
  });
}

/** Proactive agent follow-up after a manual profile edit — message + action tuned to the flow step. */
function proactiveAfterEdit(changed, searchChanged) {
  if (!changed) return;                          // nothing actually changed — stay quiet
  const src  = activeSrc;
  const role = (cvState.profile.titles_held || [])[0] || cvState.profile.title || "";
  let text = "", actionHtml = "";

  if (tailorState.active) {
    const job = (tailorState.job && tailorState.job.title) ? `״${tailorState.job.title}״` : "המשרה";
    text = `ראיתי שעדכנת את הפרופיל. שאמשיך להתאים את קורות החיים ל${job} לפי העדכון? כתוב לי מה לחדד.`;
  } else if (role) {
    const verb = searchChanged ? "לחפש מחדש" : "לחפש";
    text = searchChanged
      ? `ראיתי שעדכנת את הפרופיל. ${verb} משרות **${role}** לפי הקריטריונים החדשים?`
      : `עדכנתי את הפרופיל. ${verb} משרות **${role}** מתי שתרצה.`;
    const label = (searchChanged ? "חפש מחדש משרות " : "חפש משרות ") + role + " ↻";
    actionHtml = `<div class="proactive-actions"><button type="button" class="btn primary proactive-btn" data-role="${escapeHtml(role)}">${escapeHtml(label)}</button></div>`;
  } else {
    text = "עדכנתי את הפרופיל. איזה תפקיד שאחפש עבורך?";
  }

  hero.style.display = "none";
  addAgent(src, text, actionHtml);
  cvState.history.push({ role: "assistant", content: text });
}

// Tailoring mode: while active, the composer talks to /api/cv-tailor about one focused job.
const tailorState = { active: false, job: null, history: [], composed: false,
                      sectionIndex: 0, hasCv: false };

// "✨ שפר קו״ח למשרה זו" buttons are injected into dynamic job cards — delegate from the stage.
stage.addEventListener("click", (e) => {
  const btn = e.target.closest(".improve-btn");
  if (!btn || tailorState.active) return;
  try { enterTailoring(activeSrc, JSON.parse(btn.dataset.job)); } catch { /* malformed data-job */ }
});

// Proactive "search again" button from the post-edit follow-up.
stage.addEventListener("click", (e) => {
  const btn = e.target.closest(".proactive-btn");
  if (!btn) return;
  const role = btn.dataset.role || "";
  btn.disabled = true;
  if (role) runSearch(activeSrc, role, false);
});
// Last rendered matches per tab — used to resolve which job the user means ("the first", a title).
const lastMatches = { dataset: [], scrape: [] };

/** Pick the job object the user referred to (rank number, title, else the top match). */
function resolveJob(query, src) {
  const ms = lastMatches[src] || [];
  if (!ms.length) return null;
  const q = (query || "").trim().toLowerCase();
  const num = q.match(/\d+/);
  if (num) { const byRank = ms.find((m) => String(m.rank) === num[0]); if (byRank) return byRank; }
  if (q.length > 2) { const byTitle = ms.find((m) => (m.title || "").toLowerCase().includes(q)); if (byTitle) return byTitle; }
  return ms[0];
}
function jobForTailor(m) {
  return {
    title: m.title || "",
    company: m.company && m.company !== "nan" ? m.company : "",
    description: m.description || m.blurb || "",
    skills: m.skills || [],
  };
}

/** Render one tailor response: agent reply in the main chat + animated CV update in the editor. */
function handleTailorResponse(src, data) {
  settlePending();   // the user's reply that triggered this response confirms the prior change
  // A 500 returns {error:…} with no reply (e.g. the LLM call failed server-side — check the
  // app.py console; "credit balance too low"/429/529 are the usual culprits). Surface that
  // error instead of the misleading generic placeholder, so the real cause isn't hidden.
  const reply = data.reply || data.error || "אין תגובה מהשרת — בדקו את לוג השרת (python app.py).";
  addAgent(src, reply, "");
  tailorState.history.push({ role: "assistant", content: reply });
  if (data.proposed_cv) {
    renderCvChange(cvState.cv_text || "", data.proposed_cv);   // diff -> highlight -> typewriter
    cvState.cv_text = data.proposed_cv;
    cvDockBanner.hidden = false;
    tailorState.composed = true;   // a full rewrite now exists -> later turns are AMEND
  }
  if (typeof data.next_section_index === "number") tailorState.sectionIndex = data.next_section_index;
  const total = data.total_sections || 0;
  if (!tailorState.composed && total && tailorState.sectionIndex >= total) {
    autoComposeCv(src);            // all sections gathered -> write the full CV (no user bubble)
  }
  if (data.done) exitTailoring(src);
}

/** End-of-walk: ask the server to compose the full CV (no user message). */
async function autoComposeCv(src) {
  const t = addThinking(src);
  try {
    const data = await postJSON("/api/cv-tailor", {
      cv_text: getDockCv(), job: tailorState.job,
      history: tailorState.history, user_msg: "",   // full walk -> COMPOSE needs every fact
      has_cv: tailorState.hasCv, section_index: tailorState.sectionIndex });
    t.remove();
    handleTailorResponse(src, data);
  } catch { t.remove(); addAgent(src, "לא ניתן להתחבר לשרת.", ""); }
}

async function enterTailoring(src, job) {
  tailorState.active = true;
  tailorState.job = job;
  tailorState.history = [];
  tailorState.composed = false;
  tailorState.sectionIndex = 0;
  tailorState.hasCv = !!cvState.hasRealCv;
  pendingMark = null;
  setDockCv(cvState.cv_text || "");
  cvDockJob.textContent = [job.title, job.company].filter(Boolean).join(" · ");
  cvDockBanner.hidden = true;
  hideProfile();                       // focus the workspace: profile collapses while improving
  openEditor();                        // the CV improver panel opens on the LEFT
  promptEl.placeholder = "ענה לסוכן… (כתוב \"סיימתי\" לסיום)";
  // Proactive opener — the agent leads with the summary, then gaps.
  const t = addThinking(src);
  try {
    const data = await postJSON("/api/cv-tailor",
      { open: true, cv_text: getDockCv(), job,
        has_cv: tailorState.hasCv, section_index: 0 });
    t.remove();
    handleTailorResponse(src, data);
  } catch { t.remove(); addAgent(src, "לא ניתן להתחבר לשרת.", ""); }
}

function exitTailoring(src) {
  tailorState.active = false;
  flushTypewriter();   // finish any in-flight rewrite so the editor holds the FULL CV before we read it
  settlePending();
  if (getDockCv().trim()) cvState.cv_text = getDockCv();   // keep edits
  closeEditor();                       // close the LEFT editor; profile (right) stays open
  renderIdentity(cvState.profile);     // refresh the identity card with the tailored CV
  openProfile();
  promptEl.placeholder = PLACEHOLDERS[src] || PLACEHOLDERS.dataset;

  // ── Auto-save the tailored CV and inform the user with a download link ──
  const jobTitle = (tailorState.job && tailorState.job.title) || "משרה";
  const company  = (tailorState.job && tailorState.job.company) || "";
  const finalCv  = cvState.cv_text || "";
  if (finalCv.trim()) {
    savedCvLib.addEntry(jobTitle, company, finalCv);

    // Render a markdown bubble; the download link is injected as a real DOM <a>,
    // not through innerHTML, to keep CV content out of the HTML injection path.
    const labelParts = [jobTitle, company].filter(Boolean);
    const label = labelParts.join(" · ");
    const mdText = `שמרתי גרסה מותאמת ל-**${label}**. `;
    const w = document.createElement("div");
    w.className = "msg agent";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    const mdDiv = document.createElement("div");
    mdDiv.className = "md";
    mdDiv.innerHTML = mdSafe(mdText);     // mdSafe/DOMPurify — no CV text here, only job label via markdown
    const dlLink = makeCvDownloadLink(jobTitle, finalCv, "להורדה ↓");
    bubble.appendChild(mdDiv);
    bubble.appendChild(dlLink);
    w.appendChild(bubble);
    threads[src].appendChild(w);
    scrollDown();
  }
}

/** One tailoring turn from the main composer (routed instead of search). */
async function submitTailorTurn(src, prompt) {
  addUser(src, prompt);
  tailorState.history.push({ role: "user", content: prompt });
  clearPrompt(); const signal = beginRequest();
  const t = addThinking(src);
  try {
    const data = await postJSON("/api/cv-tailor", {
      cv_text: getDockCv(), job: tailorState.job,
      history: tailorState.history.slice(0, -1), user_msg: prompt,
      composed: tailorState.composed,
      has_cv: tailorState.hasCv, section_index: tailorState.sectionIndex,
    }, signal);
    t.remove();
    handleTailorResponse(src, data);
  } catch (err) { t.remove(); if (!isAbort(err)) addAgent(src, "לא ניתן להתחבר לשרת. ודאו ש-`python app.py` רץ.", ""); }
  finally { endRequest(); promptEl.focus(); }
}

/* =====================================================================
   PROFILE INTERVIEW — build a profile by answering 4 questions (no CV).
   The vertical stepper lives inside the right-hand profile panel.
   ===================================================================== */
const INTERVIEW_STEPS = [
  { key: "role", label: "תפקיד" }, { key: "experience", label: "ניסיון" },
  { key: "location", label: "מיקום" }, { key: "skills", label: "כישורים" },
];

/** Is a stepper field already filled in the profile? */
function fieldFilled(p, key) {
  if (!p) return false;
  if (key === "role") return (p.titles_held || []).length > 0;
  if (key === "experience") return p.years != null || !!p.seniority;
  if (key === "location") return !!(p.city || p.state);
  if (key === "skills") return (p.skills || []).length > 0;
  return false;
}

/** Render the 4-node stepper from profile state. `nextField` (a step key or null)
 *  is the active node; filled fields show done, so editing one flips it back.
 *  Safe DOM only (textContent/createElement). */
function renderStepper(nextField) {
  cvStepper.hidden = false;
  cvStepper.textContent = "";
  const p = cvState.profile || {};
  cvStepper.classList.toggle("complete", !nextField);
  INTERVIEW_STEPS.forEach((s, i) => {
    const filled = fieldFilled(p, s.key);
    const state = filled ? "done" : (s.key === nextField ? "active" : "pending");
    const node = document.createElement("div");
    node.className = "cvs-node " + state;
    const glyph = document.createElement("div");
    glyph.className = "cvs-glyph";
    glyph.textContent = filled ? "✓" : String(i + 1);
    const body = document.createElement("div");
    body.className = "cvs-body";
    const label = document.createElement("div");
    label.className = "cvs-label";
    label.textContent = s.label;
    body.appendChild(label);                    // labels only — captured values live in the profile card
    node.appendChild(glyph);
    node.appendChild(body);
    cvStepper.appendChild(node);
  });
}

/** Start the no-CV interview: open the panel, show the stepper, ask question 1. */
async function startInterview() {
  if (onboardState.active) return;
  onboardState.active = true;
  onboardState.nextField = "role";
  onboardState.skipped = [];
  cvState.profile = {};
  cvState.hasRealCv = false;                 // built via interview -> tailoring builds from scratch
  hero.style.display = "none";
  cvIdentityCard.textContent = "";          // panel shows the stepper, card fills in progressively
  renderStepper("role");
  openProfile();
  const src = activeSrc;
  const signal = beginRequest();
  const t = addThinking(src);
  try {
    const data = await postJSON("/api/profile-build",
      { profile: {}, user_msg: "", history: [], us_only: src === "dataset" }, signal);
    t.remove();
    const reply = data.reply || data.question;
    addAgent(src, reply, "");
    cvState.history.push({ role: "assistant", content: reply });
  } catch (err) { t.remove(); if (!isAbort(err)) addAgent(src, "לא ניתן להתחבר לשרת. ודאו ש-`python app.py` רץ.", ""); }
  finally { endRequest(); }
}

/** Enter a (possibly pre-filled) interview from a /api/chat mode:"interview" response. */
function enterInterview(src, data) {
  onboardState.active   = true;
  onboardState.skipped  = [];
  // seeded interview: the next question's step index -> its field key (null when done)
  onboardState.nextField = Number.isInteger(data.next_step)
    ? (INTERVIEW_STEPS[data.next_step] || {}).key || null : null;
  cvState.profile       = data.profile || {};
  hero.style.display = "none";
  cvIdentityCard.textContent = "";
  renderStepper(onboardState.nextField);
  renderIdentity(cvState.profile);
  openProfile();
  if (data.reply) {
    addAgent(src, data.reply, "");
    cvState.history.push({ role: "assistant", content: data.reply });
  }
  if (data.done) {                 // everything was provided in one message
    finishInterview(src, data);
  } else if (data.question) {
    addAgent(src, data.question, "");
  }
}

/** One interview turn: send the answer, advance the stepper, ask the next question. */
async function handleInterviewAnswer(src, answer) {
  addUser(src, answer);
  cvState.history.push({ role: "user", content: answer });
  clearPrompt(); const signal = beginRequest();
  const t = addThinking(src);
  try {
    const data = await postJSON("/api/profile-build",
      { profile: cvState.profile || {}, user_msg: answer,
        history: cvState.history.slice(-8), skipped: onboardState.skipped || [],
        us_only: src === "dataset" }, signal);
    t.remove();
    cvState.profile = data.profile || cvState.profile;
    onboardState.nextField = data.next_field || null;
    onboardState.skipped = data.skipped || onboardState.skipped;
    renderIdentity(cvState.profile);          // progressively fill the identity card
    renderStepper(onboardState.nextField);
    if (data.reply) {
      addAgent(src, data.reply, "");
      cvState.history.push({ role: "assistant", content: data.reply });
    }
    if (data.done) finishInterview(src, data);
  } catch (err) { t.remove(); if (!isAbort(err)) addAgent(src, "לא ניתן להתחבר לשרת. ודאו ש-`python app.py` רץ.", ""); }
  finally { endRequest(); promptEl.focus(); }
}

/** Finalize: built profile + synthesized cv_text seed cvState exactly like the CV path. */
function finishInterview(src, data) {
  onboardState.active = false;
  cvState.cv_text = data.cv_text || "";
  cvState.profile = data.profile;
  cvState.history = [];
  cvState.roles   = [];
  cvState.searchCriteria = null;
  cvState.hasRealCv = false;                  // interview stub -> tailoring builds from scratch
  renderStepper(null);                        // all nodes done -> compact summary
  renderIdentity(cvState.profile);
  cvUploadLabel.classList.add("loaded");
  cvClipText.textContent = "✓ פרופיל";
  const role = (cvState.profile.titles_held || [])[0] || "";
  if (role) {
    const greeting = `מצוין — בניתי לך פרופיל. מחפש עכשיו משרות ${role} שמתאימות לך.`;
    addAgent(src, greeting, "");
    cvState.history.push({ role: "assistant", content: greeting });
    runSearch(src, role, false);     // auto-run the ranked search (no user bubble)
  } else {
    const greeting = "מצוין — בניתי לך פרופיל. איזה תפקיד שאחפש עבורך?";
    addAgent(src, greeting, "");
    cvState.history.push({ role: "assistant", content: greeting });
  }
}


/* ---------- init ---------- */
(async () => {
  try {
    const h = await (await fetch("/api/health")).json();
    if (h.ok) setStatus("live", h.llm ? `מחובר · ${h.jobs.toLocaleString()} משרות` : "מחובר · ללא מפתח LLM");
  } catch { setStatus("", "מנותק"); }
})();
