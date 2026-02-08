
    const DB_URL = "spec_evolution_v1.sqlite3";
    const BUCKETS = [
        { id: 1, name: "Logic", color: "#39ff14" }, { id: 2, name: "SQLite", color: "#adff2f" },
        { id: 3, name: "Async", color: "#00ff00" }, { id: 4, name: "Arch", color: "#ff003c" },
        { id: 5, name: "Meta", color: "#666" }, { id: 6, name: "Context", color: "#d946ef" },
        { id: 7, name: "Eng", color: "#00f2ff" }, { id: 8, name: "Math", color: "#bf00ff" },
        { id: 9, name: "Clarify", color: "#00ffcc" }, { id: 10, name: "Other", color: "#475569" }
    ];

    let DB = null;
    let COMMITS = [];
    let FILTERED = []; 
    let GROUPS = new Map();
    let DOC_CACHE = new Map();
    let PATCH_CACHE = new Map();
    let METRICS = [];
    let BASE_DOC = "";
    
    const STATE = { idx: 0, q: "", tab: "spec", isPlaying: false, playInterval: null, timeBucket: "day" };

    async function init() {
        try {
            const SQL = await window.initSqlJs({ locateFile: file => `https://cdn.jsdelivr.net/npm/sql.js@1.10.3/dist/${file}` });
            const res = await fetch(DB_URL);
            if (!res.ok) throw new Error("Neural Core Data Fetch Failed");
            const buf = await res.arrayBuffer();
            DB = new SQL.Database(new Uint8Array(buf));
            
            const baseRes = DB.exec("SELECT text FROM base_doc LIMIT 1");
            if (baseRes.length) BASE_DOC = baseRes[0].values[0][0];
            
            loadCommits();
            loadGroups();
            setupEventListeners();
            syncNeuralFilters();
            await computeSynapticMetrics();
            renderCommitList();
            drawSynapticMap();
            
            if (COMMITS.length > 0) {
                await selectCommit(COMMITS.length - 1);
            }
            hideLoader();
        } catch (e) {
            console.error(e);
            document.getElementById("loadingOverlay").innerHTML = `<div style="color:var(--accent); font-family:var(--font-mono); font-size:12px; text-align:center; padding:20px;">REANIMATION_FAILURE:<br>${e.message}</div>`;
        }
    }

    function loadCommits() {
        const res = DB.exec("SELECT idx, hash, short, date_iso, author, subject, add_lines, del_lines, impact, primary_bucket, labels_json FROM commits ORDER BY idx ASC");
        if (!res.length) return;
        COMMITS = res[0].values.map(r => ({ 
            idx: r[0], hash: r[1], short: r[2], date: r[3], author: r[4], 
            subject: r[5], add: r[6], del: r[7], impact: r[8], 
            primary: r[9], labels: JSON.parse(r[10]) 
        }));
        document.getElementById("commitSlider").max = COMMITS.length - 1;
        document.getElementById("kpiCommits").textContent = COMMITS.length;
    }

    function loadGroups() {
        const res = DB.exec("SELECT commit_hash, labels_json FROM change_groups");
        if (!res.length) return;
        res[0].values.forEach(r => {
            const hash = r[0];
            const labels = JSON.parse(r[1]);
            if (!GROUPS.has(hash)) GROUPS.set(hash, []);
            GROUPS.get(hash).push(...labels);
        });
    }

    async function computeSynapticMetrics() {
        let lines = BASE_DOC.split("
");
        let prevText = BASE_DOC;
        METRICS = [];
        for (let i = 0; i < COMMITS.length; i++) {
            const p = await getPatch(i);
            if (p) lines = applySynapticPatch(lines, p);
            const text = lines.join("
");
            const tokens = text.split(/\s+/).length;
            // Use commit impact as a proxy for volatility to avoid expensive Levenshtein calc
            const lev = (COMMITS[i] && COMMITS[i].impact) ? COMMITS[i].impact : 0; 
            METRICS.push({ lines: lines.length, tokens, lev });
            prevText = text;
            if (i % 10 === 0) DOC_CACHE.set(i, text);
        }
    }

    function synapticDistance(a, b) {
        const wa = a.split(/\s+/), wb = b.split(/\s+/);
        const m = wa.length, n = wb.length;
        if (m === 0) return n; if (n === 0) return m;
        let prev = Array.from({length: n + 1}, (_, i) => i);
        for (let i = 1; i <= m; i++) {
            let curr = [i];
            for (let j = 1; j <= n; j++) {
                curr[j] = wa[i-1] === wb[j-1] ? prev[j-1] : Math.min(prev[j-1], prev[j], curr[j-1]) + 1;
            }
            prev = curr;
        }
        return prev[n];
    }

    window.selectCommit = async function(idx) {
        STATE.idx = Math.max(0, Math.min(COMMITS.length - 1, idx));
        document.getElementById("commitSlider").value = STATE.idx;
        const c = COMMITS[STATE.idx];
        const m = METRICS[STATE.idx];
        
        const dockSubject = document.getElementById("dockSubject");
        dockSubject.textContent = c.subject;
        if(!STATE.isPlaying) decodeText(dockSubject, c.subject, 0.8);
        
        document.getElementById("dockDetails").textContent = `HASH: ${c.short} // BY: ${c.author} // TIME: ${dayjs(c.date).format("MMM DD HH:mm")}`;
        document.getElementById("activeLabel").textContent = `LOG_ENTRY_${STATE.idx} // ${c.short}`;
        
        const ml = document.getElementById("metricLines");
        const mt = document.getElementById("metricTokens");
        const mev = document.getElementById("metricLev");
        const md = document.getElementById("metricDensity");
        if(ml) ml.textContent = m.lines.toLocaleString();
        if(mt) mt.textContent = m.tokens.toLocaleString();
        if(mev) mev.textContent = m.lev.toLocaleString();
        if(md) md.textContent = (m.tokens / m.lines).toFixed(2);

        renderDNA(c);
        renderCommitList();
        await updateView();
        
        const sel = document.querySelector(".commit-card.selected");
        if (sel) sel.scrollIntoView({ block: "nearest", behavior: "smooth" });
    };

    function renderDNA(c) {
        const dna = document.getElementById("dnaBar");
        dna.innerHTML = "";
        
        // Brain Glow logic
        document.querySelectorAll('.brain-glow').forEach(el => el.style.opacity = '0');
        const primary = c.primary;
        if([1, 8, 9].includes(primary)) document.getElementById('brainLogic').style.opacity = '1';
        if([7, 3, 2].includes(primary)) document.getElementById('brainEng').style.opacity = '1';
        if([4, 6, 5].includes(primary)) document.getElementById('brainArch').style.opacity = '1';

        const labels = GROUPS.get(c.hash) || c.labels;
        if (!labels.length) {
            const s = document.createElement("div"); s.className="dna-seg"; s.style.width="100%"; s.style.background=BUCKETS[9].color;
            dna.appendChild(s); return;
        }
        const counts = {}; labels.forEach(l => counts[l] = (counts[l] || 0) + 1);
        const total = labels.length;
        Object.keys(counts).forEach(l => {
            const bucket = BUCKETS.find(b => b.id === Number(l)) || BUCKETS[9];
            const s = document.createElement("div"); s.className="dna-seg";
            s.style.width = (counts[l] / total * 100) + "%";
            s.style.background = bucket.color;
            dna.appendChild(s);
        });
    }

    async function updateView() {
        if (STATE.tab === "spec") await renderSpec();
        else if (STATE.tab === "timeline") renderCharts();
        else if (STATE.tab === "diff") await renderDiff();
    }

    async function renderSpec() {
        const el = document.getElementById("specContent");
        el.innerHTML = '<div style="color: var(--primary); font-family: var(--font-mono); font-size: 11px;">RECONSTRUCTING_TISSUE...</div>';
        const text = await reconstructSnapshot(STATE.idx);
        const md = window.markdownit({ html: true, highlight: (str, lang) => (lang && hljs.getLanguage(lang)) ? hljs.highlight(str, { language: lang }).value : '' });
        el.innerHTML = DOMPurify.sanitize(md.render(text));
        document.getElementById("kpiLines").textContent = text.split("
").length.toLocaleString();
    }

    async function reconstructSnapshot(idx) {
        if (DOC_CACHE.has(idx)) return DOC_CACHE.get(idx);
        let startIdx = 0;
        let lines = BASE_DOC.split("
");
        for (let i = Math.floor(idx / 10) * 10; i >= 0; i -= 10) {
            if (DOC_CACHE.has(i)) {
                startIdx = i + 1;
                lines = DOC_CACHE.get(i).split("
");
                break;
            }
        }
        for (let i = startIdx; i <= idx; i++) {
            const p = await getPatch(i);
            if (p) lines = applySynapticPatch(lines, p);
            if (i % 10 === 0 && !DOC_CACHE.has(i)) DOC_CACHE.set(i, lines.join("
"));
        }
        const res = lines.join("
");
        DOC_CACHE.set(idx, res);
        return res;
    }

    function applySynapticPatch(lines, patch) {
        const out = [...lines];
        const patchLines = patch.split("
");
        let offset = 0;
        for (let i = 0; i < patchLines.length; i++) {
            const line = patchLines[i];
            if (line.startsWith("@@")) {
                const m = line.match(/@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/);
                if (m) {
                    const os = parseInt(m[1]), oc = parseInt(m[2] || "1");
                    const startPos = os - 1 + offset;
                    const newSeg = [];
                    i++;
                    while (i < patchLines.length && !patchLines[i].startsWith("@@")) {
                        const row = patchLines[i];
                        if (row.startsWith("+")) newSeg.push(row.slice(1));
                        else if (!row.startsWith("-")) newSeg.push(row.slice(1));
                        i++;
                    }
                    i--;
                    // Avoid stack overflow with spread operator for large patches
                    // out.splice(startPos, oc, ...newSeg);
                    
                    // Batched splice implementation
                    const CHUNK_SIZE = 10000;
                    if (newSeg.length > CHUNK_SIZE) {
                         // Remove the old segment first
                         out.splice(startPos, oc);
                         // Insert new segment in chunks
                         for (let j = 0; j < newSeg.length; j += CHUNK_SIZE) {
                             const chunk = newSeg.slice(j, j + CHUNK_SIZE);
                             out.splice(startPos + j, 0, ...chunk);
                         }
                    } else {
                         out.splice(startPos, oc, ...newSeg);
                    }
                    
                    offset += (parseInt(m[4] || "1") - oc);
                }
            }
        }
        return out;
    }

    async function getPatch(idx) {
        if (PATCH_CACHE.has(idx)) return PATCH_CACHE.get(idx);
        const res = DB.exec("SELECT patch FROM patches WHERE idx = ?", [idx]);
        const p = res.length ? res[0].values[0][0] : "";
        PATCH_CACHE.set(idx, p); return p;
    }

    async function renderDiff() {
        const el = document.getElementById("diffContent");
        const p = await getPatch(STATE.idx);
        if (!p) { el.innerHTML = "SYSTEM_STABLE: NO_DELTA"; return; }
        const diffHtml = Diff2Html.html(p, { drawFileList: false, matching: 'lines', outputFormat: 'line-by-line', colorScheme: 'dark' });
        el.innerHTML = DOMPurify.sanitize(diffHtml);
    }

    function drawSynapticMap() {
        const cvs = document.getElementById("mapCanvas");
        const ctx = cvs.getContext("2d");
        const w = cvs.width = cvs.clientWidth;
        const h = cvs.height = cvs.clientHeight;
        if (!w || !h || COMMITS.length === 0) return;
        ctx.clearRect(0, 0, w, h);
        const barW = w / COMMITS.length;
        COMMITS.forEach((c, i) => {
            const labels = GROUPS.get(c.hash) || c.labels;
            const buckets = labels.length ? labels : [10];
            const counts = {}; buckets.forEach(l => counts[l] = (counts[l] || 0) + 1);
            const total = buckets.length;
            let cy = 0;
            Object.keys(counts).forEach(l => {
                const b = BUCKETS.find(x => x.id === Number(l)) || BUCKETS[9];
                const sh = (counts[l] / total) * h;
                ctx.fillStyle = b.color; ctx.globalAlpha = 0.6;
                ctx.fillRect(i * barW, cy, barW - 1, sh);
                cy += sh;
            });
            if (c.impact > 250) { ctx.fillStyle = "#fff"; ctx.globalAlpha = 0.2; ctx.fillRect(i * barW, 0, barW - 1, h); }
        });
    }

    let charts = {};
    function renderCharts() {
        const t = { backgroundColor: 'transparent', textStyle: { fontFamily: 'JetBrains Mono', color: '#4a7a4a' }, grid: { top: 30, bottom: 30, left: 40, right: 20 }, xAxis: { axisLine: { lineStyle: { color: '#1a221a' } }, splitLine: { show: false } }, yAxis: { axisLine: { lineStyle: { color: '#1a221a' } }, splitLine: { lineStyle: { color: '#111', type: 'dashed' } } } };
        if (!charts.velocity) charts.velocity = echarts.init(document.getElementById("chartVelocity"));
        charts.velocity.setOption({ ...t, series: [{ data: COMMITS.map(c => [c.date, c.impact]), type: 'line', smooth: true, lineStyle: { color: '#39ff14' }, areaStyle: { color: 'rgba(57, 255, 20, 0.1)' } }] });
        if (!charts.dist) charts.dist = echarts.init(document.getElementById("chartDistribution"));
        const counts = {}; COMMITS.forEach(c => counts[c.primary] = (counts[c.primary] || 0) + 1);
        charts.dist.setOption({ ...t, series: [{ type: 'pie', radius: ['40%', '70%'], data: BUCKETS.map(b => ({ value: counts[b.id] || 0, name: b.name, itemStyle: { color: b.color } })) }] });
        renderMassGrowth();
    }

    function renderMassGrowth() {
        if (!charts.mass) charts.mass = echarts.init(document.getElementById("chartMass"));
        const buckets = groupCommits(STATE.timeBucket);
        const labels = Object.keys(buckets).sort();
        const series = BUCKETS.map(b => ({ name: b.name, type: 'bar', stack: 'total', itemStyle: { color: b.color }, data: labels.map(l => buckets[l][b.id] || 0) }));
        const t = { backgroundColor: 'transparent', textStyle: { fontFamily: 'JetBrains Mono', color: '#4a7a4a' }, grid: { top: 30, bottom: 30, left: 40, right: 20 }, xAxis: { axisLine: { lineStyle: { color: '#1a221a' } } }, yAxis: { axisLine: { lineStyle: { color: '#1a221a' } }, splitLine: { lineStyle: { color: '#111' } } } };
        charts.mass.setOption({ ...t, tooltip: { trigger: 'axis' }, xAxis: { type: 'category', data: labels }, series }, true);
    }

    function groupCommits(unit) {
        const res = {};
        COMMITS.forEach(c => {
            let k; const d = dayjs(c.date);
            if (unit === "day") k = d.format("YYYY-MM-DD");
            else if (unit === "hour") k = d.format("MM-DD HH:00");
            else if (unit === "15m") k = d.format("MM-DD HH:") + (Math.floor(d.minute()/15)*15).toString().padStart(2,'0');
            else k = d.format("MM-DD HH:") + (Math.floor(d.minute()/5)*5).toString().padStart(2,'0');
            if (!res[k]) res[k] = {};
            res[k][c.primary] = (res[k][c.primary] || 0) + c.impact;
        });
        return res;
    }

    function setupEventListeners() {
        document.getElementById("searchInput").addEventListener("input", e => { STATE.q = e.target.value; syncNeuralFilters(); renderCommitList(); });
        document.getElementById("commitSlider").addEventListener("input", e => selectCommit(parseInt(e.target.value)));
        
        ["Spec", "Timeline", "Diff"].forEach(t => {
            const btn = document.getElementById("tab" + t);
            btn.addEventListener("click", e => {
                ["Spec", "Timeline", "Diff"].forEach(tt => { document.getElementById("tab" + tt).classList.remove("active"); document.getElementById("view" + tt).classList.add("hidden"); });
                e.target.classList.add("active"); document.getElementById("view" + t).classList.remove("hidden");
                
                // Kinetic title reveal
                const titleMap = { "Spec": "THE_LEDGER", "Timeline": "DIAGNOSTICS", "Diff": "DELTA" };
                decodeText(btn, titleMap[t], 0.5);
                
                STATE.tab = t.toLowerCase(); updateView();
            });
        });

        ["btnDay", "btnHour", "btn15m", "btn5m"].forEach(id => {
            const el = document.getElementById(id);
            if(el) el.addEventListener("click", () => {
                STATE.timeBucket = id.replace("btn", "").toLowerCase();
                renderMassGrowth();
            });
        });

        document.getElementById("btnOpenLogs").addEventListener("click", () => document.getElementById("sidebar").classList.add("active"));
        document.getElementById("btnCloseLogs").addEventListener("click", () => document.getElementById("sidebar").classList.remove("active"));
        
        document.getElementById("btnPlay").addEventListener("click", () => {
            STATE.isPlaying = !STATE.isPlaying;
            document.getElementById("btnPlay").textContent = STATE.isPlaying ? "STABILIZE" : "REANIMATE";
            
            if (STATE.isPlaying) {
                document.body.style.animation = 'shake 0.5s infinite';
                STATE.playInterval = setInterval(() => { 
                    if (STATE.idx < COMMITS.length - 1) {
                        selectCommit(STATE.idx + 1);
                        if(Math.random() > 0.5) createElectricArc();
                    } else { 
                        clearInterval(STATE.playInterval); 
                        STATE.isPlaying = false; 
                        document.getElementById("btnPlay").textContent = "REANIMATE"; 
                        document.body.style.animation = 'none';
                    } 
                }, 200);
            } else { 
                clearInterval(STATE.playInterval); 
                document.body.style.animation = 'none';
            }
        });

        document.getElementById("btnPrev").addEventListener("click", () => selectCommit(STATE.idx - 1));
        document.getElementById("btnNext").addEventListener("click", () => selectCommit(STATE.idx + 1));
        window.addEventListener("resize", () => { drawSynapticMap(); Object.values(charts).forEach(c => c.resize()); });
    }

    function syncNeuralFilters() { const q = STATE.q.toLowerCase(); FILTERED = COMMITS.filter(c => !q || c.subject.toLowerCase().includes(q) || c.hash.includes(q)); }
    
    function renderCommitList() {
        const container = document.getElementById("commitList");
        container.innerHTML = FILTERED.slice().reverse().map(c => {
            const b = BUCKETS.find(x => x.id === c.primary) || BUCKETS[9];
            const sel = c.idx === STATE.idx ? "selected" : "";
            return `<div class="commit-card ${sel}" onclick="window.selectCommit(${c.idx}); document.getElementById('sidebar').classList.remove('active');"><div class="commit-card-header"><span>ENT_${c.idx}</span><span style="color: var(--fg-dim)">${dayjs(c.date).format("MMM DD")}</span></div><div class="commit-card-subject">${escapeHtml(c.subject)}</div><div class="commit-card-meta"><span style="color:${b.color}">${b.name}</span><span style="color:var(--primary)">+${c.add}</span></div></div>`;
        }).join("");
    }

    function hideLoader() {
        document.getElementById("loadingOverlay").style.opacity = '0';
        setTimeout(() => {
            document.getElementById("loadingOverlay").classList.add("hidden");
            // Start kinetic reveals
            decodeText(document.querySelector('.brand h1'), "FrankenSQLite // Evolutionary_DNA", 1.5);
            decodeText(document.querySelector('.sidebar-header span'), "LABORATORY_LOGS", 1.2, 0.5);
            startLabEffects();
        }, 500);
    }

    // --- Kinetic & Lab Effects ---

    const CHARS = "01$!@#%^&*()_+{}:<>?[]|";

    function decodeText(element, targetText, duration = 1, delay = 0) {
        if (!element || !targetText) return;
        const dur = Math.max(0.01, duration);
        setTimeout(() => {
            let iteration = 0;
            const maxIterations = targetText.length;
            const startTime = Date.now();
            element.innerText = "";
            const interval = setInterval(() => {
                const elapsed = (Date.now() - startTime) / 1000;
                iteration = (elapsed / dur) * maxIterations;
                element.innerText = targetText.split("").map((char, index) => {
                    if (index < iteration) return targetText[index];
                    if (char === " ") return " ";
                    return CHARS[Math.floor(Math.random() * CHARS.length)];
                }).join("");
                if (iteration >= maxIterations) { clearInterval(interval); element.innerText = targetText; }
            }, 30);
        }, delay * 1000);
    }

    function startLabEffects() {
        function packetLoop() {
            if (document.visibilityState === 'visible' && Math.random() > 0.98) { createDataPacket(); }
            requestAnimationFrame(packetLoop);
        }
        packetLoop();
        setInterval(updateSystemHud, 1000);
        setInterval(maybeSpark, 3000);
        const flashlight = document.getElementById('flashlight');
        document.addEventListener('mousemove', (e) => {
            let x = e.clientX, y = e.clientY;
            const targets = document.querySelectorAll('button, .commit-card, .tab');
            let nearestDist = 150;
            targets.forEach(t => {
                const r = t.getBoundingClientRect();
                const tx = r.left + r.width/2, ty = r.top + r.height/2;
                const d = Math.hypot(x - tx, y - ty);
                if (d < nearestDist) { x = x * 0.85 + tx * 0.15; y = y * 0.85 + ty * 0.15; }
            });
            if(flashlight) flashlight.style.transform = `translate(${x}px, ${y}px) translate(-50%, -50%)`;
        });
    }

    function updateSystemHud() {
        const isReanimating = STATE.isPlaying;
        const load = isReanimating ? (80 + Math.random() * 20) : (5 + Math.random() * 15);
        const integrity = isReanimating ? (90 + Math.random() * 5) : (99 + Math.random() * 1);
        const temp = (isReanimating ? (42 + Math.random() * 5) : (36 + Math.random() * 0.5)).toFixed(1);
        const voltage = isReanimating ? (400 + Math.floor(Math.random() * 100)) : (230 + Math.floor(Math.random() * 10));

        const shift = (load / 100) * 2.5;
        const offR = document.querySelector('#crt-filter feOffset[result="red-chan"]');
        const offB = document.querySelector('#crt-filter feOffset[result="blue-chan"]');
        if(offR) offR.setAttribute('dx', shift);
        if(offB) offB.setAttribute('dx', -shift);

        const hl = document.getElementById('hudLoad'), hi = document.getElementById('hudIntegrity');
        if(hl) hl.style.width = load + '%'; if(hi) hi.style.width = integrity + '%';
        const ht = document.getElementById('hudTemp'), hv = document.getElementById('hudVoltage');
        if(ht) ht.innerText = temp + '°C'; if(hv) hv.innerText = voltage + 'V';
        const syncs = isReanimating ? ["STRESS", "CRITICAL", "REANIMATING", "OVERLOAD"] : ["STABLE", "SYNCING", "IDLE", "CALIBRATED"];
        const hs = document.getElementById('hudSync');
        if(hs) {
            hs.innerText = syncs[Math.floor(Math.random() * syncs.length)];
            hs.style.color = isReanimating ? 'var(--accent)' : 'var(--electric)';
        }
    }

    function maybeSpark() { const threshold = STATE.isPlaying ? 0.3 : 0.8; if (Math.random() > threshold) createElectricArc(); }

    function createElectricArc() {
        const path = document.getElementById('arcPath');
        const w = window.innerWidth, h = window.innerHeight;
        const points = [{x: 10, y: 10}, {x: w-10, y: 10}, {x: 10, y: h-10}, {x: w-10, y: h-10}];
        const p1 = points[Math.floor(Math.random() * points.length)];
        let p2 = points[Math.floor(Math.random() * points.length)];
        while(p1 === p2) p2 = points[Math.floor(Math.random() * points.length)];
        let d = `M ${p1.x} ${p1.y}`;
        const segments = 12;
        for(let i=1; i<segments; i++) {
            const tx = p1.x + (p2.x - p1.x) * (i / segments), ty = p1.y + (p2.y - p1.y) * (i / segments);
            d += ` L ${tx + (Math.random() - 0.5) * 120} ${ty + (Math.random() - 0.5) * 120}`;
        }
        d += ` L ${p2.x} ${p2.y}`;
        path.setAttribute('d', d); path.style.animation = 'none'; path.getBoundingClientRect(); 
        path.style.animation = 'spark 0.3s ease-out forwards';
    }

    function escapeHtml(text) { return text ? text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;") : ""; }

    init();
