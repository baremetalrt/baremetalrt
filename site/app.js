/* BareMetalRT — Web UI Application */

// Theme: dark only
document.documentElement.removeAttribute('data-theme');
localStorage.removeItem('baremetalrt-theme');

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let generating = false;
let currentMode = 'tp2'; // TP is the live demo; 1 GPU coming soon
let cachedPing = null;
let maintenance = false; // set to false to re-enable demo
let conversationHistory = []; // [{role: "user"|"assistant", content: "..."}]

// ---------------------------------------------------------------------------
// Mode switcher
// ---------------------------------------------------------------------------

function setMode(mode) {
  currentMode = mode;
  localStorage.setItem('baremetalrt-mode', mode);

  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-mode') === mode);
  });

  const mesh = document.getElementById('mesh');
  if (mode === 'pp1') {
    mesh.classList.add('single-gpu');
  } else {
    mesh.classList.remove('single-gpu');
  }

  // Reset node displays
  document.getElementById('node0').classList.remove('active');
  document.getElementById('node1').classList.remove('active');
  document.getElementById('mesh').classList.remove('both-active');
  document.getElementById('node0-info').innerHTML = '<span class="wait">Waiting for node...</span>';
  document.getElementById('node1-info').innerHTML = '<span class="wait">Waiting for node...</span>';

  // Toggle welcome content
  const wTp = document.getElementById('welcome-tp');
  const wPp = document.getElementById('welcome-pp');
  if (wTp) wTp.style.display = mode === 'tp2' ? '' : 'none';
  if (wPp) wPp.style.display = mode === 'pp1' ? '' : 'none';

  // Restore input prompt hint when switching modes
  const inputPrompt = document.getElementById('input-prompt');
  if (inputPrompt) inputPrompt.style.display = '';

  refresh();
}

// Apply saved mode on load
(function() {
  const btn = document.querySelector('.mode-btn[data-mode="' + currentMode + '"]');
  if (btn) {
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  }
  if (currentMode === 'pp1') {
    document.getElementById('mesh').classList.add('single-gpu');
  }
  // Always sync welcome content to current mode
  const wTp = document.getElementById('welcome-tp');
  const wPp = document.getElementById('welcome-pp');
  if (wTp) wTp.style.display = currentMode === 'tp2' ? '' : 'none';
  if (wPp) wPp.style.display = currentMode === 'pp1' ? '' : 'none';
})();

// ---------------------------------------------------------------------------
// Mesh status
// ---------------------------------------------------------------------------

const _lastNode = {};
function renderNode(id, data) {
  if (!data) return;
  const key = data.hostname + '|' + data.gpu + '|' + data.ip + '|' + data.vram_mb + '|' + data.ping_ms;
  if (_lastNode[id] === key) return;
  _lastNode[id] = key;
  const vram = data.vram_mb ? '<span class="vram">' + Math.round(data.vram_mb / 1024) + 'GB</span>' : '';
  const ping = data.ping_ms ? '<div class="ping">PEER PING: ' + (Math.round(data.ping_ms / 10) * 10) + 'ms</div>' : '';
  // Split GPU name: "NVIDIA GeForce RTX 3060" → line1: "NVIDIA GeForce RTX", line2: "3060 · 8GB"
  let gpuLine1 = data.gpu || '';
  let gpuLine2 = '';
  const rtxIdx = gpuLine1.toLowerCase().indexOf('rtx');
  if (rtxIdx !== -1) {
    const afterRtx = rtxIdx + 3;
    gpuLine2 = gpuLine1.substring(afterRtx).trim();
    gpuLine1 = gpuLine1.substring(0, afterRtx).trim();
  }
  if (vram) gpuLine2 = gpuLine2 ? gpuLine2 + ' \u00b7 ' + vram : vram;
  const html =
    '<div class="host">' + data.hostname + '</div>' +
    '<div class="gpu">' + gpuLine1 + '</div>' +
    (gpuLine2 ? '<div class="gpu-model">' + gpuLine2 + '</div>' : '') +
    '<div class="ip">' + data.ip + '</div>' + ping;
  document.getElementById(id).innerHTML = html;
}

function formatEngineName(name) {
  if (!name) return '';
  // mistral-7b-tp2 → Mistral 7B
  const n = name.toLowerCase();
  if (n.includes('mistral') && n.includes('7b')) return 'Mistral 7B';
  if (n.includes('tinyllama')) return 'TinyLlama 1.1B';
  if (n.includes('llama') && n.includes('70b')) return 'Llama 70B';
  return name;
}

async function refresh() {
  try {
    const badge = document.getElementById('status-badge');
    const info = document.getElementById('model-info'); // may not exist on all pages

    if (currentMode === 'pp1') {
      // Solo / single GPU mode — orchestrator-routed via cluster
      const r = await fetch('/api/cluster');
      const d = await r.json();
      const soloNode = (d.nodes || []).find(n =>
        n.status === 'ready' && n.engine && n.engine.endsWith('-tp1')
      );

      if (soloNode) {
        badge.className = 'status-badge ready';
        badge.textContent = 'READY';
        const model = formatEngineName(soloNode.engine);
        const vramGb = soloNode.vram_mb ? Math.round(soloNode.vram_mb / 1024) + 'GB' : '';
        if (info) info.textContent = (model || 'TensorRT-LLM') + ' · Solo · ' + vramGb;
        renderNode('node0-info', soloNode);
        document.getElementById('node0').classList.add('active');
        document.getElementById('mesh').classList.add('single-gpu');
        document.getElementById('prompt').disabled = false;
        document.getElementById('prompt').placeholder = 'Enter prompt...';
        document.getElementById('send-btn').disabled = generating;
      } else {
        badge.className = 'status-badge starting';
        badge.textContent = 'WAITING';
        document.getElementById('node0-info').innerHTML = '<span class="wait">Waiting for node...</span>';
        document.getElementById('send-btn').disabled = true;
        document.getElementById('prompt').disabled = true;
      }
      return;
    }

    if (maintenance) {
      // Maintenance mode — demo offline
      badge.className = 'status-badge starting';
      badge.textContent = 'MAINTENANCE';
      document.getElementById('send-btn').disabled = true;
      document.getElementById('prompt').disabled = true;
      document.getElementById('prompt').placeholder = 'Demo temporarily offline — compute is reserved for GPU 1';
      return;
    }

    // TP=2 mode — fetch cluster status from orchestrator
    const r = await fetch('/api/cluster?mode=' + currentMode);
    const d = await r.json();
    const session = d.session;

    {
      document.getElementById('prompt').disabled = false;
      document.getElementById('prompt').placeholder = 'Enter prompt...';
      if (session && session.status === 'matched') {
        badge.className = 'status-badge ready';
        badge.textContent = 'READY';
        const model = formatEngineName(session.engine_name);
        const totalVram = d.total_vram_gb ? d.total_vram_gb + 'GB combined' : '';
        if (info) info.textContent = (model || 'TensorRT-LLM') + ' \u00b7 TP=2 over TCP \u00b7 ' + totalVram;
        renderNode('node0-info', session.rank0);
        renderNode('node1-info', session.rank1);
        document.getElementById('node0').classList.add('active');
        document.getElementById('node1').classList.add('active');
        document.getElementById('mesh').classList.add('both-active');

        // Fetch peer ping once, then cache it
        if (cachedPing === null) {
          try {
            const pingResp = await fetch('/api/ping');
            const pingData = await pingResp.json();
            if (pingData.ping_ms != null) cachedPing = Math.round(pingData.ping_ms / 10) * 10;
          } catch(e) {}
        }
        if (cachedPing !== null) {
          session.rank0.ping_ms = cachedPing;
          session.rank1.ping_ms = cachedPing;
          renderNode('node0-info', session.rank0);
          renderNode('node1-info', session.rank1);
        }
      } else {
        const online = d.nodes ? d.nodes.filter(n => n.status !== 'offline').length : 0;
        if (online > 0) {
          badge.className = 'status-badge registered';
          badge.textContent = online + ' NODE' + (online > 1 ? 'S' : '');
          const engines = d.nodes.filter(n => n.status !== 'offline' && n.engine).map(n => n.engine);
          const model = engines.length > 0 ? formatEngineName(engines[0]) : '';
          if (info) info.textContent = (model || 'TensorRT-LLM') + ' \u00b7 TP=2 \u00b7 Waiting for peer...';
          d.nodes.forEach(n => {
            if (n.status === 'offline') return;
            const elId = n.rank === 1 ? 'node1-info' : 'node0-info';
            const nodeEl = n.rank === 1 ? 'node1' : 'node0';
            renderNode(elId, { hostname: n.hostname, gpu: n.gpu, ip: n.ip, vram_mb: n.vram_mb });
            document.getElementById(nodeEl).classList.add('active');
          });
        } else {
          badge.className = 'status-badge starting';
          badge.textContent = 'WAITING';
          if (info) info.textContent = 'TensorRT-LLM 0.12 \u00b7 TP=2 \u00b7 Waiting for GPU nodes...';
        }
        }

      document.getElementById('send-btn').disabled =
        !(session && session.status === 'matched') || generating;
    }
  } catch(e) {}
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

function newChat() {
  conversationHistory = [];
  document.getElementById('messages').innerHTML = '';
  document.getElementById('mesh').classList.remove('chat-active');
  document.getElementById('prompt').focus();
}

function addMsg(role, text) {
  const w = document.getElementById('welcome');
  if (w) w.remove();
  const el = document.createElement('div');
  el.className = 'msg ' + role;
  if (role === 'assistant' && text) {
    el.innerHTML = renderMarkdown(text);
  } else {
    el.textContent = text;
  }
  document.getElementById('messages').appendChild(el);
  el.scrollIntoView({ behavior: 'smooth' });
  return el;
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function renderMarkdown(text) {
  // Render LaTeX before escaping HTML
  if (typeof katex !== 'undefined') {
    text = text.replace(/\\\[([\s\S]*?)\\\]/g, (_, expr) => {
      try { return katex.renderToString(expr.trim(), { displayMode: true }); } catch(e) { return _; }
    });
    text = text.replace(/\$\$([\s\S]*?)\$\$/g, (_, expr) => {
      try { return katex.renderToString(expr.trim(), { displayMode: true }); } catch(e) { return _; }
    });
    text = text.replace(/\\\(([\s\S]*?)\\\)/g, (_, expr) => {
      try { return katex.renderToString(expr.trim(), { displayMode: false }); } catch(e) { return _; }
    });
    text = text.replace(/\$([^\$\n]+?)\$/g, (_, expr) => {
      try { return katex.renderToString(expr.trim(), { displayMode: false }); } catch(e) { return _; }
    });
  }
  // Preserve KaTeX HTML through the escape step
  const katexBlocks = [];
  text = text.replace(/<span class="katex[\s\S]*?<\/span><\/span>/g, (m) => {
    katexBlocks.push(m);
    return `__KATEX_${katexBlocks.length - 1}__`;
  });
  let html = esc(text);
  html = html.replace(/__KATEX_(\d+)__/g, (_, i) => katexBlocks[parseInt(i)]);
  // Code blocks: ```lang\n...\n```
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
    const id = 'cb-' + Math.random().toString(36).slice(2, 8);
    return `<pre><code class="lang-${lang || 'text'}">${code.trim()}</code><button class="copy-btn" onclick="copyCode('${id}')">Copy</button><span id="${id}" style="display:none">${code.trim()}</span></pre>`;
  });
  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Headers
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
  // Unordered lists
  html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');
  // Ordered lists
  html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
  // Tables: | col | col |
  html = html.replace(/((?:^\|.+\|$\n?)+)/gm, (table) => {
    const rows = table.trim().split('\n').filter(r => r.trim());
    if (rows.length < 2) return table;
    let out = '<table style="border-collapse:collapse;margin:8px 0;font-size:13px;">';
    rows.forEach((row, i) => {
      if (row.match(/^\|[\s-:|]+\|$/)) return;
      const cells = row.split('|').filter(c => c.trim());
      const tag = i === 0 ? 'th' : 'td';
      out += '<tr>' + cells.map(c => `<${tag} style="border:1px solid var(--border);padding:4px 8px;">${c.trim()}</${tag}>`).join('') + '</tr>';
    });
    out += '</table>';
    return out;
  });
  // Blockquote
  html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');
  // Line breaks
  html = html.replace(/\n/g, '<br>');
  html = html.replace(/<\/li><br>/g, '</li>');
  html = html.replace(/<\/ul><br>/g, '</ul>');
  html = html.replace(/<\/blockquote><br>/g, '</blockquote>');
  html = html.replace(/<\/pre><br>/g, '</pre>');
  html = html.replace(/<\/h[123]><br>/g, m => m.replace('<br>', ''));
  return html;
}

function copyCode(id) {
  const el = document.getElementById(id);
  if (el) {
    navigator.clipboard.writeText(el.textContent);
    const btn = el.previousElementSibling;
    if (btn) { btn.textContent = 'Copied!'; setTimeout(() => btn.textContent = 'Copy', 1500); }
  }
}

async function chat() {
  const ta = document.getElementById('prompt');
  const msg = ta.value.trim();
  if (!msg || generating) return;
  ta.value = '';
  ta.style.height = 'auto';
  generating = true;
  document.getElementById('send-btn').disabled = true;
  document.getElementById('mesh').classList.add('generating');

  // Hide cards, welcome, and prompt on first message — chat becomes priority
  document.getElementById('mesh').classList.add('chat-active');
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.style.display = 'none';
  const prompt = document.getElementById('input-prompt');
  if (prompt) prompt.style.display = 'none';

  addMsg('user', msg);
  const el = addMsg('assistant', '');
  el.innerHTML = '<span class="cursor"></span>';

  let fullText = '', totalMs = 0, tokenCount = 0;

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg, mode: currentMode, history: conversationHistory }),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: resp.statusText }));
      el.innerHTML = '<span style="color:var(--red)">' + esc(err.error || 'Request failed') + '</span>';
      generating = false;
      document.getElementById('mesh').classList.remove('generating');
      document.getElementById('send-btn').disabled = false;
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const d = JSON.parse(line.slice(6));
          if (d.error) {
            const isContextFull = d.error.toLowerCase().includes('context');
            el.innerHTML = renderMarkdown(fullText) + '<span style="color:var(--red)"> [' + esc(d.error) + ']</span>' +
              (isContextFull ? '<br><button class="new-chat-btn" onclick="newChat()">New conversation</button>' : '');
            generating = false;
            document.getElementById('mesh').classList.remove('generating');
            document.getElementById('send-btn').disabled = false;
            return;
          }
          if (d.token) {
            fullText += d.token;
            tokenCount++;
            totalMs += d.time_ms || 0;
            el.innerHTML = renderMarkdown(fullText) + '<span class="cursor"></span>';
            el.scrollIntoView({ behavior: 'smooth' });
          }
          if (d.done) {
            const avg = tokenCount > 0 ? (totalMs / tokenCount).toFixed(0) : 0;
            const tps = tokenCount > 0 ? (1000 * tokenCount / totalMs).toFixed(1) : 0;
            el.innerHTML = renderMarkdown(fullText) +
              '<div class="meta">' + tokenCount + ' tok | ' + avg + 'ms/tok | ' + tps + ' tok/s &middot; BareMetalRT</div>';
          }
        } catch(e) {}
      }
    }
  } catch(e) {
    if (fullText) {
      const avg = tokenCount > 0 ? (totalMs / tokenCount).toFixed(0) : '?';
      const tps = tokenCount > 0 ? (1000 * tokenCount / totalMs).toFixed(1) : '?';
      el.innerHTML = renderMarkdown(fullText) +
        '<div class="meta">' + tokenCount + ' tok | ' + avg + 'ms/tok | ' + tps + ' tok/s &middot; BareMetalRT</div>';
    } else {
      el.innerHTML = '<span style="color:var(--red)">Connection lost. Try again.</span>';
    }
  }

  if (!el.querySelector('.meta') && tokenCount > 0) {
    const avg = (totalMs / tokenCount).toFixed(0);
    const tps = (1000 * tokenCount / totalMs).toFixed(1);
    el.innerHTML = renderMarkdown(fullText) +
      '<div class="meta">' + tokenCount + ' tok | ' + avg + 'ms/tok | ' + tps + ' tok/s &middot; BareMetalRT</div>';
  }

  // Track conversation history (keep last 6 turns to fit context window)
  if (fullText) {
    conversationHistory.push({ role: 'user', content: msg });
    conversationHistory.push({ role: 'assistant', content: fullText });
    if (conversationHistory.length > 12) {
      conversationHistory = conversationHistory.slice(-12);
    }
  }

  generating = false;
  document.getElementById('mesh').classList.remove('generating');
  document.getElementById('send-btn').disabled = false;
}

// ---------------------------------------------------------------------------
// Input handling
// ---------------------------------------------------------------------------

const ta = document.getElementById('prompt');
ta.addEventListener('input', () => { ta.style.height = 'auto'; ta.style.height = ta.scrollHeight + 'px'; });
ta.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); chat(); }
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

refresh();
setInterval(refresh, 3000);
