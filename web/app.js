let user = null;
let isDemo = false;
let generating = false;
let conversationHistory = [];
let currentConvId = null;
let currentModel = null;
let gpuConnected = false;
let _lastGpuState = null; // track changes to avoid unnecessary rerenders

// -- Multi-GPU state ----------------------------------------------------------
let _gpuMode = '1gpu';       // '1gpu' or '2gpu'
let _userDevices = [];        // [{node_id, hostname, gpu_name, gpu_vram_mb, ws_connected}]
let _activeNodeId = null;
let _deviceIdx = 0;
let _tp2PollTimer = null;

// -- Persistence (encoded localStorage — never sent to server) ---------------

function _storageKey() {
  const uid = user?.id || 'anon';
  return 'bmrt_conv_v2_' + uid;
}

function _getConversations() {
  try {
    const key = _storageKey();
    const enc = localStorage.getItem(key);
    if (enc) return JSON.parse(atob(enc));
    // Migration from older formats (only for non-anon users)
    if (user?.id) {
      const old = localStorage.getItem('bmrt_conv_v2');
      if (old) {
        // Migrate old shared data to this user's key, then remove shared key
        const convs = JSON.parse(atob(old));
        localStorage.removeItem('bmrt_conv_v2');
        _setConversations(convs);
        return convs;
      }
    }
    const old_enc = localStorage.getItem('bmrt_conversations_enc');
    if (old_enc) { localStorage.removeItem('bmrt_conversations_enc'); }
    const old_plain = localStorage.getItem('bmrt_conversations');
    if (old_plain) {
      const convs = JSON.parse(old_plain);
      localStorage.removeItem('bmrt_conversations');
      _setConversations(convs);
      return convs;
    }
    return [];
  } catch(e) { return []; }
}

function _setConversations(convs) {
  try {
    localStorage.setItem(_storageKey(), btoa(JSON.stringify(convs)));
  } catch(e) {
    localStorage.setItem(_storageKey(), JSON.stringify(convs));
  }
}

function saveHistory() {
  if (conversationHistory.length === 0) return;
  let convs = _getConversations();
  if (currentConvId) {
    const idx = convs.findIndex(c => c.id === currentConvId);
    if (idx >= 0) {
      convs[idx].messages = conversationHistory;
      convs[idx].title = conversationHistory[0]?.content?.slice(0, 60) || 'Untitled';
      convs[idx].updated_at = Date.now();
    }
  } else {
    currentConvId = 'conv_' + Date.now() + '_' + Math.random().toString(36).slice(2, 6);
    convs.push({
      id: currentConvId,
      model: currentModel || 'default',
      title: conversationHistory[0]?.content?.slice(0, 60) || 'Untitled',
      messages: conversationHistory,
      updated_at: Date.now(),
    });
  }
  _setConversations(convs);
  loadSidebar();
}

function loadHistory() {
  const convs = _getConversations();
  if (convs.length === 0) return;
  // Load most recent
  const latest = convs.sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0))[0];
  currentConvId = latest.id;
  currentModel = latest.model;
  conversationHistory = latest.messages || [];
  if (conversationHistory.length > 0) {
    const msgs = document.getElementById('messages');
    msgs.innerHTML = '';
    for (const m of conversationHistory) {
      addMsg(m.role, m.content);
    }
    showChat();
  }
}

function newChat() {
  currentConvId = null;
  conversationHistory = [];
  document.getElementById('messages').innerHTML = '';
  if (currentView === 'chat') {
    // Stay in chat, just clear
    document.getElementById('prompt').focus();
  } else {
    showModels();
  }
  loadSidebar();
}

// -- Auth -------------------------------------------------------------------

let authMode = 'login'; // 'login' or 'register'

async function checkAuth() {
  try {
    const r = await fetch('/auth/me');
    if (r.ok) {
      user = await r.json();
      isDemo = user.email === 'demo@baremetalrt.ai';
      document.getElementById('auth-buttons').style.display = 'none';
      document.getElementById('user-menu').style.display = 'flex';
      document.getElementById('mode-switcher').style.display = 'flex';
      document.getElementById('header-status').style.display = '';
      document.getElementById('hamburger-wrap').style.display = '';
      document.getElementById('app-layout').style.display = 'flex';
      closeAuth();
      showModels();
      document.getElementById('user-initials').textContent = user.first_name || user.name || user.email.split('@')[0];
      document.getElementById('prompt').placeholder = 'Send a message...';
      // Hide demo link when signed in
      const demoLink = document.getElementById('demo-link');
      if (demoLink) demoLink.style.display = 'none';
      if (user.is_admin) {
        const adminLink = document.getElementById('admin-link');
        if (adminLink) adminLink.style.display = 'block';
        const hAdminHeader = document.getElementById('hamburger-admin-header');
        if (hAdminHeader) hAdminHeader.style.display = '';
        const hAdminLink = document.getElementById('hamburger-admin-link');
        if (hAdminLink) hAdminLink.style.display = '';
      }

      if (isDemo) applyDemoRestrictions();
      if (!isDemo) fetchDevices();
      checkNode();
      checkDaemonVersion();
      if (!isDemo) checkPendingClaims();
      loadHistory();
      loadSidebar();
    } else {
      await autoDemo();
    }
  } catch(e) {
    await autoDemo();
  }
}

async function autoDemo() {
  // Don't auto-login if user just intentionally signed out
  if (sessionStorage.getItem('bmrt_logged_out')) {
    sessionStorage.removeItem('bmrt_logged_out');
    showUnauthenticated();
    return;
  }
  try {
    const r = await fetch('/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: 'demo@baremetalrt.ai', password: 'demo2026!' }),
    });
    if (r.ok) { window.location.reload(); return; }
  } catch(e) { console.error('autoDemo:', e); }
  showUnauthenticated();
}

function showUnauthenticated() {
  // Hide everything
  document.getElementById('app-layout').style.display = 'none';
  document.getElementById('mode-switcher').style.display = 'none';
  document.getElementById('header-status').style.display = 'none';
  document.getElementById('user-menu').style.display = 'none';
  const nav = document.querySelectorAll('.header-link');
  nav.forEach(l => l.style.display = 'none');
  // Show login
  showAuth('login');
}

function showAuth(mode) {
  authMode = mode;
  const modal = document.getElementById('auth-modal');
  modal.classList.add('active');
  document.getElementById('auth-error').style.display = 'none';
  document.getElementById('auth-email').value = '';
  document.getElementById('auth-password').value = '';
  document.getElementById('auth-name').value = '';

  if (mode === 'register') {
    document.getElementById('auth-title').textContent = 'Create Account';
    document.getElementById('auth-submit').textContent = 'Sign Up';
    document.getElementById('auth-name').style.display = '';
    document.getElementById('auth-switch').innerHTML = 'Already have an account? <a onclick="showAuth(\'login\')">Sign in</a>';
  } else {
    document.getElementById('auth-title').textContent = 'Sign In';
    document.getElementById('auth-submit').textContent = 'Sign In';
    document.getElementById('auth-name').style.display = 'none';
    document.getElementById('auth-switch').innerHTML = 'No account? <a onclick="showAuth(\'register\')">Sign up</a>';
  }
  document.getElementById('auth-email').focus();
}

function closeAuth() {
  document.getElementById('auth-modal').classList.remove('active');
}

async function submitAuth() {
  const email = document.getElementById('auth-email').value.trim();
  const password = document.getElementById('auth-password').value;
  const name = document.getElementById('auth-name').value.trim();
  const errorEl = document.getElementById('auth-error');
  const btn = document.getElementById('auth-submit');

  if (!email || !password) {
    errorEl.textContent = 'Email and password required';
    errorEl.style.display = 'block';
    return;
  }

  btn.disabled = true;
  btn.textContent = 'Loading...';
  errorEl.style.display = 'none';

  try {
    const endpoint = authMode === 'register' ? '/auth/register' : '/auth/login';
    const body = authMode === 'register'
      ? { email, password, name }
      : { email, password };

    const r = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const data = await r.json();
    if (!r.ok) {
      errorEl.textContent = data.error || 'Something went wrong';
      errorEl.style.display = 'block';
      btn.disabled = false;
      btn.textContent = authMode === 'register' ? 'Sign Up' : 'Sign In';
      return;
    }

    closeAuth();
    await checkAuth();
  } catch(e) {
    errorEl.textContent = 'Connection error';
    errorEl.style.display = 'block';
    btn.disabled = false;
    btn.textContent = authMode === 'register' ? 'Sign Up' : 'Sign In';
  }
}

async function tryDemo() {
  try {
    const r = await fetch('/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: 'demo@baremetalrt.ai', password: 'demo2026!' }),
    });
    if (r.ok) { window.location.reload(); }
    else { document.getElementById('auth-error').textContent = 'Demo temporarily unavailable'; document.getElementById('auth-error').style.display = 'block'; }
  } catch(e) { document.getElementById('auth-error').textContent = 'Demo temporarily unavailable'; document.getElementById('auth-error').style.display = 'block'; }
}

async function logout() {
  await fetch('/auth/logout', { method: 'POST' });
  sessionStorage.setItem('bmrt_logged_out', '1');
  window.location = '/';
}

// -- Demo restrictions ------------------------------------------------------

function applyDemoRestrictions() {
  // Grey out hamburger menu items that demo users shouldn't access
  const menuItems = document.querySelectorAll('.hamburger-panel .menu-item');
  menuItems.forEach(item => {
    const text = item.textContent.trim();
    if (['Model Storage', 'Downloads', 'API Keys', 'Restart GPU Node', 'Shutdown GPU Node'].includes(text)) {
      item.style.opacity = '0.35';
      item.style.pointerEvents = 'none';
      item.title = 'Not available in demo mode';
      item.onclick = null;
    }
  });
  // Grey out account settings in dropdown
  const dropItems = document.querySelectorAll('.user-dropdown .dropdown-item');
  dropItems.forEach(item => {
    const text = item.textContent.trim();
    if (['Account Settings', 'Model Storage', 'Downloads', 'API Keys'].includes(text)) {
      item.style.opacity = '0.35';
      item.style.pointerEvents = 'none';
      item.title = 'Not available in demo mode';
      item.onclick = null;
    }
  });
}

function demoBlock() {
  showModeToast('Not available in demo mode', { clientX: window.innerWidth / 2, clientY: 80 });
  return true;
}

// -- Node status ------------------------------------------------------------

async function checkNode() {
  try {
    const r = await fetch('/api/gpu-status');
    const d = await r.json();

    // Update active node from server
    if (d.active_node_id) _activeNodeId = d.active_node_id;

    // Show/hide GPU carousel arrows flanking the card
    if (_gpuMode === '1gpu' && _userDevices.length > 1) {
      const idx = _userDevices.findIndex(dev => dev.node_id === _activeNodeId);
      if (idx >= 0) _deviceIdx = idx;
      _showGpuNav(true);
    } else {
      _showGpuNav(false);
    }

    const badge = document.getElementById('header-status');
    const stateKey = d.connected ? 'on' : 'off';
    const stateChanged = stateKey !== _lastGpuState;
    _lastGpuState = stateKey;

    if (d.connected) {
      gpuConnected = true;
      badge.className = 'status-badge online';
      badge.textContent = 'READY';
      document.getElementById('status-dot').className = 'dot green';
      document.getElementById('model-info').textContent = 'GPU connected';
      document.getElementById('vram-info').textContent = '';
      document.getElementById('prompt').disabled = false;
      document.getElementById('send-btn').disabled = false;
      document.getElementById('input-area').style.display = '';
      if (document.getElementById('gpu-dot')) document.getElementById('gpu-dot').className = 'gpu-dot online';
      // Fetch GPU details on connect (always refresh to keep card current)
      await refreshGpuCard();
      if (stateChanged && currentView === 'models') loadModels();
      // Ensure currentModel is always set when GPU is connected
      if (!currentModel || currentModel === 'default') {
        const dn = document.getElementById('gpu-display-name');
        if (dn && dn.textContent && dn.textContent !== 'Detecting GPU...' && dn.textContent !== 'No model loaded') {
          currentModel = dn.textContent;
        }
      }
      if (conversationHistory.length > 0 && currentView !== 'models') {
        showChat();
      }
    } else {
      gpuConnected = false;
      const bannerEl = document.getElementById('maintenance-banner');
      const inMaintenance = bannerEl && bannerEl.style.display !== 'none';
      badge.className = inMaintenance ? 'status-badge starting' : 'status-badge offline';
      badge.textContent = inMaintenance ? 'MAINTENANCE' : 'OFFLINE';
      document.getElementById('status-dot').className = 'dot yellow';
      document.getElementById('model-info').textContent = 'No GPU connected';
      document.getElementById('prompt').disabled = true;
      document.getElementById('send-btn').disabled = true;
      document.getElementById('prompt').placeholder = 'Connect a GPU to start chatting...';
      document.getElementById('gpu-card').classList.remove('active');
      if (document.getElementById('gpu-dot')) document.getElementById('gpu-dot').className = 'gpu-dot offline';
      if (document.getElementById('gpu-status-text')) document.getElementById('gpu-status-text').textContent = '';
      document.getElementById('gpu-display-name').textContent = 'No GPU connected';
      document.getElementById('gpu-vram-display').textContent = '';
      document.getElementById('gpu-engine-display').textContent = '';
    }
  } catch(e) {
    badge.className = 'status-badge loading';
    badge.textContent = 'CHECKING';
    document.getElementById('status-dot').className = 'dot yellow';
    document.getElementById('model-info').textContent = 'Checking connection...';
  }
}

// -- Version check ----------------------------------------------------------

let _versionChecked = false;
async function checkDaemonVersion() {
  try {
    const r = await fetch('/api/system-info');
    const d = await r.json();
    if (d.version && d.version !== '0.0.0') {
      const el = document.getElementById('daemon-version');
      if (el) el.textContent = 'v' + d.version;
    }
  } catch(e) { console.error('checkDaemonVersion:', e); }

  // Check GitHub for newer release (once per session)
  if (_versionChecked) return;
  _versionChecked = true;
  try {
    const r = await fetch('https://api.github.com/repos/baremetalrt/baremetalrt/releases/latest');
    const d = await r.json();
    if (!d.tag_name) return;
    const latest = d.tag_name.replace(/^v/, '');
    const hr = await fetch('/health');
    const h = await hr.json();
    const current = h.version || '';
    if (latest && current && latest !== current) {
      const banner = document.getElementById('update-banner');
      if (banner) {
        banner.style.display = '';
        banner.innerHTML = `Update available: <strong>v${latest}</strong> (you have v${current}) &mdash; ` +
          `<a href="${d.html_url}" target="_blank" style="color:#8b5cf6;text-decoration:underline;">Download</a> ` +
          `<span onclick="this.parentElement.style.display='none'" style="color:#55555e;cursor:pointer;margin-left:12px;">Dismiss</span>`;
      }
    }
  } catch(e) { console.error('checkDaemonVersion/github:', e); }
}

// -- Chat -------------------------------------------------------------------

function addMsg(role, text) {
  const el = document.createElement('div');
  el.className = 'msg ' + role;
  if (role === 'assistant' && text) {
    el.innerHTML = renderMarkdown(text);
  } else {
    el.textContent = text;
  }
  const msgs = document.getElementById('messages');
  msgs.appendChild(el);
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
    // Display math: \[ ... \] or $$ ... $$
    text = text.replace(/\\\[([\s\S]*?)\\\]/g, (_, expr) => {
      try { return katex.renderToString(expr.trim(), { displayMode: true }); } catch(e) { return _; }
    });
    text = text.replace(/\$\$([\s\S]*?)\$\$/g, (_, expr) => {
      try { return katex.renderToString(expr.trim(), { displayMode: true }); } catch(e) { return _; }
    });
    // Inline math: \( ... \) or $ ... $
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
  // Restore KaTeX HTML
  html = html.replace(/__KATEX_(\d+)__/g, (_, i) => katexBlocks[parseInt(i)]);
  // Code blocks: ```lang\n...\n```
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
    const id = 'cb-' + Math.random().toString(36).slice(2, 8);
    return `<pre><code class="lang-${lang || 'text'}">${code.trim()}</code><button class="copy-btn" onclick="copyCode('${id}')">Copy</button><span id="${id}" style="display:none">${code.trim()}</span></pre>`;
  });
  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Images: ![alt](url)
  html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" class="msg-img" loading="lazy">');
  // Links: [text](url)
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  // Auto-link bare URLs (not already inside href or src)
  html = html.replace(/(?<!="|=')((https?:\/\/)[^\s<]+)/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');
  // Strikethrough
  html = html.replace(/~~(.+?)~~/g, '<del>$1</del>');
  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Horizontal rules
  html = html.replace(/^(?:---|\*\*\*|___)$/gm, '<hr>');
  // Headers
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
  // Task lists: - [x] or - [ ]
  html = html.replace(/^- \[x\] (.+)$/gm, '<li class="task done"><input type="checkbox" checked disabled> $1</li>');
  html = html.replace(/^- \[ \] (.+)$/gm, '<li class="task"><input type="checkbox" disabled> $1</li>');
  // Unordered lists
  html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li[\s>].*<\/li>\n?)+/g, '<ul>$&</ul>');
  // Ordered lists
  html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
  // Tables: | col | col |
  html = html.replace(/((?:^\|.+\|$\n?)+)/gm, (table) => {
    const rows = table.trim().split('\n').filter(r => r.trim());
    if (rows.length < 2) return table;
    let out = '<table>';
    rows.forEach((row, i) => {
      if (row.match(/^\|[\s-:|]+\|$/)) return;
      const cells = row.split('|').filter(c => c.trim());
      const tag = i === 0 ? 'th' : 'td';
      out += '<tr>' + cells.map(c => `<${tag}>${c.trim()}</${tag}>`).join('') + '</tr>';
    });
    out += '</table>';
    return out;
  });
  // Blockquote
  html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');
  // Line breaks (but not inside pre)
  html = html.replace(/\n/g, '<br>');
  // Clean up double <br> inside block elements
  html = html.replace(/<\/li><br>/g, '</li>');
  html = html.replace(/<\/ul><br>/g, '</ul>');
  html = html.replace(/<\/blockquote><br>/g, '</blockquote>');
  html = html.replace(/<\/pre><br>/g, '</pre>');
  html = html.replace(/<\/table><br>/g, '</table>');
  html = html.replace(/<\/h[123]><br>/g, m => m.replace('<br>', ''));
  html = html.replace(/<hr><br>/g, '<hr>');
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
  if (!msg || generating || !user) return;
  ta.value = '';
  ta.style.height = 'auto';
  generating = true;
  document.getElementById('send-btn').disabled = true;
  document.getElementById('gpu-card').classList.add('generating');
  document.body.classList.add('inferring');

  // Switch to chat view
  showChat();

  addMsg('user', msg);
  conversationHistory.push({ role: 'user', content: msg });
  saveHistory();

  const el = addMsg('assistant', '');
  // Context-aware verbs while waiting for first token
  const msgLower = msg.toLowerCase();
  let verbs = ['Thinking...', 'Reasoning...', 'Composing...'];
  if (msgLower.match(/code|function|class|implement|debug|fix|write.*script|python|javascript/)) {
    verbs = ['Analyzing...', 'Writing code...', 'Compiling logic...'];
  } else if (msgLower.match(/plan|design|architect|strategy|how.*should/)) {
    verbs = ['Planning...', 'Mapping approach...', 'Structuring...'];
  } else if (msgLower.match(/explain|what.*is|how.*does|why|describe/)) {
    verbs = ['Researching...', 'Formulating...', 'Synthesizing...'];
  } else if (msgLower.match(/math|calcul|solve|equation|multiply|divide|sum/)) {
    verbs = ['Calculating...', 'Computing...', 'Solving...'];
  } else if (msgLower.match(/write|essay|story|poem|letter|email|draft/)) {
    verbs = ['Drafting...', 'Composing...', 'Writing...'];
  } else if (msgLower.match(/translate|language|spanish|french|chinese/)) {
    verbs = ['Translating...', 'Processing language...', 'Interpreting...'];
  } else if (msgLower.match(/list|compare|pros.*cons|options|recommend/)) {
    verbs = ['Evaluating...', 'Comparing...', 'Analyzing...'];
  }
  let verbIdx = 0;
  el.innerHTML = `<div class="thinking"><div class="thinking-dots"><span></span><span></span><span></span></div><span id="thinking-verb">${verbs[0]}</span></div>`;
  const verbInterval = setInterval(() => {
    verbIdx = (verbIdx + 1) % verbs.length;
    const v = document.getElementById('thinking-verb');
    if (v) v.textContent = verbs[verbIdx];
  }, 2000);

  let fullText = '';
  let firstToken = true;

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ..._apiHeaders() },
      body: JSON.stringify({
        message: msg,
        history: conversationHistory.slice(-10),
        max_tokens: 4096,
      }),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: resp.statusText }));
      el.innerHTML = '<span style="color:#f43f5e">' + esc(err.error || 'Request failed') + '</span>';
      generating = false;
      document.getElementById('gpu-card').classList.remove('generating');
  document.body.classList.remove('inferring');
      document.getElementById('send-btn').disabled = false;
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let tokenCount = 0;
    const genStart = performance.now();
    let genDone = false;

    let wasTruncated = false;

    function processLine(line) {
      if (!line.startsWith('data: ')) return;
      try {
        const data = JSON.parse(line.slice(6));
        if (data.error) {
          // If we already have partial text, show it with the error appended
          if (fullText) {
            fullText += '\n\n**Error:** ' + data.error;
            el.innerHTML = renderMarkdown(fullText);
          } else {
            el.innerHTML = '<span style="color:#f43f5e">' + esc(data.error) + '</span>';
          }
          genDone = true;
          return;
        }
        if (data.token) {
          fullText += data.token;
          tokenCount++;
          if (firstToken) { firstToken = false; clearInterval(verbInterval); }
          el.innerHTML = renderMarkdown(fullText) + '<span class="cursor"></span>';
          el.scrollIntoView({ behavior: 'smooth' });
        }
        if (data.done) {
          if (data.total_tokens) tokenCount = data.total_tokens;
          if (data.truncated) {
            wasTruncated = true;
            fullText += '\n\n---\n*Response truncated — token limit reached. Start a new message to continue.*';
          }
          genDone = true;
        }
      } catch(e) { console.error('chat/sse-parse:', e); }
    }

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        processLine(line);
        if (genDone) break;
      }
      if (genDone) break;
    }

    // Process any remaining data in buffer
    if (buffer.trim()) processLine(buffer.trim());

    // If stream ended without a done signal and we have partial text, mark as truncated
    if (!genDone && fullText && tokenCount > 0) {
      wasTruncated = true;
      fullText += '\n\n---\n*Response interrupted — connection lost. Your partial response is shown above.*';
    }

    const elapsed = (performance.now() - genStart) / 1000;
    const tokSec = tokenCount > 0 && elapsed > 0 ? (tokenCount / elapsed).toFixed(1) : null;

    if (fullText) {
      let statsHtml = '';
      if (tokenCount > 0 && tokSec) {
        const truncLabel = wasTruncated ? ' · truncated' : '';
        statsHtml = `<div class="gen-stats">${tokenCount} tokens · ${tokSec} tok/s · ${elapsed.toFixed(1)}s${truncLabel}</div>`;
      }
      el.innerHTML = renderMarkdown(fullText) + statsHtml;
      conversationHistory.push({ role: 'assistant', content: fullText });
      saveHistory();
    }
  } catch(e) {
    // If we had partial text before the error, preserve it
    if (fullText) {
      el.innerHTML = renderMarkdown(fullText) +
        '<div class="gen-stats" style="color:#f43f5e">Connection lost — partial response shown</div>';
      conversationHistory.push({ role: 'assistant', content: fullText });
      saveHistory();
    } else {
      el.innerHTML = '<span style="color:#f43f5e">Connection error — please try again</span>';
    }
  }

  clearInterval(verbInterval);
  generating = false;
  document.getElementById('gpu-card').classList.remove('generating');
  document.body.classList.remove('inferring');
  document.getElementById('send-btn').disabled = false;
  document.getElementById('prompt').focus();
}

// -- Models -----------------------------------------------------------------

let currentView = 'models'; // 'models', 'chat'

function showModels() {
  currentView = 'models';
  document.getElementById('models-panel').classList.remove('hidden');
  document.getElementById('messages').classList.remove('active');
  document.getElementById('input-area').style.display = 'none';
  document.getElementById('header-back').style.display = 'none';
  document.getElementById('model-bar').style.display = 'none';
  loadModels();
}

function showChat() {
  currentView = 'chat';
  document.getElementById('models-panel').classList.add('hidden');
  document.getElementById('messages').classList.add('active');
  document.getElementById('input-area').style.display = '';
  document.getElementById('model-bar').style.display = '';
  document.getElementById('header-back').style.display = '';
  if (gpuConnected) {
    document.getElementById('prompt').disabled = false;
    document.getElementById('send-btn').disabled = false;
  }
  document.getElementById('prompt').focus();
}

function resumeChat() {
  if (conversationHistory.length > 0) showChat();
}

// updateConversationCard removed — sidebar handles conversation history

// -- GPU device carousel & mode switching -----------------------------------

async function fetchDevices() {
  try {
    const r = await fetch('/api/devices');
    const d = await r.json();
    _userDevices = (d.devices || []).sort((a, b) => (b.gpu_vram_mb || 0) - (a.gpu_vram_mb || 0));
  } catch(e) { console.error('fetchDevices:', e); }
}

function _showGpuNav(show) {
  const nav = document.getElementById('gpu-nav');
  const title = document.getElementById('gpu-nav-title');
  if (nav) nav.style.display = show ? 'flex' : 'none';
  if (title) title.style.display = show ? '' : 'none';
  if (show) {
    const dev = _userDevices[_deviceIdx];
    const label = document.getElementById('gpu-nav-label');
    if (label && dev) label.textContent = _cleanGpuName(dev.gpu_name) || dev.hostname;
  }
}

async function prevDevice() {
  if (_userDevices.length < 2) return;
  _deviceIdx = (_deviceIdx - 1 + _userDevices.length) % _userDevices.length;
  await selectDevice(_userDevices[_deviceIdx].node_id);
}

async function nextDevice() {
  if (_userDevices.length < 2) return;
  _deviceIdx = (_deviceIdx + 1) % _userDevices.length;
  await selectDevice(_userDevices[_deviceIdx].node_id);
}

async function selectDevice(nodeId) {
  try {
    _activeNodeId = nodeId;
    _lastGpuState = null;
    _showGpuNav(_userDevices.length > 1);

    // Instantly update card from cached device data
    const dev = _userDevices.find(d => d.node_id === nodeId);
    if (dev) {
      const st = document.getElementById('gpu-status-text');
      if (st) st.textContent = _cleanGpuName(dev.gpu_name);
      const vr = document.getElementById('gpu-vram-display');
      if (vr && dev.gpu_vram_mb) vr.textContent = Math.round(dev.gpu_vram_mb / 1024) + 'GB VRAM';
      document.getElementById('gpu-display-name').textContent = 'No model loaded';
      document.getElementById('gpu-card').classList.remove('active');
      document.getElementById('gpu-layers-wrap').style.display = 'none';
      document.getElementById('unload-btn').style.display = 'none';
      _swapGpuSvg(dev.gpu_name);
    }

    // Server calls in background — don't block the UI
    fetch(`/api/set-active-node/${nodeId}`, { method: 'POST' }).then(() => {
      checkNode();
      refreshGpuCard();
      loadModels();
    });
  } catch(e) { console.error('selectDevice:', e); }
}

function setGpuMode(mode) {
  _gpuMode = mode;
  document.querySelectorAll('.mode-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.mode === (mode === '1gpu' ? '1gpu' : 'tp2'));
  });
  if (mode === '1gpu') {
    show1GpuLayout();
  } else {
    show2GpuLayout();
  }
}

function show1GpuLayout() {
  document.getElementById('gpu-card').style.display = '';
  _showGpuNav(_userDevices.length > 1);
  document.getElementById('tp2-panel').style.display = 'none';
  if (_tp2PollTimer) { clearInterval(_tp2PollTimer); _tp2PollTimer = null; }
  checkNode();
  loadModels();
}

function show2GpuLayout() {
  document.getElementById('gpu-card').style.display = 'none';
  _showGpuNav(false);
  document.getElementById('tp2-panel').style.display = '';
  _pollTp2Status();
  _tp2PollTimer = setInterval(_pollTp2Status, 5000);
}

async function _pollTp2Status() {
  try {
    const [clusterR, sessionR] = await Promise.all([
      fetch('/api/cluster'),
      fetch('/api/session'),
    ]);
    const cluster = await clusterR.json();
    const session = await sessionR.json();

    // Find user's nodes from cluster data
    const cnodes = cluster.nodes || [];

    // Update rank 0 card
    const r0 = session.status === 'matched' ? session.rank0 : cnodes.find(n => n.rank === 0);
    const r1 = session.status === 'matched' ? session.rank1 : cnodes.find(n => n.rank === 1);

    _updateTp2Card(0, r0, session.status);
    _updateTp2Card(1, r1, session.status);

    const statusEl = document.getElementById('tp2-session-status');
    if (session.status === 'matched') {
      statusEl.textContent = 'Session matched \u2014 ready for inference';
      statusEl.className = 'tp2-session-status matched';
    } else {
      const online = cnodes.filter(n => n.status !== 'offline').length;
      statusEl.textContent = online < 2 ? `Waiting for both GPUs... (${online}/2 online)` : 'Matching session...';
      statusEl.className = 'tp2-session-status';
    }
  } catch(e) { console.error('_pollTp2Status:', e); }
}

function _updateTp2Card(rank, data, sessionStatus) {
  const prefix = `tp2-gpu${rank}`;
  const nameEl = document.getElementById(`${prefix}-name`);
  const hostEl = document.getElementById(`${prefix}-host`);
  const vramEl = document.getElementById(`${prefix}-vram`);
  const statusEl = document.getElementById(`${prefix}-status`);

  if (data) {
    nameEl.textContent = data.gpu || data.gpu_name || '--';
    hostEl.textContent = data.hostname || '';
    vramEl.textContent = data.vram_mb ? `${Math.round(data.vram_mb / 1024)} GB VRAM` : '';
    if (sessionStatus === 'matched') {
      statusEl.textContent = 'MATCHED';
      statusEl.className = 'gpu-mini-status matched';
    } else {
      statusEl.textContent = 'ONLINE';
      statusEl.className = 'gpu-mini-status online';
    }
  } else {
    nameEl.textContent = '--';
    hostEl.textContent = '';
    vramEl.textContent = '';
    statusEl.textContent = 'OFFLINE';
    statusEl.className = 'gpu-mini-status offline';
  }
}

function _cleanGpuName(name) {
  if (!name) return '';
  return name.replace(/\s+GPU$/i, '').trim();
}

function _swapGpuSvg(gpuName) {
  const isLaptop = /laptop|mobile|notebook/i.test(gpuName || '');
  const desktop = document.getElementById('gpu-svg-desktop');
  const laptop = document.getElementById('gpu-svg-laptop');
  if (desktop) desktop.style.display = isLaptop ? 'none' : '';
  if (laptop) laptop.style.display = isLaptop ? '' : 'none';
}

function _apiHeaders() {
  const h = {};
  if (_gpuMode === '2gpu') h['X-GPU-Mode'] = 'tp2';
  return h;
}

// -- Family carousel --------------------------------------------------------
let _allModels = [];
let _families = ['All'];
let _familyIdx = 0;

function prevFamily() {
  _familyIdx = (_familyIdx - 1 + _families.length) % _families.length;
  renderModelCards();
}

function nextFamily() {
  _familyIdx = (_familyIdx + 1) % _families.length;
  renderModelCards();
}

function renderModelCards() {
  const family = _families[_familyIdx];
  document.getElementById('family-name').textContent = family;
  const list = document.getElementById('model-list');
  list.innerHTML = '';

  const filtered = family === 'All' ? _allModels : _allModels.filter(m => m.family === family);
  const sorted = filtered.sort((a, b) => {
    if (a.fits === b.fits) return 0;
    if (a.fits === true) return -1;
    return 1;
  });

  for (const m of sorted) {
    const _se = s => s.replace(/-tp\d.*/, '').replace(/-/g, '');
    const _si = s => s.replace(/-[\d.]+[a-z]*$/, '').replace(/-/g, '');
    const isActive = gpuConnected && (_allModels._activeModelId === m.id || (_allModels._activeModel && _se(_allModels._activeModel).includes(_si(m.id))));
    if (isActive) continue;

    const card = document.createElement('div');
    card.className = 'model-card';
    if (m.fits === false && !m.downloaded) card.style.opacity = '0.5';

    const fitBadge = (m.fits === false) ? `<span class="fit-badge no-fit">TP \u00b7 MESH</span>` : '';

    let action = '';
    if (m.fits === false && !m.downloaded) {
      action = `<button class="model-btn disabled" title="${m.name} requires ${Math.round(m.vram_fp16_mb/1024)}GB VRAM. Enable TP · Home Mesh to split across multiple GPUs." style="opacity:0.5;">Run on Mesh</button>`;
    } else if (!m.downloaded) {
      action = `<button class="model-btn primary" onclick="pullModel('${m.id}')">Pull</button>`;
    } else if (m.fits === false) {
      action = `<button class="model-btn disabled" title="${m.name} requires ${Math.round(m.vram_fp16_mb/1024)}GB VRAM. Enable TP · Home Mesh to split across multiple GPUs." style="opacity:0.5;">Run on Mesh</button>`;
    } else if (!m.engine_built) {
      action = `<button class="model-btn primary" onclick="buildModel('${m.id}')">Build</button>`;
    } else {
      const btnLabel = _allModels._activeModel ? 'Hot Swap' : 'Load';
      action = `<button class="model-btn primary" onclick="loadModel('${m.id}')">${btnLabel}</button>`;
    }

    const vramGb = m.vram_fp16_mb ? Math.round(m.vram_fp16_mb / 1024) : 0;
    const numLayers = m.num_layers || 32;
    let layerHtml = '';
    for (let i = 0; i < numLayers; i++) {
      layerHtml += `<div class="layer" data-i="${i}"></div>`;
    }

    card.innerHTML = `
      ${(m.downloaded && m.engine_built && !isDemo) ? `<span class="delete-x" onclick="confirmDeleteModel('${m.id}', event)" title="Delete">&times;</span>` : ''}
      <div class="card-top">
        <div class="name">${m.name}</div>
        <div class="meta"><span class="vram-highlight">${m.params_b}B</span> · ${vramGb}GB · ${m.context_length} context</div>
        <div class="shard">
          <div class="shard-label">${numLayers} transformer layers</div>
          <div class="shard-layers">${layerHtml}</div>
        </div>
      </div>
      <div class="desc">${m.description}</div>
      <div class="progress" id="prog-${m.id}" style="display:none"></div>
      <div class="model-actions">${action}</div>
      ${fitBadge ? `<div style="margin-top:4px">${fitBadge}</div>` : ''}
    `;
    list.appendChild(card);
  }

  if (list.children.length === 0) {
    list.innerHTML = '<div style="color:var(--silver-dark);text-align:center;font-size:13px;">No models in this family</div>';
  }
}

async function loadModels() {
  try {
    const r = await fetch('/api/models');
    const d = await r.json();
    if (d.error) { document.getElementById('model-list').textContent = d.error; return; }

    // Update GPU card from this same response (avoid double-fetching /api/models)
    _updateGpuCard(d);

    _allModels = d.models || [];
    _allModels._activeModel = d.active_model || '';
    _allModels._activeModelId = d.active_model_id || '';

    // Build family list
    const familySet = new Set(_allModels.map(m => m.family || 'Other'));
    _families = ['All', ...Array.from(familySet)];
    if (_familyIdx >= _families.length) _familyIdx = 0;

    renderModelCards();

    // Resume progress UI for any in-progress downloads or builds
    for (const m of _allModels) {
      try {
        const sr = await fetch(`/api/models/${m.id}/status`);
        const sd = await sr.json();
        const active = sd.pull?.status === 'downloading' || sd.pull?.status === 'paused' ? sd.pull
                     : sd.build?.status === 'building' ? sd.build : null;
        if (active) {
          const el = document.getElementById('prog-' + m.id);
          if (el) {
            const pct = active.percent != null ? active.percent : null;
            _setProgress(el, active.progress || active.status, pct, m.id, active.status);
            pollModelStatus(m.id);
          }
        }
      } catch(e) { console.error('loadModels/status:', e); }
    }
  } catch(e) {
    document.getElementById('model-list').textContent = 'No GPU connected.';
  }
}

function _setProgress(el, text, pct, modelId, status) {
  if (!el) return;
  el.style.display = '';
  el.classList.add('active');
  const hasPct = pct != null && pct >= 0;
  const barWidth = hasPct ? Math.min(100, Math.max(0, pct)) + '%' : '100%';
  const pctLabel = hasPct ? `<span class="progress-pct">${pct}%</span>` : '';
  let controls = '';
  if (status === 'downloading' && modelId) {
    controls = `<button class="prog-ctrl" onclick="pausePull('${modelId}')" title="Pause">&#9646;&#9646;</button>` +
      `<button class="prog-ctrl cancel" onclick="cancelPull('${modelId}')" title="Cancel">&#10005;</button>`;
  } else if (status === 'paused' && modelId) {
    controls = `<button class="prog-ctrl" onclick="pullModel('${modelId}')" title="Resume">&#9654;</button>` +
      `<button class="prog-ctrl cancel" onclick="cancelPull('${modelId}')" title="Cancel">&#10005;</button>`;
  }
  el.innerHTML = `<div class="progress-row"><span class="progress-text" data-full="${text}">${text}</span>${pctLabel}${controls}</div>` +
    `<div class="progress-bar-track"><div class="progress-bar-fill" style="width:${barWidth}"></div></div>`;
}

async function pullModel(id) {
  if (isDemo) { demoBlock(); return; }
  _setProgress(document.getElementById('prog-' + id), 'Downloading', null, id, 'downloading');
  const r = await fetch(`/api/models/${id}/pull`, { method: 'POST' });
  const d = await r.json().catch(() => ({}));
  if (d.status === 'already_downloaded') {
    const el = document.getElementById('prog-' + id);
    if (el) el.classList.remove('active');
    loadModels();
    return;
  }
  pollModelStatus(id);
}

async function pausePull(id) {
  const el = document.getElementById('prog-' + id);
  _setProgress(el, 'Pausing', null, id);
  await fetch(`/api/models/${id}/pause`, { method: 'POST' });
}

async function cancelPull(id) {
  if (!confirm('Cancel download and delete partial files?')) return;
  const el = document.getElementById('prog-' + id);
  _setProgress(el, 'Cancelling', null, id);
  await fetch(`/api/models/${id}/cancel`, { method: 'POST' });
  setTimeout(loadModels, 1000);
}

async function buildModel(id) {
  if (isDemo) { demoBlock(); return; }
  _setProgress(document.getElementById('prog-' + id), 'Building engine', null);
  if (_gpuMode === '2gpu') {
    await fetch(`/api/tp2/build/${id}`, { method: 'POST', headers: _apiHeaders() });
  } else {
    await fetch(`/api/models/${id}/build`, { method: 'POST', headers: _apiHeaders() });
  }
  pollModelStatus(id);
}

function _updateGpuCard(md) {
  if (!md || md.error) return;
  if (md.gpu_name && document.getElementById('gpu-status-text')) {
    // Show hostname when multiple devices are linked
    const activeDev = _userDevices.find(d => d.node_id === _activeNodeId);
    const clean = _cleanGpuName(md.gpu_name);
    const label = (_userDevices.length > 1 && activeDev) ? `${clean} \u00b7 ${activeDev.hostname}` : clean;
    document.getElementById('gpu-status-text').textContent = label;
    _swapGpuSvg(md.gpu_name);
  }
  if (md.gpu_vram_mb) document.getElementById('gpu-vram-display').textContent = Math.round(md.gpu_vram_mb / 1024) + 'GB VRAM';
  if (md.active_model) {
    document.getElementById('gpu-card').classList.add('active');
    const name = md.active_model.replace(/-tp\d.*/, '').replace(/-/g, ' ');
    const displayName = name.charAt(0).toUpperCase() + name.slice(1);
    document.getElementById('gpu-display-name').textContent = displayName;
    const _stripEngine = s => s.replace(/-tp\d.*/, '').replace(/-/g, '');
    const _stripId = s => s.replace(/-[\d.]+[a-z]*$/, '').replace(/-/g, '');
    const _findActive = m => (md.active_model_id && md.active_model_id === m.id) || _stripEngine(md.active_model).includes(_stripId(m.id));
    const am = (md.models || []).find(_findActive);
    if (am) document.getElementById('gpu-display-name').title = am.description || '';
    currentModel = displayName;
    document.getElementById('gpu-engine-display').textContent = '';
    document.getElementById('unload-btn').style.display = '';
    document.getElementById('unload-btn').textContent = 'Unload';
    document.getElementById('gpu-chat-btn').style.display = '';
    document.getElementById('prompt').placeholder = 'Chat with ' + displayName + '...';
    document.getElementById('chat-model-text').textContent = displayName + ' \u00b7 TensorRT-LLM';
    if (am && am.num_layers) {
      const wrap = document.getElementById('gpu-layers-wrap');
      wrap.style.display = '';
      document.getElementById('gpu-layers-label').textContent = am.num_layers + ' transformer layers';
      let layerHtml = '';
      for (let i = 0; i < am.num_layers; i++) {
        layerHtml += `<div class="layer" style="animation:layerPulse 3.2s ease-in-out infinite;animation-delay:${(i*0.1).toFixed(1)}s"></div>`;
      }
      document.getElementById('gpu-layers').innerHTML = layerHtml;
    }
  } else {
    document.getElementById('gpu-display-name').textContent = 'No model loaded';
    document.getElementById('gpu-engine-display').textContent = '';
    document.getElementById('unload-btn').style.display = 'none';
    document.getElementById('gpu-chat-btn').style.display = 'none';
    document.getElementById('gpu-layers-wrap').style.display = 'none';
    document.getElementById('prompt').placeholder = 'Load a model to start chatting...';
  }
}

async function refreshGpuCard() {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 5000);
    const r = await fetch('/api/models', { signal: controller.signal });
    clearTimeout(timer);
    if (!r.ok) return;
    const md = await r.json();
    _updateGpuCard(md);
  } catch(e) { if (e.name !== 'AbortError') console.error('refreshGpuCard error:', e); }
}

async function loadModel(id) {
  const el = document.getElementById('prog-' + id);
  // Add shimmer to the button
  const btns = document.querySelectorAll('.model-btn');
  btns.forEach(b => { if (b.textContent === 'Load') b.classList.add('loading'); });
  if (el) { el.style.display = ''; el.textContent = 'Loading engine...'; }
  const r = await fetch(`/api/models/${id}/load`, { method: 'POST' });
  const d = await r.json();
  if (d.error) {
    if (el) el.textContent = d.error;
  } else {
    currentModel = id;
    currentConvId = null;
    conversationHistory = [];
    document.getElementById('messages').innerHTML = '';
    // Wait for daemon to finish loading before refresh
    await new Promise(r => setTimeout(r, 1000));
    _lastGpuState = null;
    await checkNode();
    await refreshGpuCard();
    await loadModels();
  }
}

async function unloadModel() {
  if (isDemo) { demoBlock(); return; }
  document.getElementById('unload-btn').textContent = 'Unloading...';
  const r = await fetch('/api/unload', { method: 'POST' });
  document.getElementById('unload-btn').style.display = 'none';
  document.getElementById('gpu-display-name').textContent = 'No model loaded';
  document.getElementById('gpu-engine-display').textContent = '';
  document.getElementById('prompt').disabled = true;
  document.getElementById('send-btn').disabled = true;
  document.getElementById('gpu-layers-wrap').style.display = 'none';
  const chatBtn = document.getElementById('gpu-chat-btn');
  chatBtn.style.opacity = '0.3';
  chatBtn.disabled = true;
  document.getElementById('gpu-card').classList.remove('active');
  _lastGpuState = null;
  await loadModels();
  await checkNode();
}

async function restartDaemon() {
  if (isDemo) { demoBlock(); return; }
  if (!confirm('Restart GPU node? This will pull latest updates and restart the daemon.')) return;
  try {
    await fetch('/api/daemon/restart', { method: 'POST' });
    alert('Daemon is restarting. It will reconnect in a few seconds.');
  } catch(e) { alert('Could not reach daemon.'); }
}

async function shutdownDaemon() {
  if (isDemo) { demoBlock(); return; }
  if (!confirm('Shut down GPU node? You will need to manually restart it.')) return;
  try {
    await fetch('/api/daemon/shutdown', { method: 'POST' });
    alert('Daemon is shutting down.');
  } catch(e) { alert('Could not reach daemon.'); }
}

function confirmDeleteModel(id, e) {
  if (isDemo) { demoBlock(); return; }
  if (!confirm(`Delete this model? You will need to re-download and rebuild the engine to use it again.`)) return;
  showModeToast('Deleting...', e);
  fetch(`/api/models/${id}/delete`, { method: 'POST' }).then(r => r.json()).then(d => {
    if (d.error) { showModeToast(d.error, e); return; }
    _lastGpuState = null;
    setTimeout(() => { loadModels(); checkNode(); }, 500);
  }).catch(e => console.error('deleteModel:', e));
}

function pollModelStatus(id) {
  const el = document.getElementById('prog-' + id);
  _setProgress(el, 'Starting', null);

  const poll = setInterval(async () => {
    try {
      const r = await fetch(`/api/models/${id}/status`);
      const d = await r.json();
      const active = d.pull?.status === 'downloading' || d.pull?.status === 'paused' ? d.pull : d.build;
      if (active && el) {
        const pct = active.percent != null ? active.percent : null;
        const text = active.progress || active.status || '';
        _setProgress(el, text, pct, id, active.status);
      }
      if (!active || active.status === 'done' || active.status === 'error' || active.status === 'idle' || active.status === 'paused') {
        clearInterval(poll);
        if (el) { el.classList.remove('active'); }
        setTimeout(loadModels, 1000);
      }
    } catch(e) { clearInterval(poll); }
  }, 2000);
}

// -- Sidebar ----------------------------------------------------------------

function loadSidebar() {
  const list = document.getElementById('sidebar-list');
  const convs = _getConversations().sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0));
  if (convs.length === 0) {
    list.innerHTML = '<div class="sidebar-empty">No conversations yet</div>';
    return;
  }
  list.innerHTML = '';
  // Group by model
  const groups = {};
  for (const c of convs) {
    const model = c.model || 'Unknown';
    if (!groups[model]) groups[model] = [];
    groups[model].push(c);
  }
  for (const [model, items] of Object.entries(groups)) {
    const header = document.createElement('div');
    header.style.cssText = 'font-size:10px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; color:var(--silver-dark); padding:10px 12px 4px; margin-top:4px;';
    header.textContent = model;
    list.appendChild(header);
    for (const c of items) {
      const msgCount = (c.messages || []).length;
      const item = document.createElement('div');
      item.className = 'sidebar-item' + (c.id === currentConvId ? ' active' : '');
      item.onclick = (e) => { if (!e.target.classList.contains('conv-delete')) loadConversation(c.id); };
      item.innerHTML = `<div class="conv-text"><div>${c.title || 'Untitled'}</div><div class="conv-model">${msgCount} msgs</div></div><span class="conv-delete" onclick="deleteConversation('${c.id}')" title="Delete">&times;</span>`;
      list.appendChild(item);
    }
  }
  if (convs.length >= 2) {
    const clearBtn = document.createElement('div');
    clearBtn.style.cssText = 'padding:10px 12px; text-align:center;';
    clearBtn.innerHTML = `<button onclick="confirmClearHistory()" style="font-size:10px; padding:4px 12px; border:1px solid #55555e; background:transparent; color:#55555e; cursor:pointer; font-family:inherit; letter-spacing:0.5px;">Clear all</button>`;
    list.appendChild(clearBtn);
  }
}

function loadConversation(id) {
  const convs = _getConversations();
  const conv = convs.find(c => c.id === id);
  if (!conv) return;
  currentConvId = conv.id;
  currentModel = conv.model;
  conversationHistory = conv.messages || [];
  const msgs = document.getElementById('messages');
  msgs.innerHTML = '';
  for (const m of conversationHistory) {
    addMsg(m.role, m.content);
  }
  showChat();
  loadSidebar();
}

// -- User dropdown ----------------------------------------------------------

// showSystemInfo removed — now a dedicated page at /system

function toggleHamburger() {
  document.getElementById('hamburger-panel').classList.toggle('active');
  document.getElementById('hamburger-overlay').classList.toggle('active');
}

function toggleUserDropdown() {
  document.getElementById('user-dropdown').classList.toggle('active');
}
// Close dropdown when clicking anywhere else
document.addEventListener('click', (e) => {
  if (!e.target.closest('.user-menu')) document.getElementById('user-dropdown')?.classList.remove('active');
});

// -- Mode toast -------------------------------------------------------------

let toastTimer = null;
let toastMoveHandler = null;
function showModeToast(text, e) {
  const el = document.getElementById('mode-toast');
  el.textContent = text;
  if (e) {
    el.style.left = e.clientX + 'px';
    el.style.top = (e.clientY + 16) + 'px';
  }
  // Follow mouse while visible
  if (toastMoveHandler) document.removeEventListener('mousemove', toastMoveHandler);
  toastMoveHandler = (ev) => {
    el.style.left = ev.clientX + 'px';
    el.style.top = (ev.clientY + 16) + 'px';
  };
  document.addEventListener('mousemove', toastMoveHandler);
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.classList.remove('show');
    document.removeEventListener('mousemove', toastMoveHandler);
    toastMoveHandler = null;
  }, 1500);
}

// -- Clear history ----------------------------------------------------------

function confirmClearHistory() {
  document.getElementById('confirm-toast').classList.add('active');
}

function hideToast() {
  document.getElementById('confirm-toast').classList.remove('active');
}

function deleteConversation(id) {
  let convs = _getConversations();
  convs = convs.filter(c => c.id !== id);
  _setConversations(convs);
  if (currentConvId === id) {
    currentConvId = null;
    conversationHistory = [];
    document.getElementById('messages').innerHTML = '';
    showModels();
  }
  loadSidebar();
}

function clearAllHistory() {
  hideToast();
  _setConversations([]);
  conversationHistory = [];
  currentConvId = null;
  document.getElementById('messages').innerHTML = '';
  document.getElementById('messages').classList.remove('active');
  loadSidebar();
  showModels();
}

// -- GPU Metrics ------------------------------------------------------------

async function updateGpuMetrics() {
  if (!gpuConnected || currentView !== 'models') return;
  try {
    const r = await fetch('/api/gpu-metrics');
    if (!r.ok) return;
    const d = await r.json();
    if (d.vram_total_mb) {
      const pct = Math.round((d.vram_used_mb / d.vram_total_mb) * 100);
      document.getElementById('vram-fill').style.width = pct + '%';
      const used = (d.vram_used_mb / 1024).toFixed(1);
      const total = (d.vram_total_mb / 1024).toFixed(1);
      const tempClass = d.temperature_c > 80 ? 'hot' : (d.temperature_c > 65 ? 'warm' : '');
      const el = document.getElementById('gpu-metrics');
      // Build DOM once, then update values
      if (!el.querySelector('.gpu-stats-row')) {
        el.innerHTML = `
          <div class="gpu-stats-row">
            <div class="gpu-stat-block" id="gpu-stat-temp"><div class="stat-value"></div><div class="stat-label">Temp</div></div>
            <div class="gpu-stat-block" id="gpu-stat-util"><div class="stat-value"></div><div class="stat-label">Util</div></div>
            <div class="gpu-stat-block" id="gpu-stat-power"><div class="stat-value"></div><div class="stat-label">Power</div></div>
          </div>`;
      }
      document.getElementById('gpu-vram-display').textContent = `${used}/${total}GB VRAM`;
      document.getElementById('gpu-stat-temp').className = `gpu-stat-block ${tempClass}`;
      document.getElementById('gpu-stat-temp').querySelector('.stat-value').textContent = `${d.temperature_c || '—'}°`;
      document.getElementById('gpu-stat-util').querySelector('.stat-value').textContent = `${d.gpu_util_pct !== undefined ? d.gpu_util_pct : '—'}%`;
      document.getElementById('gpu-stat-power').querySelector('.stat-value').textContent = `${d.power_w || '—'}W`;
    }
  } catch(e) { console.error('updateGpuMetrics:', e); }
}

// -- Poll -------------------------------------------------------------------

setInterval(() => { if (user) checkNode(); }, 10000);
setInterval(updateGpuMetrics, 5000);

// -- Maintenance banners ----------------------------------------------------

async function fetchBanners() {
  try {
    const r = await fetch('/admin/banners');
    if (!r.ok) return;
    const banners = await r.json();
    const b = banners.banner_1gpu;
    const el = document.getElementById('maintenance-banner');
    if (b && b.enabled) {
      el.textContent = b.message;
      el.style.display = 'block';
      // Override status badge to MAINTENANCE
      const badge = document.getElementById('header-status');
      if (badge) {
        badge.className = 'status-badge starting';
        badge.textContent = 'MAINTENANCE';
      }
      document.getElementById('prompt').disabled = true;
      document.getElementById('send-btn').disabled = true;
      document.getElementById('prompt').placeholder = b.message || 'Maintenance in progress...';
    } else {
      el.style.display = 'none';
      // Re-enable chat — checkNode() will set final state
      checkNode();
    }
  } catch(e) { console.error('fetchBanners:', e); }
}

function toggleAdminPanel() {
  const panel = document.getElementById('admin-panel');
  if (panel.style.display === 'none') {
    loadAdminBanners();
    panel.style.display = '';
  } else {
    panel.style.display = 'none';
  }
}

async function loadAdminBanners() {
  try {
    const r = await fetch('/admin/banners');
    if (!r.ok) return;
    const banners = await r.json();
    const container = document.getElementById('admin-banners');
    container.innerHTML = '';
    const labels = { banner_1gpu: '1 GPU (App)', banner_2gpu: '2 GPU (Demo)' };
    for (const [key, val] of Object.entries(banners)) {
      const row = document.createElement('div');
      row.style.cssText = 'display:flex; align-items:center; gap:10px; padding:8px 12px; background:#12121a; border-radius:6px; border:1px solid #2a2a34;';
      row.innerHTML = `
        <label style="display:flex; align-items:center; gap:8px; cursor:pointer; flex:1; color:#d8d8e0; font-size:13px;">
          <input type="checkbox" ${val.enabled ? 'checked' : ''} onchange="toggleBanner('${key}', this.checked)"
            style="accent-color:#8b5cf6; width:16px; height:16px; cursor:pointer;">
          ${labels[key] || key}
        </label>
        <span style="font-size:11px; color:${val.enabled ? '#76e651' : '#9898a4'};">${val.enabled ? 'ON' : 'OFF'}</span>`;
      container.appendChild(row);
    }
  } catch(e) { console.error('loadAdminBanners:', e); }
}

async function toggleBanner(key, enabled) {
  try {
    await fetch('/admin/banners/' + key, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    });
    loadAdminBanners();
    fetchBanners();
  } catch(e) { console.error('toggleBanner:', e); }
}

// -- Sidebar resize ---------------------------------------------------------
(function() {
  const handle = document.getElementById('sidebar-resize');
  const sidebar = document.getElementById('sidebar');
  if (!handle || !sidebar) return;
  const MIN = 180, MAX = 400;
  let dragging = false, startX, startW;
  handle.addEventListener('mousedown', function(e) {
    e.preventDefault();
    dragging = true;
    startX = e.clientX;
    startW = sidebar.offsetWidth;
    handle.classList.add('dragging');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  });
  document.addEventListener('mousemove', function(e) {
    if (!dragging) return;
    const w = Math.min(MAX, Math.max(MIN, startW + e.clientX - startX));
    sidebar.style.width = w + 'px';
  });
  document.addEventListener('mouseup', function() {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove('dragging');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    try { localStorage.setItem('bmrt-sidebar-w', sidebar.style.width); } catch(e) { console.error('sidebar/save-width:', e); }
  });
  // Restore saved width
  try {
    const saved = localStorage.getItem('bmrt-sidebar-w');
    if (saved) sidebar.style.width = saved;
  } catch(e) { console.error('sidebar/restore-width:', e); }
})();

// -- Device Claim (auto-link GPU) -------------------------------------------

async function checkPendingClaims() {
  // Plex-style: fetch claim token from localhost daemon, send to server, push key back
  // Retries — daemon may still be starting when page loads
  for (let attempt = 0; attempt < 5; attempt++) {
    if (attempt > 0) await new Promise(r => setTimeout(r, 3000));
    for (const port of [8080, 9000]) {
      try {
        const tr = await fetch(`http://localhost:${port}/api/claim/token`, { signal: AbortSignal.timeout(2000) });
        if (!tr.ok) continue;
        const claim = await tr.json();
        if (!claim.token) continue;

        const sr = await fetch('/api/claim/direct', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(claim),
        });
        if (!sr.ok) continue;
        const { api_key, token } = await sr.json();

        await fetch(`http://localhost:${port}/api/claim/accept`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ token, api_key }),
        });
        return;
      } catch(e) { console.error('checkPendingClaims:', e); }
    }
  }
}

// -- Init -------------------------------------------------------------------

// Show login immediately — hide only after successful auth
showAuth('login');
checkAuth();
fetchBanners();
