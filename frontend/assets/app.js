const state = { records: [], selected: null, network: null, chatHistory: [], page: 0, uploadedFile: null, activeNetwork: 0 };
const PAGE_SIZE = 20;
const $ = (selector) => document.querySelector(selector);

function setStatus(text, ok = true) {
  const status = $('#api-status');
  status.textContent = text;
  status.style.color = ok ? '#4d9180' : '#b7473f';
}

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function riskLabel(level) {
  return { critical: 'Critique', high: 'Élevé', watch: 'À surveiller', low: 'Faible' }[level] || level;
}

function renderSummary(summary, filename = '') {
  $('#metric-total').textContent = summary.total.toLocaleString('fr-FR');
  $('#metric-risky').textContent = summary.risky.toLocaleString('fr-FR');
  $('#metric-critical').textContent = summary.critical.toLocaleString('fr-FR');
  $('#metric-average').textContent = formatPercent(summary.average_probability);
  $('#metric-threshold').textContent = Number(summary.threshold).toFixed(2);
  $('#metric-file').textContent = filename || 'Analyse active';
}

function focusWorkspace() {
  document.querySelector('.workspace-grid').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderNetwork(network) {
  state.network = network;
  const summary = network?.summary || {};
  const networks = network?.networks || [];
  state.activeNetwork = Math.min(state.activeNetwork, Math.max(networks.length - 1, 0));
  const activeNetwork = networks[state.activeNetwork];
  $('#network-status').textContent = `${summary.networks_detected || 0} réseaux · ${summary.transactions_linked || 0} déclarations liées`;
  $('#network-select').innerHTML = networks.map((item, index) => `<option value="${index}">${item.network_id} · ${item.member_count} membres · ${formatPercent(item.network_risk)}</option>`).join('') || '<option value="0">Aucune communauté</option>';
  $('#network-select').value = String(state.activeNetwork);
  const canvas = $('#network-canvas');
  const activeRows = new Set(activeNetwork?.members?.map((member) => member.row_index) || []);
  const nodeIndex = new Map((network?.nodes || []).map((node) => [node.row_index, node]));
  const visibleNodes = (activeNetwork?.members || []).map((member) => nodeIndex.get(member.row_index) || {
    row_index: member.row_index,
    declaration_id: member.declaration_id,
    label: member.declarant_id,
    network_risk: member.network_risk,
    network_id: activeNetwork.network_id,
  }).slice(0, 45);
  if (!visibleNodes.length) {
    canvas.innerHTML = '<div class="network-empty">Aucun réseau multi-entités détecté sur ce périmètre.</div>';
  } else {
    const max = Math.max(...visibleNodes.map((node) => Number(node.network_risk) || 0), 1);
    const visibleRows = new Set(visibleNodes.map((node) => node.row_index));
    const position = (rowIndex) => {
      const index = visibleNodes.findIndex((node) => node.row_index === rowIndex);
      const angle = (index / Math.max(visibleNodes.length, 1)) * Math.PI * 2 - Math.PI / 2;
      const radius = visibleNodes.length > 1 ? Math.min(39, 17 + visibleNodes.length * 0.55) : 0;
      return { x: 50 + Math.cos(angle) * radius, y: 50 + Math.sin(angle) * radius };
    };
    const communityEdges = activeNetwork?.edges || network.edges || [];
    const lines = communityEdges.filter((edge) => visibleRows.has(edge.source) && visibleRows.has(edge.target)).slice(0, 120).map((edge) => {
      const source = position(edge.source); const target = position(edge.target);
      return `<line x1="${source.x}%" y1="${source.y}%" x2="${target.x}%" y2="${target.y}%" />`;
    }).join('');
    canvas.innerHTML = `<div class="network-map"><svg class="network-links" viewBox="0 0 100 100" preserveAspectRatio="none">${lines}</svg><div class="network-center">${activeNetwork?.network_id || 'Réseau'}</div>${visibleNodes.map((node) => {
      const point = position(node.row_index);
      const x = point.x;
      const y = point.y;
      const size = 10 + Math.round((Number(node.network_risk) || 0) / max * 18);
      return `<button class="network-node ${node.network_id ? 'linked' : ''}" title="${node.declaration_id || 'Déclaration'} · risque réseau ${formatPercent(node.network_risk)}" style="left:${x}%;top:${y}%;width:${size}px;height:${size}px" data-row="${node.row_index}"></button>`;
    }).join('')}<div class="network-legend"><span><i class="legend-link"></i> lien partagé</span><span><i class="legend-node"></i> déclaration</span></div></div>`;
    canvas.querySelectorAll('[data-row]').forEach((node) => node.addEventListener('click', () => selectRecord(Number(node.dataset.row))));
  }
  $('#network-list').innerHTML = networks.slice(0, 8).map((item, index) => `<button class="network-card ${index === state.activeNetwork ? 'active' : ''}" type="button" data-network-index="${index}"><div><b>${item.network_id}</b><small>${item.member_count} déclarations · ${item.relations.join(' · ') || 'liens partagés'}</small></div><strong>${formatPercent(item.network_risk)}</strong><small>${Number(item.estimated_value || 0).toLocaleString('fr-FR')} TND · cohésion ${formatPercent(item.cohesion || 0)}</small></button>`).join('') || '<div class="network-empty">Aucun groupe prioritaire.</div>';
  $('#network-list').querySelectorAll('[data-network-index]').forEach((card) => card.addEventListener('click', () => { state.activeNetwork = Number(card.dataset.networkIndex); renderNetwork(state.network); }));
}

function renderTable() {
  const filter = $('#risk-filter').value;
  const rows = state.records.filter((record) => filter === 'all' || record.risk_level === filter);
  const pageCount = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  state.page = Math.min(state.page, pageCount - 1);
  const pageRows = rows.slice(state.page * PAGE_SIZE, (state.page + 1) * PAGE_SIZE);
  $('#record-count').textContent = `${rows.length.toLocaleString('fr-FR')} dossiers`;
  $('#page-label').textContent = `Page ${state.page + 1} / ${pageCount}`;
  $('#prev-page').disabled = state.page === 0;
  $('#next-page').disabled = state.page >= pageCount - 1;
  const body = $('#records-body');
  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="5" class="empty-state">Aucune déclaration pour ce filtre.</td></tr>';
    return;
  }
  body.innerHTML = pageRows.map((record) => {
    const signal = record.risk_factors?.[0];
    const selected = state.selected?.row_index === record.row_index ? 'selected' : '';
    return `<tr class="${selected}" data-row="${record.row_index}">
      <td class="id-cell">${record.declaration_id ?? `ROW-${record.row_index}`}</td>
      <td class="prob-cell">${formatPercent(record.fraud_probability)}<span class="prob-bar"><i style="width:${Math.min(record.fraud_probability * 100, 100)}%"></i></span></td>
      <td><span class="risk-badge ${record.risk_level}">${riskLabel(record.risk_level)}</span></td>
      <td>${signal ? `<b>${signal.feature}</b><br><small>${signal.value ?? 'missing'}</small>` : '—'}</td>
      <td><button class="view-button" aria-label="Ouvrir">›</button></td>
    </tr>`;
  }).join('');
  body.querySelectorAll('tr[data-row]').forEach((row) => row.addEventListener('click', () => selectRecord(Number(row.dataset.row))));
}

function factorRows(factors, negative = false) {
  if (!factors?.length) return '<div class="factor-row"><small>Aucun facteur dominant.</small></div>';
  return factors.map((factor) => `<div class="factor-row ${negative ? 'negative' : ''}">
    <div><b>${factor.feature}</b><small>${factor.value ?? 'missing'}</small></div>
    <div class="factor-value">${factor.contribution > 0 ? '+' : ''}${Number(factor.contribution).toFixed(3)}</div>
  </div>`).join('');
}

function selectRecord(rowIndex) {
  state.selected = state.records.find((record) => record.row_index === rowIndex);
  if (!state.selected) return;
  const record = state.selected;
  $('#detail-empty').hidden = true;
  $('#detail-content').hidden = false;
  $('#detail-id').textContent = record.declaration_id ?? `ROW-${record.row_index}`;
  $('#detail-badge').textContent = riskLabel(record.risk_level);
  $('#detail-badge').className = `risk-badge ${record.risk_level}`;
  $('#detail-probability').textContent = formatPercent(record.fraud_probability);
  $('#detail-track').style.width = `${Math.min(record.fraud_probability * 100, 100)}%`;
  $('#detail-roi').textContent = `${Number(record.roi_score || 0).toFixed(1)}/100`;
  $('#detail-recovery').textContent = `${Number(record.estimated_recovery_tnd || 0).toLocaleString('fr-FR')} TND`;
  $('#detail-network-risk').textContent = formatPercent(record.network_risk || 0);
  $('#risk-factors').innerHTML = factorRows(record.risk_factors);
  $('#protective-factors').innerHTML = factorRows(record.protective_factors, true);
  $('#similar-transactions').innerHTML = (record.similar_transactions || []).map((match) => `<div class="similar-row"><span>${match.declaration_id ?? `ROW-${match.row_index}`}</span><b>${formatPercent(match.fraud_probability)}</b><small>${formatPercent(match.similarity)} similaire</small></div>`).join('') || '<div class="factor-row"><small>Aucun cas comparable.</small></div>';
  $('#similar-transactions').querySelectorAll('.similar-row').forEach((row, index) => row.addEventListener('click', () => showSimilarCase(record.similar_transactions[index])));
  $('#similar-case-detail').hidden = true;
  $('#llm-output').hidden = true;
  $('#chat-panel').hidden = true;
  $('#chat-messages').innerHTML = '';
  state.chatHistory = [];
  renderTable();
}

function showSimilarCase(match) {
  $('#similar-case-detail').hidden = false;
  $('#similar-case-title').textContent = `Cas comparable · ${match.declaration_id ?? `ROW-${match.row_index}`}`;
  $('#similar-case-severity').textContent = riskLabel(match.risk_level || 'watch');
  $('#similar-case-summary').textContent = `Similarité globale ${formatPercent(match.similarity)} · probabilité ${formatPercent(match.fraud_probability)}. Les lignes ci-dessous montrent concrètement les dimensions rapprochées par le KNN.`;
  $('#similar-case-comparison').innerHTML = (match.comparison || []).map((item) => `<div class="comparison-row"><div><b>${item.feature}</b><small>Votre dossier · ${item.current_value ?? 'manquant'}</small></div><div><b>${item.similar_value ?? 'manquant'}</b><small>${formatPercent(item.similarity)} proche</small></div></div>`).join('') || '<small>Comparaison détaillée indisponible.</small>';
}

async function scoreFile(file) {
  const uploadButton = document.querySelector('.upload-button');
  uploadButton.classList.add('busy');
  setStatus(`Analyse de ${file.name} en cours...`);
  const form = new FormData();
  form.append('file', file);
  const response = await fetch('/api/score', { method: 'POST', body: form });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || 'Erreur de scoring');
  state.records = payload.records;
  state.uploadedFile = file;
  state.page = 0;
  renderNetwork(payload.network);
  state.selected = null;
  renderSummary(payload.summary, payload.filename);
  renderTable();
  $('#detail-empty').hidden = false;
  $('#detail-content').hidden = true;
  setStatus(`Moteur prêt · ${payload.returned} dossiers affichés`);
  uploadButton.classList.remove('busy');
  uploadButton.textContent = '✓ CSV chargé';
  $('#download-button').hidden = false;
  focusWorkspace();
}

async function loadDemo() {
  const button = $('#demo-button');
  button.disabled = true;
  button.textContent = 'Chargement...';
  setStatus('Chargement de l’exemple et construction du graphe...');
  const response = await fetch('/api/demo');
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || 'Impossible de charger l’exemple');
  state.records = payload.records;
  state.page = 0;
  renderNetwork(payload.network);
  state.selected = null;
  renderSummary(payload.summary, 'df_syn_test_eng.csv');
  renderTable();
  setStatus(`Exemple chargé · ${payload.returned} dossiers affichés`);
  button.disabled = false;
  button.textContent = 'Téléchargé';
  focusWorkspace();
}

async function downloadEnrichedCsv() {
  if (!state.uploadedFile) {
    setStatus('Importez un CSV avant de télécharger sa version enrichie.', false);
    return;
  }
  const button = $('#download-button');
  button.disabled = true;
  button.textContent = 'Préparation...';
  const form = new FormData();
  form.append('file', state.uploadedFile);
  try {
    const response = await fetch('/api/export', { method: 'POST', body: form });
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.detail || 'Export impossible');
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${state.uploadedFile.name.replace(/\.csv$/i, '')}_enrichi.csv`;
    link.click();
    URL.revokeObjectURL(url);
    button.textContent = 'Téléchargé';
    setStatus('CSV enrichi téléchargé · Fraude + Explication');
  } catch (error) {
    button.textContent = 'Télécharger CSV enrichi';
    setStatus(error.message, false);
  } finally {
    button.disabled = false;
  }
}

async function requestExplanation() {
  if (!state.selected) return;
  const button = $('#explain-button');
  button.disabled = true;
  button.textContent = 'Génération...';
  try {
    const response = await fetch('/api/explain', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ record: state.selected }) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || 'LLM indisponible');
    const output = $('#llm-output');
    output.textContent = payload.text;
    output.hidden = false;
    $('#llm-source').textContent = payload.source === 'ollama' ? 'Llama 3 · Ollama' : 'Fallback local';
    $('#chat-panel').hidden = false;
    $('#chat-input').focus();
  } catch (error) {
    $('#llm-output').textContent = error.message;
    $('#llm-output').hidden = false;
  } finally {
    button.disabled = false;
    button.textContent = '✦ Générer une explication analyste';
  }
}

function appendChatMessage(role, text) {
  const message = document.createElement('div');
  message.className = `chat-message ${role}`;
  message.textContent = text;
  $('#chat-messages').appendChild(message);
  $('#chat-messages').scrollTop = $('#chat-messages').scrollHeight;
}

async function sendChatMessage(event) {
  event.preventDefault();
  if (!state.selected) return;
  const input = $('#chat-input');
  const message = input.value.trim();
  if (!message) return;
  input.value = '';
  appendChatMessage('user', message);
  state.chatHistory.push({ role: 'user', content: message });
  input.disabled = true;
  setStatus('Question envoyée à Ollama…');
  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ record: state.selected, history: state.chatHistory.slice(0, -1), message }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || 'Assistant indisponible');
    appendChatMessage('assistant', payload.text);
    setStatus(payload.source === 'ollama' ? 'Réponse Llama 3 reçue · Ollama' : 'Réponse locale disponible');
    state.chatHistory.push({ role: 'assistant', content: payload.text });
  } catch (error) {
    appendChatMessage('assistant', error.message);
    setStatus(`Chat indisponible · ${error.message}`, false);
  } finally {
    input.disabled = false;
    input.focus();
  }
}

$('#csv-input').addEventListener('click', (event) => {
  event.target.value = '';
});
$('#csv-input').addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (file) scoreFile(file).catch((error) => {
    document.querySelector('.upload-button').classList.remove('busy');
    setStatus(error.message, false);
  });
});
$('#demo-button').addEventListener('click', () => loadDemo().catch((error) => {
  $('#demo-button').disabled = false;
  $('#demo-button').textContent = 'Charger l’exemple';
  setStatus(error.message, false);
}));
$('#download-button').addEventListener('click', downloadEnrichedCsv);
$('#risk-filter').addEventListener('change', () => { state.page = 0; renderTable(); });
$('#prev-page').addEventListener('click', () => { state.page -= 1; renderTable(); });
$('#next-page').addEventListener('click', () => { state.page += 1; renderTable(); });
$('#network-select').addEventListener('change', (event) => { state.activeNetwork = Number(event.target.value); renderNetwork(state.network); });
$('#explain-button').addEventListener('click', requestExplanation);
$('#chat-form').addEventListener('submit', sendChatMessage);
$('#chat-send').addEventListener('click', () => sendChatMessage({ preventDefault: () => {} }));
$('#network-nav').addEventListener('click', () => {
  $('#network-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
  document.querySelectorAll('.nav-item').forEach((item) => item.classList.remove('active'));
  $('#network-nav').classList.add('active');
});
document.querySelector('.nav-item.active').addEventListener('click', () => {
  window.scrollTo({ top: 0, behavior: 'smooth' });
});
fetch('/api/health').then((response) => response.json()).then((health) => setStatus(`Moteur actif · réseau ${health.network === 'community_graph_ready' ? 'actif' : 'indisponible'}`)).catch(() => setStatus('Backend indisponible', false));
