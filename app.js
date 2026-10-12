const $ = id => document.getElementById(id);
const patients = {
  maya: {name: 'Maya Chen', dob: '1990-04-12'},
  alex: {name: 'Alex Rivera', dob: '1985-09-23'},
};
const toolNames = {
  verify_patient: 'Verify patient', search_slots: 'Search availability',
  preview_booking: 'Preview appointment', book_appointment: 'Book appointment',
  check_booking: 'Check booking result',
};
const closedStages = new Set(['booked', 'escalated', 'cancelled', 'handoff']);
let session = null;
let busy = false;
let pendingMessage = null;
let toastTimer;
let renderedSessionId = null;
let renderedTurns = 0;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function saveToken(token) {
  try {
    if (token) sessionStorage.setItem('careloop-session', token);
    else sessionStorage.removeItem('careloop-session');
  } catch { /* The conversation still works when browser storage is unavailable. */ }
}

function savedToken() {
  try { return sessionStorage.getItem('careloop-session'); }
  catch { return null; }
}

async function api(path, body) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), path === '/api/chat' ? 150000 : 15000);
  try {
    const headers = {'Content-Type': 'application/json'};
    if (session) headers['X-Session-ID'] = session.session_id;
    const response = await fetch(path, {
      method: body === undefined ? 'GET' : 'POST', headers, signal: controller.signal,
      ...(body === undefined ? {} : {body: JSON.stringify(body)}),
    });
    let result;
    try { result = await response.json(); }
    catch {
      const error = new Error(response.ok ? 'The clinic returned an invalid response. Check the conversation before retrying.' : 'The clinic service is busy or unavailable. Please try again.');
      error.status = response.status;
      throw error;
    }
    if (!response.ok) {
      const error = new Error(result.error || 'The clinic could not complete this action.');
      error.status = response.status;
      throw error;
    }
    return result;
  } catch (error) {
    if (error.name === 'AbortError' || error instanceof TypeError) {
      throw new Error('The clinic did not respond. Check the conversation before trying again.');
    }
    throw error;
  } finally { clearTimeout(timer); }
}

function toast(message) {
  clearTimeout(toastTimer);
  $('toast').textContent = message;
  $('toast').classList.remove('hidden');
  toastTimer = setTimeout(() => $('toast').classList.add('hidden'), 6500);
}

function notice(message = '') {
  $('connection-message').textContent = message;
  $('connection-banner').classList.toggle('hidden', !message);
}

function setBusy(value) {
  busy = value;
  const closed = session && closedStages.has(session.stage);
  $('send').disabled = value || !session || closed;
  $('message').disabled = value || !session || closed;
  $('new-chat').disabled = value;
  $('refresh-conversation').disabled = value;
  $('conversation-policy').disabled = value;
  $('demo-patient').disabled = value || Boolean(session?.verified);
  $('use-patient').disabled = value || !session || Boolean(session?.verified) || closed;
  document.querySelectorAll('[data-say]').forEach(button => button.disabled = value || !session || closed);
  $('chat-form').setAttribute('aria-busy', String(value));
  $('request-status').classList.toggle('hidden', !value);
}

function suggestedReply(label, message) {
  const button = element('button', '', label);
  button.type = 'button';
  button.dataset.say = message;
  $('quick-actions').append(button);
}

function updatePatient() {
  $('demo-dob').textContent = patients[$('demo-patient').value].dob;
}

function renderSession() {
  if (renderedSessionId !== session.session_id || renderedTurns > session.history.length) {
    $('messages').replaceChildren();
    renderedSessionId = session.session_id;
    renderedTurns = 0;
  }
  for (const turn of session.history.slice(renderedTurns)) {
    const row = element('div', 'message-row ' + turn.role);
    if (turn.role === 'assistant') row.append(element('span', 'bubble-icon', 'c'));
    row.append(element('div', 'bubble', turn.text));
    $('messages').append(row);
  }
  renderedTurns = session.history.length;
  $('messages').scrollTop = $('messages').scrollHeight;
  $('identity').textContent = session.verified ? 'Verified' : 'Unverified';
  $('identity').className = session.verified ? 'positive' : '';
  const status = {
    identity: 'Verify your demo patient details', request: 'Choose your appointment details',
    choose: 'Choose an available appointment', confirm: 'Waiting for your confirmation',
    clarify: 'Please clarify the changed details', booked: 'Appointment confirmed',
    booking_uncertain: 'Booking result needs checking', service_unavailable: 'Clinic service unavailable',
    escalated: 'Please seek immediate human help', cancelled: 'Request cancelled',
    handoff: 'Start a new conversation or contact the clinic',
  };
  $('agent-status').textContent = status[session.stage] || 'Ready to schedule · IST';
  $('booking-state').textContent = session.stage === 'booked' ? 'Confirmed'
    : session.stage === 'confirm' ? 'Awaiting confirmation'
    : session.stage === 'booking_uncertain' ? 'Result uncertain' : 'Not booked';
  $('booking-state').className = session.stage === 'booked' ? 'positive' : '';
  $('policy-state').textContent = (session.policy.honor_soft_preferences || session.policy.read_retries || session.policy.interpretation_notes) ? 'Updated' : 'Initial';
  $('policy-state').title = 'Policy ' + session.policy.version;
  $('model-mode').textContent = session.language_model ? 'Gemini' : 'Offline';
  $('model-mode').title = session.language_model || 'Deterministic parser';
  $('message').placeholder = closedStages.has(session.stage)
    ? 'Start a new conversation to schedule another appointment.' : 'Message the scheduling assistant…';
  $('receipt').classList.toggle('hidden', !session.receipt);
  if (session.receipt) {
    const receipt = session.receipt;
    $('receipt').textContent = `${receipt.id}\n${receipt.doctor}\n${receipt.starts.slice(0, 10)} · ${receipt.starts.slice(11, 16)} IST`;
  }
  if (session.verified) {
    const key = Object.keys(patients).find(key => patients[key].name === session.patient_name);
    if (key) $('demo-patient').value = key;
  }
  updatePatient();
  $('quick-actions').replaceChildren();
  if (session.stage === 'choose') {
    session.offers.forEach((slot, index) => suggestedReply(
      `${index + 1}. ${slot.doctor} · ${slot.starts.slice(11, 16)} IST`, String(index + 1)));
  } else if (session.stage === 'confirm') {
    suggestedReply('Confirm appointment', 'yes');
    suggestedReply('Choose another time', 'no');
  } else if (session.stage === 'booking_uncertain') {
    suggestedReply('Check booking result', 'retry');
  } else if (session.stage === 'service_unavailable') {
    suggestedReply('Retry clinic service', 'retry');
  } else if (session.history.length === 1 || session.stage === 'request') {
    suggestedReply('General care · prefer afternoon', 'General next Monday with Dr Rao, prefer afternoon if possible');
    suggestedReply('Dermatology · next Monday', 'Dermatology next Monday');
  }
  $('tool-count').textContent = session.tools.length;
  $('tool-log').replaceChildren();
  if (!session.tools.length) {
    $('tool-log').append(element('p', 'empty-note', 'Clinic actions appear here as they happen.'));
  }
  for (const event of session.tools) {
    const row = element('div', 'tool-event');
    const succeeded = ['verified', 'ok', 'booked'].includes(event.outcome);
    const info = element('div');
    info.append(element('strong', '', toolNames[event.tool] || event.tool),
      element('small', '', event.outcome.replaceAll('_', ' ').toLowerCase()));
    row.append(element('span', 'tool-dot', succeeded ? '✓' : '·'), info,
      element('span', 'tool-turn', 'T' + event.turn));
    $('tool-log').append(row);
  }
  setBusy(busy);
}

async function newConversation() {
  setBusy(true);
  try {
    session = await api('/api/session', {baseline: $('conversation-policy').value === 'initial'});
    saveToken(session.session_id);
    pendingMessage = null;
    $('message').value = '';
    notice();
    renderSession();
  } catch (error) { notice(error.message); }
  finally { setBusy(false); }
}

async function refreshConversation() {
  if (!session) return newConversation();
  setBusy(true);
  try {
    session = await api('/api/session');
    if (pendingMessage && session.acknowledged_message_ids.includes(pendingMessage.message_id)) {
      pendingMessage = null;
      $('message').value = '';
    }
    renderSession();
    notice();
  } catch (error) {
    notice(error.message);
    if (error.status === 401) {
      session = null;
      saveToken(null);
      $('agent-status').textContent = 'Conversation expired — start a new one';
    }
  } finally { setBusy(false); }
}

async function sendMessage(message) {
  if (busy || !session || !message.trim() || closedStages.has(session.stage)) return;
  if (!pendingMessage || pendingMessage.message !== message) {
    pendingMessage = {message, message_id: crypto.randomUUID(), expected_turn: session.turn};
  }
  const bookingMayHaveCommitted = ['confirm', 'booking_uncertain'].includes(session.stage);
  setBusy(true);
  try {
    session = await api('/api/chat', pendingMessage);
    pendingMessage = null;
    $('message').value = '';
    notice();
    renderSession();
  } catch (error) {
    // Reading back the session resolves a lost response without repeating a booking.
    try {
      session = await api('/api/session');
      renderSession();
      if (session.acknowledged_message_ids.includes(pendingMessage.message_id)) {
        pendingMessage = null;
        $('message').value = '';
        notice('Your message was received. The latest conversation is shown below.');
      } else {
        pendingMessage = null;
        notice(error.message + ' Review the latest details before sending another message.');
      }
    } catch (recoveryError) {
      notice(recoveryError.message + (bookingMayHaveCommitted ? ' A booking may still have completed.' : ''));
      if (recoveryError.status === 401) {
        session = null;
        saveToken(null);
        $('agent-status').textContent = 'Conversation expired — start a new one';
      }
    }
  } finally {
    setBusy(false);
    if (!$('message').disabled) $('message').focus();
  }
}

function showView(name) {
  const lab = name === 'lab';
  $('lab-view').classList.toggle('hidden', !lab);
  $('chat-view').classList.toggle('hidden', lab);
  $('chat-nav').classList.toggle('active', !lab);
  $('lab-nav').classList.toggle('active', lab);
  $('chat-nav').setAttribute('aria-current', lab ? 'false' : 'page');
  $('lab-nav').setAttribute('aria-current', lab ? 'page' : 'false');
  $('breadcrumb').textContent = lab ? 'Improvement lab' : 'Conversation';
  window.scrollTo(0, 0);
}

function transcript(history, tools, title) {
  const column = element('div', 'transcript-column');
  column.append(element('h4', '', title));
  for (const turn of history) {
    const paragraph = element('p', turn.role);
    paragraph.append(element('strong', '', turn.role === 'user' ? 'Patient: ' : 'Agent: '),
      document.createTextNode(turn.text));
    column.append(paragraph);
  }
  column.append(element('h4', '', 'Tool audit'));
  for (const event of tools) {
    column.append(element('p', 'audit-line', `T${event.turn} · ${event.tool} → ${event.outcome}`));
  }
  return column;
}

function renderExperimentEvents(events = []) {
  $('experiment-log').classList.toggle('hidden', !events.length);
  $('experiment-events').replaceChildren();
  for (const event of events) {
    const row = element('li', '', event.message + (event.reused ? ' Saved stage reused.' : ''));
    if (event.failures?.length) row.append(element('small', '', event.failures.map(item => `${item.id}: ${item.codes.join(', ')}`).join(' · ')));
    if (event.patches?.length) row.append(element('small', '', event.patches.map(item => `${item.field} = ${JSON.stringify(item.value)}`).join(' · ')));
    $('experiment-events').append(row);
    if (event.stage === 'baseline') {
      $('before-score').textContent = event.score + '%';
      $('before-count').textContent = `${event.passed}/${event.total} scenarios passed`;
    } else if (event.stage === 'replay') {
      $('after-score').textContent = event.score + '%';
      $('after-count').textContent = `${event.passed}/${event.total} scenarios passed`;
    }
  }
}

function resetLoopDisplay() {
  ['before-score', 'after-score', 'regression-count', 'validation-score'].forEach(id => $(id).textContent = '—');
  $('before-count').textContent = 'Evaluating starting policy';
  $('after-count').textContent = 'Waiting for candidate replay';
  $('gate-label').textContent = 'Promotion requires zero';
  $('validation-count').textContent = 'Checked after candidate selection';
  ['step-run', 'step-repair', 'step-replay', 'step-promote'].forEach(id => {
    $(id).classList.remove('done');
    $(id).querySelector('b').textContent = '○';
  });
  $('patches').replaceChildren(element('p', 'empty-note', 'Waiting for the evaluation to identify failures.'));
  $('results').replaceChildren(element('p', 'empty-note', 'Results will appear when this experiment finishes.'));
  $('results-count').textContent = 'Evaluation in progress';
  $('download-report').classList.add('hidden');
  renderExperimentEvents();
}

function renderLoop(result) {
  renderExperimentEvents(result.events);
  if (result.model) {
    const models = result.proposal_model && result.proposal_model !== result.model
      ? `Conversations: ${result.model}. Repairs: ${result.proposal_model}.`
      : `Evaluated with ${result.model}.`;
    const requests = result.model_usage.requests;
    $('loop-progress').textContent = `${models} ${requests} API ${requests === 1 ? 'request' : 'requests'}; ${result.model_usage.cache_hits} identical requests reused.`;
  }
  $('before-score').textContent = result.before.score + '%';
  $('after-score').textContent = result.after.score + '%';
  $('before-count').textContent = `${result.before.passed}/${result.before.total} scenarios passed`;
  $('after-count').textContent = `${result.after.passed}/${result.after.total} scenarios passed`;
  $('regression-count').textContent = result.gate.regressions.length;
  $('gate-label').textContent = result.gate.promoted ? 'All safety checks passed' : 'Candidate rejected';
  $('validation-score').textContent = result.validation_after.score + '%';
  $('validation-count').textContent = `${result.validation_before.score}% before · ${result.validation_after.passed}/${result.validation_after.total} after`;
  const applied = result.applied_to_new_conversations;
  $('loop-badge').textContent = result.gate.promoted
    ? (applied === false ? 'Passed; activation failed — run again' : 'Improved policy active') : 'Candidate rejected';
  ['step-run', 'step-repair', 'step-replay', 'step-promote'].forEach((id, index) => {
    $(id).classList.add('done');
    $(id).querySelector('b').textContent = index === 3 && (!result.gate.promoted || applied === false) ? '!' : '✓';
  });
  $('patches').replaceChildren();
  if (result.proposal.analysis) $('patches').append(element('p', '', result.proposal.analysis));
  if (result.gate.reasons.length) $('patches').append(element('p', 'table-help', result.gate.reasons.join(' · ')));
  for (const patch of result.proposal.patches) {
    const card = element('div', 'patch');
    card.append(element('div', 'patch-code', patch.failure_code),
      element('strong', '', `${patch.field} = ${JSON.stringify(patch.value)}`),
      element('p', '', patch.reason),
      element('small', '', `${patch.evidence.length} failed runs support this change`));
    $('patches').append(card);
  }
  if (!result.proposal.patches.length) $('patches').append(element('p', 'empty-note', 'No supported policy changes were proposed.'));
  $('results-count').textContent = `${result.cases.filter(item => item.after_passed).length}/${result.cases.length} passed after replay`;
  $('download-report').classList.remove('hidden');
  $('results').replaceChildren();
  const ordered = [...result.cases].sort((a, b) => Number(a.before_passed) - Number(b.before_passed));
  for (const item of ordered) {
    const details = element('details', 'case-row');
    const summary = element('summary');
    const name = element('div', 'case-name');
    name.append(element('strong', '', item.id.replaceAll('_', ' ')), element('small', '', item.description));
    summary.append(name,
      element('span', item.before_passed ? 'pass-chip' : 'fail-chip', item.before_passed ? 'Before · Pass' : 'Before · Fail'),
      element('span', item.after_passed ? 'pass-chip' : 'fail-chip', item.after_passed ? 'After · Pass' : 'After · Fail'),
      element('span', 'expand-icon', '+'));
    details.append(summary);
    if (item.failure_codes.length) {
      details.append(element('p', 'table-help', 'Initial failure: ' + item.failure_codes.join(', ')));
    }
    const compare = element('div', 'transcript-compare');
    compare.append(transcript(item.before_history, item.before_tools, 'Before policy'),
      transcript(item.after_history, item.after_tools, 'Candidate policy'));
    details.append(compare);
    const checks = element('div', 'case-checks');
    for (const check of item.checks) {
      checks.append(element('span', check.passed ? 'check-pass' : 'check-fail',
        (check.passed ? '✓ ' : '✕ ') + check.name.replaceAll('_', ' ')));
    }
    details.append(checks);
    $('results').append(details);
  }
}

$('chat-form').addEventListener('submit', event => {
  event.preventDefault();
  sendMessage($('message').value);
});
$('message').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!busy) $('chat-form').requestSubmit();
  }
});
$('new-chat').addEventListener('click', newConversation);
$('refresh-conversation').addEventListener('click', refreshConversation);
$('demo-patient').addEventListener('change', updatePatient);
$('use-patient').addEventListener('click', () => {
  const patient = patients[$('demo-patient').value];
  sendMessage(`Name: ${patient.name}, DOB: ${patient.dob}`);
});
$('quick-actions').addEventListener('click', event => {
  const button = event.target.closest('[data-say]');
  if (button) sendMessage(button.dataset.say);
});
$('chat-nav').addEventListener('click', () => showView('chat'));
$('lab-nav').addEventListener('click', () => showView('lab'));
async function watchLoop(result) {
  resetLoopDisplay();
  $('run-loop').disabled = true;
  $('evaluation-policy').disabled = true;
  $('run-loop').textContent = 'Running evaluations…';
  $('loop-badge').textContent = 'Evaluating…';
  try {
    while (result.running) {
      $('loop-progress').textContent = result.progress || 'Evaluating…';
      renderExperimentEvents(result.events);
      await new Promise(resolve => setTimeout(resolve, 1000));
      result = await api('/api/evaluation');
    }
    renderExperimentEvents(result.events);
    if (result.loop_error) throw new Error(result.loop_error);
    if (!result.pending) renderLoop(result);
  }
  catch (error) {
    toast(error.message);
    $('loop-progress').textContent = error.message;
    $('loop-badge').textContent = 'Run failed — try again';
  }
  finally { $('run-loop').disabled = false; $('evaluation-policy').disabled = false; $('run-loop').textContent = 'Run improvement loop'; }
}
$('run-loop').addEventListener('click', async () => {
  $('run-loop').disabled = true;
  try { await watchLoop(await api('/api/improve', {baseline: $('evaluation-policy').value === 'initial'})); }
  catch (error) { toast(error.message); $('run-loop').disabled = false; }
});

async function start() {
  setBusy(true);
  try {
    const health = await api('/api/health');
    $('training-count').textContent = health.mode === 'gemini' ? '32' : '24';
    $('before-count').textContent = `${health.mode === 'gemini' ? 32 : 24} multi-turn scenarios`;
    $('validation-count').textContent = `${health.mode === 'gemini' ? 12 : 6} separate scenarios`;
    $('service-note').textContent = health.mode === 'gemini' ? 'Language powered by Gemini.' : 'Offline rules · no API required.';
    $('lab-description').textContent = health.mode === 'gemini'
      ? 'Choose the current policy or start a new initial-policy comparison. Gemini proposes repairs from training failures, then the candidate is replayed and validated. Identical model responses are cached to reduce quota use.'
      : 'Offline demonstration: fixed repair rules, starting from the initial policy. Use Gemini mode for generated improvements.';
    $('proposal-description').textContent = health.mode === 'gemini'
      ? 'Gemini writes policy and interpretation changes from failed training conversations.' : 'Apply predefined repairs to supported failure codes.';
  } catch (error) { notice(error.message); }
  const token = savedToken();
  if (token) {
    session = {session_id: token};
    try { session = await api('/api/session'); renderSession(); }
    catch (error) {
      session = null;
      if (error.status === 401) saveToken(null);
      $('agent-status').textContent = 'Start a new conversation';
      notice(error.message);
    }
    finally { setBusy(false); }
  } else { await newConversation(); }
  try {
    const result = await api('/api/evaluation');
    if (result.running) watchLoop(result);
    else if (result.loop_error) {
      $('loop-progress').textContent = result.loop_error;
      $('loop-badge').textContent = 'Last run failed';
    }
    else if (!result.pending) renderLoop(result);
  } catch (error) { toast(error.message); }
}
start();
