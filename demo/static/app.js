(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const input = $('text-input');
  const btn = $('analyze-btn');
  const errBox = $('error');
  const resultBox = $('result');

  const showError = (msg) => { errBox.textContent = msg; errBox.hidden = false; };
  const pct = (v) => (v * 100).toFixed(1) + '%';

  function fillTags(listEl, items, cls) {
    listEl.className = 'tags ' + cls;
    listEl.replaceChildren();
    if (!items.length) {
      const li = document.createElement('li');
      li.className = 'none';
      li.textContent = '— none —';
      listEl.appendChild(li);
      return;
    }
    items.forEach((it) => {
      const li = document.createElement('li');
      li.textContent = it.word;
      listEl.appendChild(li);
    });
  }

  function render(r) {
    const positive = r.label === 'positive';
    $('emoji').textContent = positive ? '😊' : '😞';
    const label = $('label');
    label.textContent = r.label;
    label.className = 'label ' + (positive ? 'positive' : 'negative');
    $('pct-pos').textContent = pct(r.positive);
    $('pct-neg').textContent = pct(r.negative);
    $('bar-pos').style.width = '0%';
    $('bar-neg').style.width = '0%';
    requestAnimationFrame(() => requestAnimationFrame(() => {
      $('bar-pos').style.width = pct(r.positive);
      $('bar-neg').style.width = pct(r.negative);
    }));
    fillTags($('top-pos'), r.top_positive, 'pos');
    fillTags($('top-neg'), r.top_negative, 'neg');
    $('cleaned').textContent = r.cleaned || '(empty after cleaning)';
    resultBox.hidden = false;
  }

  async function analyze() {
    errBox.hidden = true;
    const text = input.value.trim();
    if (!text) { showError('Please enter some text first.'); return; }
    btn.disabled = true;
    btn.textContent = 'Analyzing…';
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      if (!res.ok) { showError(data.error || 'Something went wrong.'); resultBox.hidden = true; return; }
      render(data);
    } catch (e) {
      showError('Cannot reach the server.');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Analyze sentiment';
    }
  }

  btn.addEventListener('click', analyze);
  input.addEventListener('keydown', (e) => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') analyze(); });
  input.addEventListener('input', () => { $('counter').textContent = input.value.length + ' / 5000'; });
  $('clear-btn').addEventListener('click', () => {
    input.value = ''; input.dispatchEvent(new Event('input'));
    resultBox.hidden = true; errBox.hidden = true; input.focus();
  });
  document.querySelectorAll('.chip').forEach((c) => c.addEventListener('click', () => {
    input.value = c.dataset.text; input.dispatchEvent(new Event('input')); analyze();
  }));

  fetch('/api/info').then((r) => r.json()).then((d) => {
    $('model-badge').textContent = 'Naive Bayes · test accuracy ' + (d.accuracy * 100).toFixed(2) + '%';
  }).catch(() => { $('model-badge').textContent = 'Model ready'; });
})();
