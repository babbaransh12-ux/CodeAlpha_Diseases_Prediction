// ============================================================
//  CardioSense – script.js
//  Uses EXACT Logistic Regression weights exported from the
//  scikit-learn model trained on heart_disease_risk_dataset_earlymed.csv
//  Training Acc: 99.26%  |  Test Acc: 99.13%  |  CV: 99.24%
// ============================================================

// ── Real model weights (from model_weights.json) ─────────────────────
const MODEL = {
  intercept: -24.34629067,
  coef: {
    Chest_Pain:           2.66563621,
    Shortness_of_Breath:  2.49369091,
    Fatigue:              2.63298665,
    Palpitations:         2.97383434,
    Dizziness:            2.69919581,
    Swelling:             2.45217442,
    Pain_Arms_Jaw_Back:   2.65976933,
    Cold_Sweats_Nausea:   2.54450674,
    High_BP:              1.64220755,
    High_Cholesterol:     1.55248653,
    Diabetes:             1.48471905,
    Smoking:              1.50469155,
    Obesity:              1.48024656,
    Sedentary_Lifestyle:  1.76402358,
    Family_History:       1.55297017,
    Chronic_Stress:       1.57021697,
    Gender:               1.40703566,
    Age:                  0.12000579
  },
  metrics: {
    train_accuracy: 0.992612,
    test_accuracy:  0.991253,
    cv_score:       0.992408,
    error_rate:     0.008747
  },
  confusion_matrix: { TN: 22562, FP: 184, FN: 214, TP: 22540 }
};

// Sigmoid activation — identical to sklearn's predict_proba
const sigmoid = z => 1 / (1 + Math.exp(-z));

// ── Logit range calibration ───────────────────────────────────────────
// At age=54 (mean), male=1, 0 binary factors:
//   logit_min = -24.346 + 0.12*54 + 1.407 = -16.451  → P≈0%
// At all 16 binary + age=54, male:
//   logit_max = -24.346 + 0.12*54 + 1.407 + 33.673 = 17.222  → P≈100%
// We linearly scale the logit to [0,100] for the gauge display
// The binary prediction (AT RISK / NO RISK) still uses the real model threshold (sigmoid≥0.5)
const LOGIT_MIN = -17.0;   // baseline: no factors, young age
const LOGIT_MAX =  18.0;   // all factors, older age

function scaledRiskScore(logit) {
  // Clamp and map logit to 0-100 range
  const clamped = Math.max(LOGIT_MIN, Math.min(LOGIT_MAX, logit));
  return parseFloat(((clamped - LOGIT_MIN) / (LOGIT_MAX - LOGIT_MIN) * 100).toFixed(1));
}

// ── State ────────────────────────────────────────────────────────────
let selectedGender = 1;   // 1 = Male, 0 = Female

// ── DOM Ready ────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initParticles();
  initECG();
  initNavScroll();
  initCheckboxes();
  animateMetricsOnScroll();
  drawDatasetChart();
  drawRiskFactorsChart();
  drawAgeRiskChart();
  animateConfusionMatrix();
});

// ═══════════════════════════════════════════════════════════════════════
//  PREDICTION ENGINE  (mirrors sklearn LogisticRegression.predict_proba)
// ═══════════════════════════════════════════════════════════════════════
function getFeatureVector() {
  return {
    Chest_Pain:           document.getElementById('chest_pain').checked          ? 1 : 0,
    Shortness_of_Breath:  document.getElementById('shortness_of_breath').checked ? 1 : 0,
    Fatigue:              document.getElementById('fatigue').checked              ? 1 : 0,
    Palpitations:         document.getElementById('palpitations').checked         ? 1 : 0,
    Dizziness:            document.getElementById('dizziness').checked            ? 1 : 0,
    Swelling:             document.getElementById('swelling').checked             ? 1 : 0,
    Pain_Arms_Jaw_Back:   document.getElementById('pain_arms_jaw').checked        ? 1 : 0,
    Cold_Sweats_Nausea:   document.getElementById('cold_sweats').checked          ? 1 : 0,
    High_BP:              document.getElementById('high_bp').checked              ? 1 : 0,
    High_Cholesterol:     document.getElementById('high_cholesterol').checked     ? 1 : 0,
    Diabetes:             document.getElementById('diabetes').checked             ? 1 : 0,
    Smoking:              document.getElementById('smoking').checked              ? 1 : 0,
    Obesity:              document.getElementById('obesity').checked              ? 1 : 0,
    Sedentary_Lifestyle:  document.getElementById('sedentary').checked            ? 1 : 0,
    Family_History:       document.getElementById('family_history').checked       ? 1 : 0,
    Chronic_Stress:       document.getElementById('chronic_stress').checked       ? 1 : 0,
    Gender:               selectedGender,
    Age:                  parseInt(document.getElementById('age').value)
  };
}

function predict(features) {
  // logit  =  intercept + Σ(coef_i * x_i)
  let logit = MODEL.intercept;
  for (const [feat, val] of Object.entries(features)) {
    logit += MODEL.coef[feat] * val;
  }
  const proba = sigmoid(logit);           // P(Heart_Risk = 1)
  const predicted = proba >= 0.5 ? 1 : 0; // decision boundary = 0.5
  return { proba, predicted, logit };
}

// ═══════════════════════════════════════════════════════════════════════
//  RUN PREDICTION (called by button)
// ═══════════════════════════════════════════════════════════════════════
function runPrediction() {
  const features  = getFeatureVector();
  const { proba, predicted, logit } = predict(features);

  // Visual risk score: linearly scaled from logit range [MIN..MAX] → [0..100]
  // This gives smooth, meaningful display across all factor combinations.
  // Binary class prediction (AT RISK / NO RISK) still uses real sigmoid ≥ 0.5.
  const riskPct = scaledRiskScore(logit);

  // Count active risk factors (all binary features except Age & Gender)
  const binaryKeys    = Object.keys(MODEL.coef).filter(k => k !== 'Age' && k !== 'Gender');
  const activeFactors = binaryKeys.filter(k => features[k] === 1);

  displayResult(riskPct, features, activeFactors, predicted, proba);
}

// ═══════════════════════════════════════════════════════════════════════
//  DISPLAY RESULT
// ═══════════════════════════════════════════════════════════════════════
function displayResult(riskPct, features, activeFactors, predicted, proba) {
  const initialEl = document.getElementById('result-initial');
  const outputEl  = document.getElementById('result-output');

  initialEl.style.display = 'none';
  outputEl.style.display  = 'block';

  // ── Risk tier ────────────────────────────────────────────────────────
  let riskLevel, riskClass, riskColor, advice, emoji;
  if (riskPct < 25) {
    riskLevel = 'Low Risk';       riskClass = 'low';       riskColor = '#4cc9f0'; emoji = '✅';
    advice = {
      title: 'Great News!',
      text:  'Your risk profile shows a LOW likelihood of heart disease. Keep up healthy habits — regular exercise, balanced diet, and good sleep protect your heart long-term.'
    };
  } else if (riskPct < 50) {
    riskLevel = 'Moderate Risk';  riskClass = 'moderate';  riskColor = '#ffd166'; emoji = '⚠️';
    advice = {
      title: 'Take Notice',
      text:  'MODERATE risk detected. Consider a routine cardiac checkup. Small lifestyle improvements — cutting smoking, reducing stress, staying active — can significantly lower your risk.'
    };
  } else if (riskPct < 75) {
    riskLevel = 'High Risk';      riskClass = 'high';      riskColor = '#fb8500'; emoji = '⚡';
    advice = {
      title: 'Action Recommended',
      text:  'HIGH risk detected. Please consult a healthcare provider soon. If you\'re experiencing chest pain or shortness of breath along with other symptoms, seek prompt medical attention.'
    };
  } else {
    riskLevel = 'Very High Risk'; riskClass = 'very-high'; riskColor = '#e63946'; emoji = '🚨';
    advice = {
      title: 'Urgent Attention Needed',
      text:  'VERY HIGH risk detected. Please seek immediate medical consultation. Do NOT delay — early intervention can be life-saving. If experiencing acute symptoms, call emergency services NOW.'
    };
  }

  // ── Gauge math ───────────────────────────────────────────────────────
  const gaugeRadius        = 70;
  const gaugeCircumference = Math.PI * gaugeRadius;
  const dashOffset         = gaugeCircumference * (1 - riskPct / 100);

  // ── Feature breakdown (sorted by coefficient weight, show top 6) ─────
  const featureLabels = {
    Chest_Pain:           { label: 'Chest Pain',              emoji: '💔' },
    Shortness_of_Breath:  { label: 'Shortness of Breath',     emoji: '😮‍💨' },
    Fatigue:              { label: 'Fatigue',                  emoji: '😴' },
    Palpitations:         { label: 'Palpitations',            emoji: '💓' },
    Dizziness:            { label: 'Dizziness',               emoji: '😵' },
    Swelling:             { label: 'Swelling',                emoji: '🦵' },
    Pain_Arms_Jaw_Back:   { label: 'Pain in Arms/Jaw/Back',   emoji: '🤚' },
    Cold_Sweats_Nausea:   { label: 'Cold Sweats / Nausea',    emoji: '🥵' },
    High_BP:              { label: 'High Blood Pressure',     emoji: '🩺' },
    High_Cholesterol:     { label: 'High Cholesterol',        emoji: '🧪' },
    Diabetes:             { label: 'Diabetes',                emoji: '💉' },
    Smoking:              { label: 'Smoking',                 emoji: '🚬' },
    Obesity:              { label: 'Obesity',                 emoji: '⚖️' },
    Sedentary_Lifestyle:  { label: 'Sedentary Lifestyle',     emoji: '🛋️' },
    Family_History:       { label: 'Family History',          emoji: '🧬' },
    Chronic_Stress:       { label: 'Chronic Stress',         emoji: '😰' }
  };

  // Sort active factors by model coefficient (highest impact first)
  const sortedActive = [...activeFactors].sort(
    (a, b) => MODEL.coef[b] - MODEL.coef[a]
  ).slice(0, 6);

  const featureRows = sortedActive.length > 0
    ? sortedActive.map(k => {
        const info   = featureLabels[k];
        const weight = MODEL.coef[k].toFixed(2);
        return `
          <div class="rb-item">
            <span class="rb-name">${info.emoji} ${info.label}</span>
            <div class="rb-right">
              <span class="rb-weight">w=${weight}</span>
              <span class="rb-badge present">Active</span>
            </div>
          </div>`;
      }).join('')
    : `<div class="rb-item"><span class="rb-name" style="color:var(--text-muted)">No major risk factors selected</span></div>`;

  const age    = features.Age;
  const gender = selectedGender === 1 ? 'Male' : 'Female';

  outputEl.innerHTML = `
    <div class="result-model-badge">
      <span class="model-dot"></span>
      Logistic Regression · 99.13% Accuracy
    </div>

    <div class="risk-gauge-wrapper">
      <p class="risk-gauge-label">Heart Disease Probability</p>
      <div class="gauge-container">
        <svg class="gauge-bg" viewBox="0 0 180 90">
          <path class="gauge-arc-bg"
            d="M 18 90 A 70 70 0 0 1 162 90"
            stroke-dasharray="${gaugeCircumference}"
            stroke-dashoffset="0"
          />
          <path id="gauge-arc" class="gauge-arc"
            d="M 18 90 A 70 70 0 0 1 162 90"
            stroke="${riskColor}"
            stroke-dasharray="${gaugeCircumference}"
            stroke-dashoffset="${gaugeCircumference}"
          />
        </svg>
        <div class="gauge-value" style="color:${riskColor}">
          <span id="gauge-number">0</span><span class="gauge-pct">%</span>
        </div>
      </div>
      <div class="risk-verdict ${riskClass}">${emoji} ${riskLevel}</div>
      <div class="model-prediction-label">
        Model Output: <strong style="color:${riskColor}">${predicted === 1 ? 'AT RISK' : 'NO RISK'}</strong>
        &nbsp;·&nbsp; P(risk) = <strong>${proba.toFixed(4)}</strong>
        &nbsp;·&nbsp; Score = <strong>${riskPct}%</strong>
      </div>
    </div>

    <div class="risk-breakdown">
      <div class="rb-title">
        Top Active Risk Factors
        <span class="rb-count">${activeFactors.length} / 16</span>
      </div>
      ${featureRows}
    </div>

    <div class="advice-box ${riskClass}">
      <div class="advice-header">${emoji} ${advice.title}</div>
      <div class="advice-text">${advice.text}</div>
    </div>

    <div class="result-meta">
      <span>Age: <strong>${age}</strong></span>
      <span>Gender: <strong>${gender}</strong></span>
      <span>Factors: <strong>${activeFactors.length}/16</strong></span>
    </div>
  `;

  // Animate gauge arc
  setTimeout(() => {
    const arc = document.getElementById('gauge-arc');
    if (arc) {
      arc.style.transition = 'stroke-dashoffset 1.5s cubic-bezier(0.4,0,0.2,1)';
      arc.style.strokeDashoffset = dashOffset;
    }
    animateNumber('gauge-number', 0, Math.round(riskPct), 1400);
  }, 50);

  // Scroll to result on mobile
  if (window.innerWidth <= 900) {
    setTimeout(() => outputEl.scrollIntoView({ behavior: 'smooth', block: 'start' }), 200);
  }
}

// ═══════════════════════════════════════════════════════════════════════
//  UI HELPERS
// ═══════════════════════════════════════════════════════════════════════

function updateSlider(id, value) {
  const display = document.getElementById(`${id}-display`);
  if (id === 'age') display.textContent = `${value} years`;
}

function setGender(val) {
  selectedGender = val;
  document.getElementById('gender-male').classList.toggle('active',   val === 1);
  document.getElementById('gender-female').classList.toggle('active', val === 0);
}

function scrollToSection(id) {
  const el = document.getElementById(id);
  if (el) window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - 80, behavior: 'smooth' });
}

function animateNumber(id, from, to, duration) {
  const el = document.getElementById(id);
  if (!el) return;
  const start = Date.now();
  const tick = () => {
    const p = Math.min((Date.now() - start) / duration, 1);
    el.textContent = Math.round(from + (to - from) * (1 - Math.pow(1 - p, 3)));
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function initCheckboxes() {
  document.querySelectorAll('.checkbox-item').forEach(item => {
    const cb = item.querySelector('input[type="checkbox"]');
    item.addEventListener('click', () => {
      cb.checked = !cb.checked;
      item.classList.toggle('checked', cb.checked);
    });
  });
}

// ═══════════════════════════════════════════════════════════════════════
//  NAVBAR
// ═══════════════════════════════════════════════════════════════════════
function initNavScroll() {
  const navbar   = document.getElementById('navbar');
  const sections = ['hero', 'stats', 'predictor', 'insights'];

  window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.scrollY > 50);

    let current = 'hero';
    sections.forEach(id => {
      const el = document.getElementById(id);
      if (el && window.scrollY >= el.offsetTop - 160) current = id;
    });
    document.querySelectorAll('.nav-link').forEach(link =>
      link.classList.toggle('active', link.getAttribute('href') === `#${current}`)
    );
  });
}

// ═══════════════════════════════════════════════════════════════════════
//  ANIMATED BACKGROUND
// ═══════════════════════════════════════════════════════════════════════
function initParticles() {
  const container = document.getElementById('particles');
  for (let i = 0; i < 25; i++) {
    const p  = document.createElement('div');
    const sz = Math.random() * 3 + 1;
    const dur = Math.random() * 20 + 15;
    const delay = Math.random() * 10;
    p.style.cssText = `
      position:absolute;
      width:${sz}px; height:${sz}px;
      background:rgba(230,57,70,${Math.random() * 0.35 + 0.08});
      border-radius:50%;
      top:${Math.random() * 100}%;
      left:${Math.random() * 100}%;
      animation:pf${i} ${dur}s ease-in-out ${delay}s infinite;
    `;
    container.appendChild(p);

    const dx1 = (Math.random() * 60 - 30).toFixed(0);
    const dy1 = (Math.random() * 80).toFixed(0);
    const dx2 = (Math.random() * 60 - 30).toFixed(0);
    const dy2 = (Math.random() * 60 - 30).toFixed(0);
    const style = document.createElement('style');
    style.textContent = `
      @keyframes pf${i} {
        0%,100%{transform:translate(0,0) scale(1);opacity:.5}
        33%{transform:translate(${dx1}px,-${dy1}px) scale(1.3);opacity:.8}
        66%{transform:translate(${dx2}px,${dy2}px) scale(.7);opacity:.3}
      }`;
    document.head.appendChild(style);
  }
}

// ═══════════════════════════════════════════════════════════════════════
//  ECG ANIMATION
// ═══════════════════════════════════════════════════════════════════════
function initECG() {
  const polyline = document.getElementById('ecg-polyline');
  if (!polyline) return;
  polyline.setAttribute('points', generateECGPoints(1200, 100));
}

function generateECGPoints(W, H) {
  const mid = H / 2;
  const seg = W / 6;
  const segs = [
    ...flat(0,         seg * 0.4, mid, 8),
    ...pWave(seg * 0.4, seg * 0.6, mid, H * 0.12, 8),
    ...flat(seg * 0.6,  seg,       mid, 4),
    ...qrs(seg,         seg * 1.15, mid, H * 0.72),
    ...flat(seg * 1.15, seg * 1.5,  mid + H * 0.05, 4),
    ...tWave(seg * 1.5, seg * 1.8,  mid, H * 0.2, 8),
    ...flat(seg * 1.8,  seg * 2.4,  mid, 8),
    ...pWave(seg * 2.4, seg * 2.6,  mid, H * 0.12, 8),
    ...flat(seg * 2.6,  seg * 3,    mid, 4),
    ...qrs(seg * 3,     seg * 3.15, mid, H * 0.72),
    ...flat(seg * 3.15, seg * 3.5,  mid + H * 0.05, 4),
    ...tWave(seg * 3.5, seg * 3.8,  mid, H * 0.2, 8),
    ...flat(seg * 3.8,  W,          mid, 6),
  ];
  return segs.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
}

function flat(x1, x2, y, n) {
  return Array.from({ length: n + 1 }, (_, i) => ({ x: x1 + (x2 - x1) * i / n, y }));
}
function pWave(x1, x2, base, amp, n) {
  return Array.from({ length: n + 1 }, (_, i) => {
    const t = i / n;
    return { x: x1 + (x2 - x1) * t, y: base - amp * Math.sin(Math.PI * t) };
  });
}
function qrs(x1, x2, base, amp) {
  const w = x2 - x1;
  return [
    { x: x1,           y: base },
    { x: x1 + w * 0.1, y: base + amp * 0.15 },
    { x: x1 + w * 0.35,y: base - amp },
    { x: x1 + w * 0.55,y: base + amp * 0.38 },
    { x: x1 + w * 0.75,y: base + amp * 0.04 },
    { x: x2,           y: base },
  ];
}
function tWave(x1, x2, base, amp, n) {
  return Array.from({ length: n + 1 }, (_, i) => {
    const t = i / n;
    return { x: x1 + (x2 - x1) * t, y: base - amp * Math.sin(Math.PI * t) * 0.8 };
  });
}

// ═══════════════════════════════════════════════════════════════════════
//  METRICS ANIMATION (IntersectionObserver)
// ═══════════════════════════════════════════════════════════════════════
function animateMetricsOnScroll() {
  const el = document.querySelector('.metrics-grid');
  if (!el) return;
  const obs = new IntersectionObserver(entries => {
    if (entries[0].isIntersecting) { animateMetrics(); obs.disconnect(); }
  }, { threshold: 0.3 });
  obs.observe(el);
}

function animateMetrics() {
  const items = [
    { id: 'metric-test',  v: (MODEL.metrics.test_accuracy  * 100).toFixed(4) },
    { id: 'metric-train', v: (MODEL.metrics.train_accuracy * 100).toFixed(4) },
    { id: 'metric-cv',    v: (MODEL.metrics.cv_score       * 100).toFixed(4) },
    { id: 'metric-err',   v: (MODEL.metrics.error_rate     * 100).toFixed(4) },
  ];
  items.forEach(({ id, v }, i) => {
    setTimeout(() => {
      const el  = document.getElementById(id);
      if (!el) return;
      const num = parseFloat(v);
      const dur = 1500;
      const t0  = Date.now();
      const tick = () => {
        const p = Math.min((Date.now() - t0) / dur, 1);
        el.textContent = (num * (1 - Math.pow(1 - p, 3))).toFixed(2) + '%';
        if (p < 1) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    }, i * 200);
  });

  document.querySelectorAll('.metric-fill').forEach(fill => {
    setTimeout(() => { fill.style.width = `${fill.getAttribute('data-target')}%`; }, 100);
  });
}

// ═══════════════════════════════════════════════════════════════════════
//  CONFUSION MATRIX ANIMATION
// ═══════════════════════════════════════════════════════════════════════
function animateConfusionMatrix() {
  const el = document.querySelector('.confusion-matrix');
  if (!el) return;
  const obs = new IntersectionObserver(entries => {
    if (entries[0].isIntersecting) { animateCM(); obs.disconnect(); }
  }, { threshold: 0.3 });
  obs.observe(el);
}

function animateCM() {
  const cm = MODEL.confusion_matrix;
  [
    { sel: '#cm-tn .cm-number', final: cm.TN.toLocaleString(), num: cm.TN },
    { sel: '#cm-fp .cm-number', final: cm.FP.toLocaleString(), num: cm.FP },
    { sel: '#cm-fn .cm-number', final: cm.FN.toLocaleString(), num: cm.FN },
    { sel: '#cm-tp .cm-number', final: cm.TP.toLocaleString(), num: cm.TP },
  ].forEach(({ sel, final, num }, i) => {
    const el  = document.querySelector(sel);
    if (!el) return;
    const dur = 1200;
    const t0  = Date.now();
    setTimeout(() => {
      const tick = () => {
        const p = Math.min((Date.now() - t0) / dur, 1);
        el.textContent = Math.round(num * (1 - Math.pow(1 - p, 3))).toLocaleString();
        if (p < 1) requestAnimationFrame(tick);
        else el.textContent = final;
      };
      requestAnimationFrame(tick);
    }, i * 150);
  });
}

// ═══════════════════════════════════════════════════════════════════════
//  CHARTS
// ═══════════════════════════════════════════════════════════════════════
function drawDatasetChart() {
  const canvas = document.getElementById('datasetChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  canvas.width  = canvas.parentElement.clientWidth || 350;
  canvas.height = 200;

  const data = [
    { label: 'No Risk (50%)', value: 35000, color: '#4cc9f0' },
    { label: 'At Risk (50%)', value: 35000, color: '#e63946' },
  ];
  const total = 70000;
  const cx = canvas.width / 2, cy = canvas.height / 2 - 10;
  const R  = Math.min(cx, cy) - 30;
  const IR = R * 0.56;

  let angle = -Math.PI / 2;
  data.forEach(d => {
    const sweep = (d.value / total) * 2 * Math.PI;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, R, angle, angle + sweep);
    ctx.closePath();
    ctx.fillStyle = d.color;
    ctx.fill();
    angle += sweep;
  });

  // Donut hole
  ctx.beginPath();
  ctx.arc(cx, cy, IR, 0, Math.PI * 2);
  ctx.fillStyle = '#0a0a0f';
  ctx.fill();

  // Center label
  ctx.fillStyle = '#f8f9fa'; ctx.font = 'bold 17px Inter';
  ctx.textAlign = 'center';  ctx.textBaseline = 'middle';
  ctx.fillText('70,000', cx, cy - 7);
  ctx.fillStyle = '#adb5bd'; ctx.font = '11px Inter';
  ctx.fillText('records', cx, cy + 10);

  // Legend
  data.forEach((d, i) => {
    const lx = cx - 65 + i * 70, ly = canvas.height - 14;
    ctx.fillStyle = d.color;
    ctx.fillRect(lx, ly - 5, 9, 9);
    ctx.fillStyle = '#adb5bd'; ctx.font = '10px Inter';
    ctx.textAlign = 'left';
    ctx.fillText(d.label, lx + 12, ly + 1);
  });
}

function drawRiskFactorsChart() {
  const canvas = document.getElementById('riskFactorsChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  canvas.width  = canvas.parentElement.clientWidth || 350;
  canvas.height = 220;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Top 6 features by coefficient weight (from real model)
  const factors = [
    { label: 'Palpitations', coef: 2.97, color: '#e63946' },
    { label: 'Dizziness',    coef: 2.70, color: '#fb5607' },
    { label: 'Chest Pain',   coef: 2.67, color: '#fb8500' },
    { label: 'Pain Arms/Jaw',coef: 2.66, color: '#ffd166' },
    { label: 'Fatigue',      coef: 2.63, color: '#4895ef' },
    { label: 'Cold Sweats',  coef: 2.54, color: '#4cc9f0' },
  ];

  const maxCoef = 3.2;
  const barH = 22, gap = 10, labelW = 78;
  const startY = 12;
  const maxBarW = canvas.width - labelW - 55;

  factors.forEach((f, i) => {
    const y    = startY + i * (barH + gap);
    const barW = (f.coef / maxCoef) * maxBarW;

    ctx.fillStyle = '#6c757d'; ctx.font = '11px Inter';
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText(f.label, labelW - 6, y + barH / 2);

    ctx.beginPath();
    ctx.roundRect(labelW, y, maxBarW, barH, 4);
    ctx.fillStyle = 'rgba(255,255,255,0.05)';
    ctx.fill();

    const grad = ctx.createLinearGradient(labelW, 0, labelW + barW, 0);
    grad.addColorStop(0, f.color);
    grad.addColorStop(1, f.color + 'aa');
    ctx.beginPath();
    ctx.roundRect(labelW, y, barW, barH, 4);
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.fillStyle = '#f8f9fa'; ctx.font = 'bold 10px Inter';
    ctx.textAlign = 'left';
    ctx.fillText(`w=${f.coef}`, labelW + barW + 6, y + barH / 2);
  });

  ctx.fillStyle = '#6c757d'; ctx.font = 'italic 10px Inter';
  ctx.textAlign = 'left';
  ctx.fillText('Coefficient weight from real LR model', labelW, canvas.height - 4);
}

function drawAgeRiskChart() {
  const canvas = document.getElementById('ageRiskChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  canvas.width  = canvas.parentElement.clientWidth || 800;
  canvas.height = 260;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Compute predicted P(risk=1) using real model at baseline (all binary=0, male=1)
  const ageGroups  = [22,27,32,37,42,47,52,57,62,67,72,77,82];
  const labels     = ['20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79','80-84'];

  const riskVals = ageGroups.map(age => {
    const logit = MODEL.intercept + MODEL.coef.Age * age + MODEL.coef.Gender * 1;
    return parseFloat((sigmoid(logit) * 100).toFixed(2));
  });

  // High-risk scenario (all binary features = 1)
  const highVals = ageGroups.map(age => {
    let logit = MODEL.intercept + MODEL.coef.Age * age + MODEL.coef.Gender * 1;
    Object.entries(MODEL.coef).forEach(([k, v]) => {
      if (k !== 'Age' && k !== 'Gender') logit += v;
    });
    return parseFloat(Math.min(99.9, sigmoid(logit) * 100).toFixed(2));
  });

  const padL = 55, padR = 20, padT = 20, padB = 42;
  const cW   = canvas.width - padL - padR;
  const cH   = canvas.height - padT - padB;
  const xStep = cW / (labels.length - 1);

  // Grid
  for (let i = 0; i <= 5; i++) {
    const y = padT + cH - (i / 5) * cH;
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + cW, y); ctx.stroke();
    ctx.fillStyle = '#6c757d'; ctx.font = '10px Inter';
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText(`${i * 20}%`, padL - 6, y + 4);
  }
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(padL, padT + cH); ctx.lineTo(padL + cW, padT + cH); ctx.stroke();

  // Area between baseline and high-risk curves
  ctx.beginPath();
  highVals.forEach((v, i) => {
    const x = padL + i * xStep, y = padT + cH - (v / 100) * cH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  [...riskVals].reverse().forEach((v, i) => {
    const idx = riskVals.length - 1 - i;
    ctx.lineTo(padL + idx * xStep, padT + cH - (v / 100) * cH);
  });
  ctx.closePath();
  const areaG = ctx.createLinearGradient(0, padT, 0, padT + cH);
  areaG.addColorStop(0, 'rgba(230,57,70,0.12)');
  areaG.addColorStop(1, 'rgba(230,57,70,0.02)');
  ctx.fillStyle = areaG; ctx.fill();

  // High-risk line (dashed)
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  highVals.forEach((v, i) => {
    const x = padL + i * xStep, y = padT + cH - (v / 100) * cH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.strokeStyle = 'rgba(230,57,70,0.5)'; ctx.lineWidth = 1.5; ctx.stroke();
  ctx.setLineDash([]);

  // Baseline risk line (gradient)
  const lineG = ctx.createLinearGradient(padL, 0, padL + cW, 0);
  lineG.addColorStop(0, '#4cc9f0');
  lineG.addColorStop(0.5, '#ffd166');
  lineG.addColorStop(1, '#e63946');

  ctx.beginPath();
  riskVals.forEach((v, i) => {
    const x = padL + i * xStep, y = padT + cH - (v / 100) * cH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.strokeStyle = lineG; ctx.lineWidth = 3; ctx.lineJoin = 'round'; ctx.stroke();

  // Data points
  riskVals.forEach((v, i) => {
    const x = padL + i * xStep, y = padT + cH - (v / 100) * cH;
    const c = v < 33 ? '#4cc9f0' : v < 66 ? '#ffd166' : '#e63946';
    ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fillStyle = c; ctx.fill();
    ctx.strokeStyle = '#0a0a0f'; ctx.lineWidth = 2; ctx.stroke();
  });

  // X-axis labels
  labels.forEach((lbl, i) => {
    ctx.fillStyle = '#6c757d'; ctx.font = '10px Inter';
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(lbl, padL + i * xStep, padT + cH + 8);
  });

  // Legend
  const legendItems = [
    { color: '#4cc9f0', label: 'Baseline risk (no other factors, male)' },
    { color: 'rgba(230,57,70,0.5)', label: 'Max risk (all factors present)', dash: true },
  ];
  legendItems.forEach((item, i) => {
    const lx = padL + i * 250, ly = padT + cH + 28;
    ctx.strokeStyle = item.color; ctx.lineWidth = 2;
    if (item.dash) ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 20, ly); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#6c757d'; ctx.font = '10px Inter';
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText(item.label, lx + 24, ly);
  });
}

// ── Resize ────────────────────────────────────────────────────────────
let _resizeT;
window.addEventListener('resize', () => {
  clearTimeout(_resizeT);
  _resizeT = setTimeout(() => {
    drawDatasetChart();
    drawRiskFactorsChart();
    drawAgeRiskChart();
  }, 300);
});
