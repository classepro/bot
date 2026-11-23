// Oracle de fidélité : exécute le code de ROBOTV4.5.html sans le réécrire.
// Les blocs sont EXTRAITS du fichier d'origine par découpage textuel, puis
// évalués. Si la reproduction Python est juste, les deux donnent le même
// résultat sur les mêmes saisies.
//
// Usage : node oracle.js '<json {"inputs": ["1.5,2.0,..."], "settings": {...}}>'
'use strict';
const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, '..', 'ROBOTV4.5.html'), 'utf8');

function grabFunction(name) {
  const re = new RegExp('function ' + name + '\\s*\\([^)]*\\)\\s*\\{', 'm');
  const m = re.exec(html);
  if (!m) throw new Error('fonction introuvable : ' + name);
  let i = m.index + m[0].length;
  let depth = 1;
  while (i < html.length && depth > 0) {
    const c = html[i];
    if (c === '{') depth++;
    else if (c === '}') depth--;
    i++;
  }
  return html.slice(m.index, i);
}

// L742 -> avant `const predictionResult` : le calcul pur, tel quel.
const calcStart = html.indexOf('let oddsArray = oddsInput.split');
const calcStop = html.lastIndexOf('const predictionResult = {', html.indexOf('predictionHistory.unshift', calcStart));
const calcSrc = html.slice(calcStart, calcStop);

const trendFn = new Function(grabFunction('calculateTrend') + '\nreturn calculateTrend;')();
const compute = new Function(
  'oddsInput',
  'settings',
  'calculateTrend',
  'betAmount',
  'Swal',
  calcSrc + '\nreturn { averageOdds: averageOdds, probabilities: probabilities, used: oddsArray.join(", ") };'
);

const payload = JSON.parse(process.argv[2]);
const out = [];
for (const raw of payload.inputs) {
  let refused = false;
  const Swal = { fire: opts => { if (opts && opts.title === 'Données Insuffisantes') refused = true; } };
  try {
    const res = compute(raw, payload.settings, trendFn, 0, Swal);
    if (res === undefined) {
      out.push({ error: 'refus : moins de 5 côtes valides', refused: true });
      continue;
    }
    out.push({
      // `averageOdds` brut n'est PAS la cible affichée : L783 applique toFixed(2).
      target: parseFloat(res.averageOdds.toFixed(2)),
      target_raw: res.averageOdds,
      target_string: res.averageOdds.toFixed(2),
      displayed_probability: parseFloat(res.probabilities[0]),
      used: res.used,
      refused: false,
    });
  } catch (e) {
    out.push({ error: String((e && e.message) || e), refused: refused });
  }
}
process.stdout.write(JSON.stringify(out));
