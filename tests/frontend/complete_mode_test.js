// Headless test of ReaLJam complete mode: loads the real frontend script in a
// vm context with stubbed browser objects, then drives key presses, releases
// and the agent gate by hand. Time is simulated: the clock is a variable and
// setTimeout callbacks run when the test advances time.
const fs = require('fs'), vm = require('vm');
// Run: node tests/frontend/complete_mode_test.js
const path = require('path');
const src = fs.readFileSync(process.argv[2] ||
  path.join(__dirname, '../../realchords/realjam/frontend/play_from_midi_input.js'), 'utf8');

let nowMs = 0; const timers = [];
const ctx = {
  console, Math, JSON, Promise, Map, Set, Date, Number, String, Array, Object, Error, URLSearchParams,
  setTimeout: (fn, ms) => { timers.push({ at: nowMs + (ms || 0), fn }); return timers.length; },
  clearTimeout: () => {},
  window: { location: { search: '', hostname: 'x', origin: 'http://test' }, isSecureContext: true },
  document: { getElementById: () => ({}), createElement: () => ({ style: {} }), body: { appendChild() {} },
              addEventListener() {}, querySelector: () => null },
};
const frameMs = 187.5;  // 80 BPM, 16th notes
ctx.Tone = { Transport: { bpm: { value: 80 }, get position() {
  const f = nowMs / frameMs; return `${Math.floor(f / 16)}:${Math.floor((f % 16) / 4)}:${f % 4}`; },
  get seconds() { return nowMs / 1000; } },
  Time: () => ({ toSeconds: () => frameMs / 1000 }),
  context: { get currentTime() { return nowMs / 1000; } } };
// Candidate list the fake server returns: C, Am, F, G, Cmaj7 (pitch classes)
const CANDS = [
  { symbol: 'C', pitchClasses: [0, 4, 7], root: 0, onsetToken: 1000, holdToken: 10, prob: 0.4 },
  { symbol: 'Am', pitchClasses: [0, 4, 9], root: 9, onsetToken: 1001, holdToken: 11, prob: 0.2 },
  { symbol: 'F', pitchClasses: [0, 5, 9], root: 5, onsetToken: 1002, holdToken: 12, prob: 0.15 },
  { symbol: 'G', pitchClasses: [2, 7, 11], root: 7, onsetToken: 1003, holdToken: 13, prob: 0.15 },
  { symbol: 'Cmaj7', pitchClasses: [0, 4, 7, 11], root: 0, onsetToken: 1004, holdToken: 14, prob: 0.1 },
];
const requests = [];
// The list the fake server returns for a requested frame (a test can make one frame's differ)
let candsFor = () => CANDS;
const AM_FIRST = CANDS.map(c => ({ ...c, prob: c.symbol === 'Am' ? 0.9 : c.prob / 10 }))
  .sort((a, b) => b.prob - a.prob);
// Stored voicings (/voicings): for each chord, its tones in every octave
// window from C3 to C4 over the root at C2 or C3, and without that bass
const voicingsFor = symbol => {
  const chord = CANDS.find(c => c.symbol === symbol);
  if (!chord) return [];
  const out = [], seen = new Set();
  for (const bass of [null, 36 + chord.root, 48 + chord.root]) {
    for (let o = 48; o <= 60; o++) {
      const upper = chord.pitchClasses.map(pc => o + ((pc - o) % 12 + 12) % 12);
      const pitches = [...new Set([...(bass === null ? [] : [bass]), ...upper])].sort((a, b) => a - b);
      if (!seen.has(pitches.join())) { seen.add(pitches.join()); out.push({ pitches, count: 1000 - out.length }); }
    }
  }
  // and bass-fifth-octave shapes (A2 E3 A3), as stored voicings often are
  for (const bass of [36 + chord.root, 48 + chord.root]) {
    out.push({ pitches: [bass, bass + 7, bass + 12], count: 500 });
  }
  return out;
};
// The model's chord vocabulary, for input mode's exact lookup (/chord_vocab)
const VOCAB = [
  { symbol: 'C', pitchClasses: [0, 4, 7], root: 0 }, { symbol: 'C/E', pitchClasses: [0, 4, 7], root: 0 },
  { symbol: 'C5', pitchClasses: [0, 7], root: 0 }, { symbol: 'Cmaj7', pitchClasses: [0, 4, 7, 11], root: 0 },
  { symbol: 'G', pitchClasses: [2, 7, 11], root: 7 },
  { symbol: 'Am', pitchClasses: [0, 4, 9], root: 9 }, { symbol: 'F', pitchClasses: [0, 5, 9], root: 5 },
];
// The voicing fit the fake server answers (/voicing_fit): symbol -> fit, a
// test sets it; null = every chord 1 (neutral); 'fail' = a server error
let fitTable = null;
let tokenMapFails = false;
const fitRequests = [];
ctx.fetch = async (url, opts) => {
  if (url.includes('/chord_vocab')) return { ok: true, json: async () => VOCAB };
  // Model switch (2026-11-19): ids of the old model's vocabulary in the new one's
  if (url.includes('/chord_token_map')) {
    if (tokenMapFails) return { ok: false, status: 500, json: async () => ({}) };
    const map = Array.from({ length: 3000 }, (_, i) => i);
    Object.assign(map, { 1000: 2000, 1001: 2001, 10: 20, 11: 21 });
    return { ok: true, json: async () => map };
  }
  if (url.includes('/voicing_fit')) {
    fitRequests.push(JSON.parse(opts.body));
    if (fitTable === 'fail') return { ok: false, status: 500, json: async () => ({}) };
    return { ok: true, json: async () => ({ fit: VOCAB.map(c => (fitTable ? fitTable[c.symbol] || 0 : 1)) }) };
  }
  if (url.includes('/voicings?chord=')) {
    const symbol = decodeURIComponent(url.split('chord=')[1]);
    return { ok: true, json: async () => voicingsFor(symbol) };
  }
  // music21 reading (server): A2 G3 C#4 -> A7 (incomplete); anything else: none
  if (url.includes('/recognize_chord')) {
    const { pitches } = JSON.parse(opts.body);
    const a7 = JSON.stringify(pitches.sort((a, b) => a - b)) === '[45,55,61]';
    return { ok: true, json: async () => a7
      ? { symbol: 'A7', root: 9, commonName: 'incomplete dominant-seventh chord', how: 'incomplete',
          omitted: ['5'], display: 'A7 (omit 5)', onsetToken: 4321, holdToken: 321 } : null };
  }
  const body = JSON.parse(opts.body);
  requests.push(body); return { ok: true, json: async () => candsFor(body.frame) };
};
vm.createContext(ctx);
vm.runInContext(src, ctx);

// Wire the globals the functions use
const sounded = [];
const colors = new Map();  // pitch -> colour shown on the keys
const blocks = [];  // waterfall blocks drawn (scheduleNoteOn / scheduleNoteOff)
vm.runInContext(`
  chordTimingSelect = { value: 'complete' };
  chordTimingInfo = { textContent: '' };
  agentGateCheck = { checked: true };
  temperatureInput = { valueAsNumber: 0 };
  voicingTemperatureInput = { valueAsNumber: 0 };  // the most common fitting voicing
  modelSelect = { value: 'GAPT' };
  midiChordOut = null;
  chordSynth = { triggerAttack(n) { __sounded.push(['on', n]); }, triggerRelease(n) { __sounded.push(['off', n]); } };
  visual = { noteOn(p, c) { __colors.set(p, c || 'orange'); }, noteOff(p) { __colors.delete(p); },
            recolorNote(p, c) { if (__colors.has(p)) __colors.set(p, c); },
            scheduleNoteOn(p, frame, color, alpha, symbol, live, labelColor) { __blocks.push({ p, on: true, color, symbol, live, labelColor }); },
            scheduleNoteOff(p) { __blocks.push({ p, on: false }); } };
  curSession = { startFrame: 0, noteHistory: [], chordHistory: [], chordTokens: [], heldKeys: new Set(),
    onsetGroup: null, chordKeys: new Set(), melodyKeys: new Set(), completion: null, heldChord: null,
    completionHistory: [], gestureHistory: [], gateHistory: [], candidates: null, referenceOnly: false,
    playedNotes: [], playedChords: [] };
`, Object.assign(ctx, { __sounded: sounded, __colors: colors, __blocks: blocks }));

async function advance(ms) {
  const end = nowMs + ms;
  for (;;) {
    timers.sort((a, b) => a.at - b.at);
    const t = timers[0];
    if (!t || t.at > end) break;
    timers.shift(); nowMs = t.at; await t.fn();
    for (let i = 0; i < 5; i++) await null;
  }
  nowMs = end;
  for (let i = 0; i < 20; i++) await null;
}
const run = code => vm.runInContext(code, ctx);
// Like playNote / releaseNote in complete mode: light the key, then hand it on
const down = p => run(`visual.noteOn(${p}); onCompletionNoteDown(${p})`);
const up = p => run(`visual.noteOff(${p}); onCompletionNoteUp(${p})`);
const S = () => run('curSession');
let fails = 0;
const check = (name, cond) => { console.log((cond ? 'PASS ' : 'FAIL ') + name); if (!cond) fails++; };

(async () => {
  run('candidateLoop()');  // as startSession does in complete mode: refresh every frame
  await advance(10);
  // 1. Spread chord C2 + E5 (whole keyboard), agent out: recognised, silent
  down(36); await advance(10); down(76);
  await advance(100);
  check('spread chord read as chord', S().gestureHistory.at(-1).kind === 'chord');
  check('recognised as C (greedy) with agent out', S().heldChord && S().heldChord.symbol === 'C');
  check('agent silent while out', !S().completion && sounded.length === 0);
  check('onset token written', S().chordTokens.includes(1000));
  check('chord keys shown blue; the top E5 (above middle C) is melody too, orange',
    colors.get(36) === 'blue' && colors.get(76) === 'orange' && S().melodyKeys.has(76));
  // 2. Melody over the chord: single notes are melody, go to noteHistory, chord unchanged
  const reqBefore = requests.length;
  down(79); await advance(100);
  check('melody key shown orange', colors.get(79) === 'orange');
  up(79); await advance(200);
  check('single note read as melody', S().gestureHistory.at(-1).kind === 'melody');
  check('melody in noteHistory (on+off)', S().noteHistory.filter(n => n.pitch === 79).length === 2);
  check('chord unchanged by melody', S().heldChord && S().heldChord.symbol === 'C');
  check('candidates refreshed every frame, with the melody', requests.length > reqBefore &&
        requests.at(-1).notes.some(n => n.pitch === 79));
  // 3. Gate opens while chord held: nothing yet; the next chord gesture is completed
  run('setAgentGate(true)'); await advance(10);
  check('gate open over a held chord: not completed (waits for the next chord)', !S().completion);
  up(36); up(76); await advance(10);
  down(36); await advance(10); down(76); await advance(100);
  check('the next chord gesture with the gate open is completed', !!S().completion && S().completion.pitches.length > 0);
  console.log('    agent added', S().completion.pitches, 'to held', [...S().chordKeys]);
  check('agent notes shown light blue', S().completion.pitches.every(p => colors.get(p) === 'lightBlue'));
  check('completion avoids held keys', S().completion.pitches.every(p => ![36, 76, 79, 84].includes(p)));
  check('gate change logged', S().gateHistory.length === 1 && S().gateHistory[0].agentIn === true);
  // 4. Gate closes: agent notes stop, chord stays
  run('setAgentGate(false)'); await advance(10);
  check('agent silent after gate close', !S().completion);
  check('chord kept in history after gate close', !!S().heldChord);
  // 5. Melody key held while chord keys released: chord ends (melody doesn't sustain it)
  down(84); await advance(100);          // melody note held
  up(36); up(76); await advance(10);
  check('chord ends when its keys are released', !S().heldChord && S().chordKeys.size === 0);
  check('melody key still tracked', S().melodyKeys.has(84));
  up(84); await advance(10);
  // 6. Agent in, new chord G (G3 B3 D4): completed and sounding; legato change to C replaces it
  run('setAgentGate(true)');
  down(55); down(59); down(62); await advance(100);
  check('G gesture completed with agent in', S().heldChord.symbol === 'G' && !!S().completion);
  down(48); down(52); await advance(100);  // C3 E3 while G still held
  check('legato change replaces chord', S().heldChord.symbol === 'C' || S().heldChord.symbol === 'Am' || S().heldChord.symbol === 'Cmaj7');
  check('only one completion sounding', !!S().completion);
  up(55); up(59); up(62); await advance(10);
  check('releasing old chord keys keeps new chord', !!S().heldChord && S().chordKeys.size === 2);
  up(48); up(52); await advance(10);
  check('new chord ends on its release', !S().heldChord);
  // A chord reaching middle C: E3 G3 C4 -> chord C and melody C4
  run('setAgentGate(false); agentGateCheck.checked = true');
  const histLen = S().noteHistory.length;
  down(52); down(55); down(60); await advance(100);
  check('E3 G3 C4: read as a chord', S().gestureHistory.at(-1).kind === 'chord' && S().gestureHistory.at(-1).melody === 60);
  check('E3 G3 C4: the chord is C, from all three notes', S().heldChord && S().heldChord.symbol === 'C' && S().chordKeys.size === 3);
  check('E3 G3 C4: C4 is a melody note too (in noteHistory, held)',
    S().noteHistory.slice(histLen).some(n => n.on && n.pitch === 60) && S().melodyKeys.has(60));
  check('E3 G3 C4: C4 shown orange (melody), E3 and G3 blue', colors.get(60) === 'orange' && colors.get(52) === 'blue');
  const label = blocks.filter(b => b.on && b.symbol).at(-1);
  check('E3 G3 C4 is exactly C: recognised, white label, info says so',
    S().heldChord.match === 'exact' && label.labelColor === 'white' &&
    run('chordTimingInfo.textContent').includes('Recognised: C (exactly your notes)'));
  check('waterfall: chord blocks with the label C at the lowest key (E3), live',
    label && label.symbol === 'C' && label.p === 52 && label.live &&
    [55, 60].every(p => blocks.some(b => b.on && b.p === p && b.live)));
  up(60); await advance(10);
  check('releasing C4 ends its melody note, the chord goes on', S().noteHistory.at(-1).pitch === 60 &&
    !S().noteHistory.at(-1).on && S().heldChord && S().chordKeys.size === 2);
  up(52); up(55); await advance(10);
  check('releasing the rest ends the chord', !S().heldChord);
  check('waterfall: the chord blocks end with the chord',
    [52, 55, 60].every(p => blocks.some(b => !b.on && b.p === p)));
  // C4 + B4: no chord is exactly C and B; the model's Cmaj7 contains both -> inferred
  down(60); down(71); await advance(100);
  const inferredLabel = blocks.filter(b => b.on && b.symbol).at(-1);
  check('C4 B4 -> Cmaj7, inferred: yellow label, info "Inferred: Cmaj7 (from C B)"',
    S().heldChord.symbol === 'Cmaj7' && S().heldChord.match === 'inferred' &&
    inferredLabel.labelColor !== 'white' &&
    run('chordTimingInfo.textContent').includes('Inferred: Cmaj7 (from C B)'));
  up(60); up(71); await advance(10);
  // C# + D: no candidate has C#; the fallback (most overlap) leaves it out -> said so
  down(61); down(62); await advance(100);
  check('C# D -> a chord without C#: inferred, info names the note left out',
    S().heldChord.match === 'partial' &&
    /Inferred: \S+ \(from C# D; C# not in the chord\)/.test(run('chordTimingInfo.textContent')));
  up(61); up(62); await advance(10);

  down(48); down(52); down(55); await advance(100);
  check('C3 E3 G3 (top below middle C): chord only, no melody', S().gestureHistory.at(-1).melody === null &&
    !S().melodyKeys.has(55));
  up(48); up(52); up(55); await advance(10);

  // Session log: the completed chords' notes and symbols were logged as they sounded
  const played = S().playedNotes, chords = S().playedChords;
  check('session log: chord notes logged on and off',
    played.some(n => n.part === 'chord' && n.on) && played.some(n => n.part === 'chord' && !n.on));
  check('session log: chords logged with symbol and the sounding notes (held keys + agent)',
    chords.length > 0 && chords.every(c => c.symbol && c.pitches.length) &&
    chords.some(c => c.symbol === 'G' && [55, 59, 62].every(p => c.pitches.includes(p))));
  check('session log: frames match the time (80 BPM)',
    played.every(n => Math.abs(n.frame - n.t / (frameMs / 1000)) < 0.01));

  // 7. Pedal hysteresis
  run('setAgentGate(false)');
  run('onAgentPedal(30)'); const a = run('agentGateHeld');
  run('onAgentPedal(45)'); const b = run('agentGateHeld');
  run('onAgentPedal(25)'); const c = run('agentGateHeld');
  run('onAgentPedal(15)'); const d = run('agentGateHeld');
  check('pedal hysteresis (30 off, 45 on, 25 still on, 15 off)', !a && b && c && !d);
  // 8. Gate switched off (checkbox unchecked): agent always in
  run('agentGateCheck.checked = false');
  down(60); down(64); await advance(100);
  check('checkbox off: agent completes without gate', !!S().completion);
  up(60); up(64); await advance(10);
  // 9. Tap released inside the window: melody with on+off
  run('agentGateCheck.checked = true');
  down(72); await advance(20); up(72); await advance(100);
  check('quick tap read as melody with on+off', S().noteHistory.filter(n => n.pitch === 72).length === 2);
  // Input mode: the agent never plays; the exact chord shown next to the model's reading
  run(`chordTimingSelect.value = 'input'; agentGateCheck.checked = false; leftPedalDown = false;`);
  await run('loadChordVocab()'); await advance(10);
  const soundedBefore = sounded.length;
  down(60); down(67); await advance(100);  // C4 G4
  const inputLabel = blocks.filter(b => b.on && b.symbol).at(-1);
  check('input mode: C4 G4 -> exact chord C5 shown (white), the model reads C',
    S().heldChord.symbol === 'C' && S().heldChord.exact[0] === 'C5' && inputLabel.symbol === 'C5' &&
    inputLabel.labelColor === 'white' &&
    run('chordTimingInfo.textContent').includes('Exact name: C5') &&
    run('chordTimingInfo.textContent').includes('Model reads: C'));
  run('onAgentPedal(100)');
  check('input mode: the agent never plays, even with the gate off and the pedal down',
    !S().completion && sounded.length === soundedBefore);
  run('onAgentPedal(0)');
  up(60); up(67); await advance(10);
  down(52); down(55); down(60); await advance(100);  // E3 G3 C4
  check('input mode: E3 G3 C4 -> C/E first (bass = lowest note), C4 melody',
    S().heldChord.exact[0] === 'C/E' && S().heldChord.exact.includes('C') && S().melodyKeys.has(60));
  up(52); up(55); up(60); await advance(10);
  // Single notes while a chord is held: below the chord body -> chord change;
  // above it: below middle C joins too (a bass-register note), from C4 melody
  down(48); down(52); down(55); await advance(100);   // C3 E3 G3 (body top G3)
  down(62); await advance(100);                        // D4 above the body -> melody
  check('input mode: single note at/above C4 over the held chord stays melody',
    S().gestureHistory.at(-1).kind === 'melody' && S().melodyKeys.has(62) && S().chordKeys.size === 3);
  up(62); await advance(10);
  down(57); await advance(100);                        // A3: above the body, below C4 -> joins
  check('input mode: single note above the chord body but below C4 joins the chord',
    S().gestureHistory.at(-1).kind === 'chord change' && S().chordKeys.has(57) && S().chordKeys.size === 4);
  up(57); await advance(10);
  up(55); down(53); await advance(100);               // G3 -> F3: one note changed, below the top
  check('input mode: one note changed below the chord top -> chord change (C3 E3 F3)',
    S().gestureHistory.at(-1).kind === 'chord change' && S().chordKeys.size === 3 &&
    [48, 52, 53].every(p => S().chordKeys.has(p)) && colors.get(53) === 'blue');
  up(48); up(52); up(53); await advance(10);
  down(60); down(67); await advance(100);              // C4 G4: G4 is the melody top, body = C4
  down(52); await advance(100);                        // E3 below the body -> chord change
  check('input mode: a melody top note is not part of the chord body (E3 under C4 [G4]) -> chord change',
    S().gestureHistory.at(-1).kind === 'chord change' && S().heldChord.exact[0] === 'C/E' &&
    S().melodyKeys.has(67));
  up(52); up(60); up(67); await advance(10);
  // Bass first: D2 alone (a chord of its own), then F3 A3 C4 E4 while D2 is held -> D2 joins
  down(38); await advance(230);                        // D2, 230 ms early
  check('input mode: a lone bass note below C4 is a chord gesture (bass), not melody',
    S().gestureHistory.at(-1).kind === 'bass' && S().chordKeys.has(38) && !S().melodyKeys.has(38) &&
    S().heldChord.loneBass && colors.get(38) === 'blue');
  down(53); down(57); down(60); down(64); await advance(100);
  check('input mode: bass struck first joins the chord above it (D2 + F3 A3 C4 E4)',
    S().chordKeys.has(38) && !S().melodyKeys.has(38) && S().chordKeys.size === 5 &&
    JSON.stringify(S().gestureHistory.at(-1).absorbed) === '[38]' && !S().heldChord.loneBass);
  up(38); up(53); up(57); up(60); up(64); await advance(10);
  // music21: A2 G3 C#4 -> label A7 (white), info shows music21's type
  down(45); down(55); down(61); await advance(100);
  const m21Label = blocks.filter(b => b.on && b.symbol).at(-1);
  check('input mode: A2 G3 C#4 -> label "A7 (omit 5)" (white), info line says so',
    m21Label.symbol === 'A7 (omit 5)' && m21Label.labelColor === 'white' &&
    run('chordTimingInfo.textContent').includes('A7 (omit 5) (music21, in the history)'));
  const onsetAt = S().heldChord.startFrame;
  await advance(400);
  run('fillChordTokens()');
  check('input mode: music21\'s A7 is what the history gets (onset, then holds); the model\'s reading kept apart',
    S().chordTokens[onsetAt] === 4321 && S().chordTokens[onsetAt + 1] === 321 &&
    S().heldChord.symbol === 'A7' && S().heldChord.modelSymbol !== undefined);
  up(45); up(55); up(61); await advance(10);
  run('agentGateCheck.checked = true');

  // Complete mode now reads like input mode: a single note below the chord changes it
  run(`chordTimingSelect.value = 'complete'; agentGateCheck.checked = true; leftPedalDown = false; agentGateHeld = false;`);
  down(48); down(52); down(55); await advance(100);
  down(53); await advance(100);
  check('complete mode: a single note below the chord changes it',
    S().gestureHistory.at(-1).kind === 'chord change');
  up(48); up(52); up(55); up(53); await advance(10);
  // Pedal up: music21's reading goes into the history (A2 G3 C#4 -> A7, token 4321)
  down(45); down(55); down(61); await advance(100);
  const startA7 = S().heldChord.startFrame;
  check('complete mode, pedal up: music21\'s chord in the history, agent silent',
    S().chordTokens[startA7] === 4321 && !S().completion);
  // Pedal down while holding it: nothing yet, the chord held keeps music21's reading
  run('onAgentPedal(100)'); await advance(100);
  check('complete mode, pedal down on a held chord: not completed, music21\'s chord stays in the history',
    !S().completion && S().chordTokens[startA7] === 4321);
  up(45); up(55); up(61); await advance(10);
  down(45); down(55); down(61); await advance(100);
  check('complete mode, pedal held, the next chord gesture: completed, the history gets the model\'s chord',
    !!S().completion && S().chordTokens[S().heldChord.startFrame] >= 1000 &&
    S().chordTokens[S().heldChord.startFrame] <= 1004);
  run('onAgentPedal(0)'); await advance(10);
  check('complete mode, pedal up again: the agent\'s notes stop', !S().completion);
  up(45); up(55); up(61); await advance(10);
  // Pedal down with empty hands: nothing plays; the next chord is completed
  run('onAgentPedal(100)'); await advance(100);
  check('complete mode, pedal down with no chord notes: the agent plays nothing yet',
    !S().heldChord && !S().completion);
  down(48); down(52); await advance(100);
  check('complete mode, pedal held, then a chord: completed', !!S().completion && !!S().heldChord);
  up(48); up(52); await advance(10);
  run('onAgentPedal(0)'); await advance(10);
  check('complete mode, pedal up: nothing sounds', !S().completion);
  // Space (2026-11-18): plays at once, only with the pedal down; space up ends it
  run('spaceDown()'); await advance(100);
  check('space, pedal up: nothing', !S().completion && !S().heldChord);
  run('spaceUp()');
  run('onAgentPedal(100)'); await advance(10);
  run('spaceDown()'); await advance(100);
  check('space, pedal down, empty hands: the agent plays a chord of its own',
    S().heldChord && S().heldChord.free && S().completion && S().completion.pitches.length >= 3 &&
    S().chordTokens[S().heldChord.startFrame] >= 1000);
  run('spaceUp()'); await advance(10);
  check('space up: its own chord ends', !S().heldChord && !S().completion);
  run('onAgentPedal(0)'); await advance(10);
  down(48); down(52); await advance(100);                   // C3 E3, pedal up: silent
  run('onAgentPedal(100)'); await advance(10);
  check('pedal down over a held chord: still silent', !S().completion);
  run('spaceDown()'); await advance(100);
  check('space over a held chord: completed at once', !!S().completion && S().completion.pitches.length > 0);
  run('spaceUp()'); await advance(10);
  check('space up: the completion stops, the chord stays', !S().completion && !!S().heldChord);
  run('spaceDown()'); await advance(100);
  run('onAgentPedal(0)'); await advance(10);
  check('pedal up while space holds a chord: the agent stops', !S().completion);
  run('spaceUp()');
  up(48); up(52); await advance(10);

  // All candidates from the vocabulary + probabilities (format "probs")
  const ranked = run(`candidatesFromProbs([{symbol: 'C', onsetToken: 1}, {symbol: 'G', onsetToken: 2},
    {symbol: 'Am', onsetToken: 3}], [0.2, 0.5, 0.3])`);
  check('candidates from probabilities: every chord, most probable first, fields kept',
    ranked.map(c => c.symbol).join() === 'G,Am,C' && ranked[0].prob === 0.5 && ranked[0].onsetToken === 2);

  // Triggered mode: the left pedal plays the scheduled chord like space
  run(`chordTimingSelect.value = 'triggered'; curSession.triggerHistory = []; curSession.triggeredChord = null;
       curSession.chordHistory = [{scheduleFrame: 0, symbol: 'F', pitches: [53, 57, 60], eventIDs: []}];
       leftPedalDown = false;`);
  run('onAgentPedal(30)');
  check('triggered: pedal below the threshold plays nothing', !S().triggeredChord);
  run('onAgentPedal(60)');
  check('triggered: pedal down plays the scheduled chord', JSON.stringify(S().triggeredChord) === JSON.stringify([53, 57, 60]));
  run('onAgentPedal(30)');
  check('triggered: pedal hovering above the off threshold keeps it', !!S().triggeredChord);
  run('onAgentPedal(10)');
  check('triggered: pedal up stops it', !S().triggeredChord &&
    S().triggerHistory.some(h => h.release));

  // Triggered variant: the history's past is rewritten to what sounded
  nowMs = 0;
  run(`chordTimingSelect.value = 'triggered-history'; curSession.startFrame = 0;
       curSession.triggerHistory = []; curSession.triggeredChord = null;
       curSession.chordHistory = [{scheduleFrame: 0, symbol: 'G', pitches: [55, 59, 62], eventIDs: []}];
       curSession.chordHoldRange = [10, 20]; curSession.chordOnsetOffset = 1000;
       curSession.chordTokens = Array.from({length: 32}, (_, f) => f === 0 ? 1013 : 13);`);
  nowMs = 4 * frameMs; run('triggerScheduledChord()');   // press at frame 4 (plan: G hold there)
  nowMs = 8 * frameMs; run('releaseTriggeredChord()');   // release at frame 8
  nowMs = 12 * frameMs; run('rewriteTriggeredHistory()');
  const hist = run('curSession.chordTokens');
  check('history variant: silence before the press (frames 0-3)', [0, 1, 2, 3].every(f => hist[f] === -1));
  check('history variant: the pressed chord, onset at the press frame then holds (4-7)',
    hist[4] === 1013 && [5, 6, 7].every(f => hist[f] === 13));
  check('history variant: silence after the release (8-11)', [8, 9, 10, 11].every(f => hist[f] === -1));
  check('history variant: the plan from now on untouched (12+)', hist[12] === 13 && hist[31] === 13);
  run(`chordTimingSelect.value = 'triggered'`);
  check('plain triggered mode: rewriting is off', !run('isTriggeredHistoryMode()') && run('isTriggeredChordMode()'));

  // Bass vs melody split (2026-11-18)
  run(`chordTimingSelect.value = 'input'`);
  [...S().heldKeys].forEach(p => up(p)); await advance(10);
  down(72); await advance(100);                         // C5 melody, held
  down(69); await advance(100);                         // A4: 3 below the held melody -> legato step
  check('split: 3 semitones below a held melody note is melody (legato)', S().gestureHistory.at(-1).kind === 'melody');
  up(72); up(69); await advance(10);
  down(81); await advance(100);                         // A5 melody, held
  down(72); await advance(100);                         // C5: well below the held melody, above C4
  check('split: a fourth or more below a held melody note is a bass note, even above C4',
    S().gestureHistory.at(-1).kind === 'bass' && S().chordKeys.has(72));
  up(72); up(81); await advance(10);
  down(64); await advance(100);                         // E4 alone, nothing held
  check('split: nothing held, at or above C4 -> melody', S().gestureHistory.at(-1).kind === 'melody');
  up(64); await advance(10);
  down(48); await advance(100); down(50); await advance(100);  // C3 then D3 (legato bass line)
  check('split: a lone bass note after another replaces it (moving bass line)',
    S().gestureHistory.at(-1).kind === 'bass' && S().chordKeys.size === 1 && S().chordKeys.has(50));
  up(48); up(50); await advance(10);

  // Completion voicing (2026-11-18): a stored voicing containing the keys
  run(`chordTimingSelect.value = 'complete'; agentGateCheck.checked = false;`);
  [...S().heldKeys].forEach(p => up(p)); await advance(10);
  down(64); await advance(100); up(64); await advance(10);        // melody E4
  down(57); down(72); await advance(100);                          // A3 + C5 (C5 melody top)
  const ceilingNow = run('recentMelodyLow()') - 3;
  check(`complete mode: the completion stays 3 below the recent melody low (<= ${ceilingNow})`,
    !!S().completion && S().completion.pitches.length > 0 && Math.max(...S().completion.pitches) <= ceilingNow);
  check('completion: a stored voicing that contains the keys at their pitches (A3), plus nothing else held',
    S().completion.voicing.includes(57) && !S().completion.pitches.includes(57) &&
    S().completion.pitches.every(p => S().completion.voicing.includes(p)) &&
    S().completion.voicing.filter(p => !S().completion.pitches.includes(p)).join() === '57');
  up(57); up(72); await advance(10);
  down(37); down(42); await advance(100);                          // C#2 F#2: no stored voicing has them
  check('completion: no stored voicing contains the keys -> the agent adds nothing',
    !!S().completion && S().completion.pitches.length === 0 && !S().completion.voicing &&
    /no stored voicing contains your keys/.test(run('chordTimingInfo.textContent')));
  up(37); up(42); await advance(10);
  // A voice moving above the bass is no new prompt (2026-11-18)
  run('curSession.noteHistory = []; curSession.melodyKeys.clear();');
  nowMs += 3000; await advance(10);                               // old melody out of the window
  down(48); down(55); await advance(100);                          // C3 G3 -> C, completed
  const first = S().completion.pitches.slice();
  const strikes = () => sounded.filter(e => e[0] === 'on').length;
  const struck = strikes();
  check('voice move setup: C3 G3 completed', first.length > 0);
  up(55); down(57); await advance(100);                            // G3 -> A3, above the bass C3
  const moved = S().heldChord;
  check('voice move: read anew (Am or F, in the history)', ['Am', 'F'].includes(moved.symbol) &&
    S().chordTokens[moved.startFrame] === moved.onsetToken && S().gestureHistory.at(-1).voiceMove);
  check('voice move: nothing struck', strikes() === struck);
  check('voice move: the agent keeps only its notes that belong to the new chord',
    S().completion.kept && S().completion.pitches.every(p => first.includes(p) && moved.pitchClasses.includes(p % 12)));
  down(41); await advance(100);                                    // F2 below: the bass moves
  check('bass move: a new prompt, the agent strikes a new completion',
    S().heldChord.symbol === 'F' && !S().completion.kept && strikes() > struck);
  up(41); up(48); up(57); await advance(10);

  // The chord goes in at its first key's frame, picked from the model's list
  // for that frame (2026-11-18). The refresh loop is stopped and the lists
  // requested by hand, so the timing is exact.
  timers.splice(0, timers.length, ...timers.filter(t => t.fn !== ctx.candidateLoop));
  run("setAgentGate(false); curSession.noteHistory = []; curSession.melodyKeys.clear();");
  nowMs = (Math.ceil(nowMs / frameMs) + 8) * frameMs; await advance(0);
  const F = Math.round(nowMs / frameMs) + 2;
  candsFor = f => (f === F ? AM_FIRST : CANDS);
  nowMs = (F - 1) * frameMs + 20; await advance(0);
  await run(`requestCandidates(curSession, ${F})`);      // the list for F (Am first), during F-1
  nowMs = F * frameMs + 10; await advance(0);
  await run(`requestCandidates(curSession, ${F + 1})`);  // the latest list: for F+1 (C first)
  nowMs = F * frameMs + 75; await advance(0);             // 75 ms late: rounds to F
  down(48); down(52); await advance(100);                 // read 50 ms later, in frame F+1
  const late = S().completionHistory.at(-1);
  check('late gesture: the chord goes in at its first key\'s frame, not when it was read',
    late.frame === F && late.readAt === F + 1);
  check('late gesture: picked from the kept list for its own frame, no new request',
    S().heldChord.symbol === 'Am' && late.refetched === false);
  check('late gesture: onset token at its frame, its hold in the next',
    S().chordTokens[F] === 1001 && S().chordTokens[F + 1] === 11);
  up(48); up(52); await advance(10);
  const G = F + 8;
  candsFor = f => (f === G ? AM_FIRST : CANDS);
  nowMs = (G - 1) * frameMs + 10; await advance(0);
  await run(`requestCandidates(curSession, ${G})`);      // the list for G, before the melody note
  down(72); await advance(60); up(72); await advance(10);  // a melody note, still in frame G-1
  nowMs = G * frameMs + 30; await advance(0);
  const reqBeforeGesture = requests.length;
  down(48); down(52); await advance(100);
  const fresh = S().completionHistory.at(-1);
  check('melody after the kept list: the model is asked again for the gesture\'s frame, with that note',
    fresh.refetched === true && fresh.frame === G && S().heldChord.symbol === 'Am' &&
    requests.slice(reqBeforeGesture).some(r => r.frame === G && r.notes.some(n => n.pitch === 72)));
  up(48); up(52); await advance(10);
  candsFor = () => CANDS;

  // Input lag (2026-11-18): the lag is taken off a key's time before rounding
  nowMs = (Math.ceil(nowMs / frameMs) + 8) * frameMs; await advance(0);
  const H = Math.round(nowMs / frameMs);
  check('input lag: 30 ms by default (no setting on the page)', run('inputLagMs()') === 30);
  nowMs = H * frameMs + 100; await advance(0);            // 100 ms after H: past half a 16th
  down(48); down(52); await advance(100);
  check('input lag 30: a chord 100 ms after the beat counts for the beat',
    S().gestureHistory.at(-1).frame === H && S().completionHistory.at(-1).frame === H &&
    S().chordTokens[H] === S().heldChord.onsetToken);
  up(48); up(52); await advance(10);
  nowMs = (H + 2) * frameMs + 65; await advance(0);       // the "and", 65 ms late
  down(79); await advance(60); up(79); await advance(10);
  check('input lag 30: a note 65 ms after the "and" stays on the "and"',
    S().noteHistory.filter(n => n.pitch === 79).at(-2).frame === H + 2);
  run('inputLagInput = { valueAsNumber: 0 }');
  nowMs = (H + 8) * frameMs + 100; await advance(0);
  down(48); down(52); await advance(100);
  check('input lag 0: the same chord counts for the 16th after the beat (as before)',
    S().gestureHistory.at(-1).frame === H + 9 && S().completionHistory.at(-1).frame === H + 9);
  up(48); up(52); await advance(10);
  run('inputLagInput = { valueAsNumber: 500 }');
  check('input lag: capped at 150 ms', run('inputLagMs()') === 150);
  run('inputLagInput = undefined');
  // Space: the agent's own chord from the list for the frame of the press
  const Sp = Math.round(nowMs / frameMs) + 4;
  candsFor = f => (f === Sp ? AM_FIRST : CANDS);
  nowMs = (Sp - 1) * frameMs + 20; await advance(0);
  await run(`requestCandidates(curSession, ${Sp})`);     // the list for Sp (Am first)
  nowMs = Sp * frameMs + 10; await advance(0);
  await run(`requestCandidates(curSession, ${Sp + 1})`); // the latest list: for Sp+1 (C first)
  nowMs = Sp * frameMs + 40; await advance(0);
  run('setAgentGate(true)'); run('spaceDown()'); await advance(50);
  const own = S().completionHistory.at(-1);
  check('space: the agent\'s own chord comes from the list for the press\'s frame, written there',
    own.match === 'agent' && own.frame === Sp && own.symbol === 'Am' && own.refetched === false &&
    S().chordTokens[Sp] === 1001);
  run('spaceUp()'); await advance(10); run('setAgentGate(false)');
  candsFor = () => CANDS;

  // The voicing fit weights the chord choice (2026-11-18): C3 E3, agent in,
  // temperature 0 -> the largest p * fit among the chords containing C and E
  run('setAgentGate(true)');
  fitTable = { C: 0, Am: 0.3, Cmaj7: 0.05 };
  const fitBefore = fitRequests.length;
  nowMs += 2 * frameMs; await advance(0);
  down(48); down(52); await advance(100);
  const fitted = S().completionHistory.at(-1);
  const fitReq = fitRequests[fitBefore];
  check('voicing fit: asked once for the gesture, with the keys at their pitches',
    fitRequests.length === fitBefore + 1 && JSON.stringify(fitReq.keys.sort()) === '[48,52]' &&
    fitReq.held.includes(48) && fitReq.held.includes(52) && 'ceiling' in fitReq);
  check('voicing fit: C (most probable, but no playable voicing with the keys) loses to Am (0.2 x 0.3)',
    S().heldChord.symbol === 'Am' && fitted.fit === 0.3 && fitted.fitAsked === true);
  up(48); up(52); await advance(10);
  fitTable = {};  // no chord can be completed around the keys
  nowMs += 2 * frameMs; await advance(0);
  down(48); down(52); await advance(100);
  check('voicing fit: none of the chords has a playable voicing -> the plain rule (C), fit null',
    S().heldChord.symbol === 'C' && S().completionHistory.at(-1).fit === null);
  up(48); up(52); await advance(10);
  fitTable = 'fail';
  nowMs += 2 * frameMs; await advance(0);
  down(48); down(52); await advance(100);
  check('voicing fit: server error -> the plain rule (C)', S().heldChord.symbol === 'C' &&
    S().completionHistory.at(-1).fitAsked === false);
  up(48); up(52); await advance(10);
  const gateWas = run('agentGateCheck.checked');
  run('agentGateCheck.checked = true; setAgentGate(false)'); fitTable = { C: 0, Am: 0.3 };
  const fitBeforeOut = fitRequests.length;
  nowMs += 2 * frameMs; await advance(0);
  down(48); down(52); await advance(100);
  check('voicing fit: agent out -> not asked, the model\'s reading (C)',
    fitRequests.length === fitBeforeOut && S().heldChord.symbol === 'C');
  up(48); up(52); await advance(10);
  fitTable = null; run(`agentGateCheck.checked = ${gateWas}`);

  // Switching models mid-session: the history is remapped by name first
  run(`curSession.model = 'GAPT'; modelSelect.value = 'GAPT';
    curSession.chordTokens = [1000, 10, 10, -1, 3, 1001, 11];
    curSession.heldChord = { symbol: 'Am', onsetToken: 1001, holdToken: 11, startFrame: 5 };
    curSession.candidates = [{ symbol: 'C' }]; curSession.candidateLists = new Map([[9, {}]]);
    modelSelect.value = 'NEW'`);
  await run('onModelChanged()'); await advance(0);
  check('model switch: history and held chord remapped, then the session switches',
    JSON.stringify(S().chordTokens) === '[2000,20,20,-1,3,2001,21]' &&
    S().heldChord.onsetToken === 2001 && S().heldChord.holdToken === 21 &&
    S().model === 'NEW' && run('sessionModel()') === 'NEW');
  check('model switch: the old model\'s candidates dropped, the switch logged',
    S().candidates === null && S().candidateLists.size === 0 &&
    S().modelSwitches.at(-1).from === 'GAPT' && S().modelSwitches.at(-1).to === 'NEW');
  const reqNew = requests.length;
  await run('requestCandidates(curSession, 50)');
  check('model switch: requests now name the new model', requests.length > reqNew && requests.at(-1).model === 'NEW');
  tokenMapFails = true;
  run("modelSelect.value = 'GAPT'"); await run('onModelChanged()'); await advance(0);
  check('model switch: remap fails -> stays on the old model (menu reset), history untouched',
    S().model === 'NEW' && run('modelSelect.value') === 'NEW' && S().chordTokens[0] === 2000);
  tokenMapFails = false;
  run("curSession.heldChord = null; curSession.chordTokens = []");

  // Certainty: exp(entropy) of the renormalised distribution
  const sure = run(`onsetCertainty([{symbol: 'C', prob: 0.5}, {symbol: 'G', prob: 0.25}, {symbol: 'F', prob: 0.25}])`);
  check('certainty: effective number of chords (0.5/0.25/0.25 -> 2.83)', Math.abs(sure.choices - 2 * Math.SQRT2) < 1e-9);
  check('certainty: top share renormalised', sure.top === 'C' && sure.topShare === 0.5);
  const uniform = run(`onsetCertainty([{symbol: 'C', prob: 0.1}, {symbol: 'G', prob: 0.1}])`);
  check('certainty: two equal chords -> 2 choices, 50%', Math.abs(uniform.choices - 2) < 1e-9 && uniform.topShare === 0.5);
  const pick = run(`pickCandidate([{symbol: 'C', pitchClasses: [0, 4, 7], prob: 0.3}, {symbol: 'Am', pitchClasses: [9, 0, 4], prob: 0.1},
    {symbol: 'G', pitchClasses: [7, 11, 2], prob: 0.6}], [60, 64], true)`);
  check('pick: share among the chords that fit (C 0.3 of C+Am 0.4 = 75%)',
    pick.chord.symbol === 'C' && Math.abs(pick.share - 0.75) < 1e-9 && pick.choices > 1 && pick.choices < 2);
  check('format: 63% / 0.4% / ≈2 / ≈37', run(`formatShare(0.634) + ' ' + formatShare(0.004) + ' ' + formatChoices(2.04) + ' ' + formatChoices(36.6)`) === '63% 0.4% ≈2 ≈37');
  run(`chordTimingSelect.value = 'complete'; curSession.certainty = onsetCertainty(curSession.candidates); updateChordTimingInfo()`);
  check('info line shows the model\'s certainty now', /Model now: ≈[\d.]+ likely chords \(top /.test(run('chordTimingInfo.textContent')));
  console.log(fails ? `${fails} FAILED` : 'ALL PASSED');
  process.exit(fails ? 1 : 0);
})();
