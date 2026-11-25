// Headless test of ReaLJam's database voicings (auto / triggered mode and the
// song's own chords in song playback): loads the real frontend script in a
// Node vm with stubbed browser objects.
// Run: node tests/frontend/database_voicing_test.js
const fs = require('fs'), vm = require('vm'), path = require('path');
const src = fs.readFileSync(process.argv[2] ||
  path.join(__dirname, '../../realchords/realjam/frontend/play_from_midi_input.js'), 'utf8');

let nowFrame = 0;
const frameSeconds = 60 / 80 / 4;  // 80 BPM, 16th notes
const ctx = {
  console, Math, JSON, Promise, Map, Set, Date, Number, String, Array, Object, Error, URLSearchParams,
  setTimeout: () => 0, clearTimeout() {},
  window: { location: { search: '', hostname: 'x', origin: 'http://t' }, isSecureContext: true },
  document: { getElementById: () => ({}), createElement: () => ({ style: {} }), body: { appendChild() {} },
              addEventListener() {}, querySelector: () => null },
  Tone: { Transport: { bpm: { value: 80 }, get position() {
    return `${Math.floor(nowFrame / 16)}:${Math.floor((nowFrame % 16) / 4)}:${nowFrame % 4}`; } },
    Time: () => ({ toSeconds: () => frameSeconds }), context: { currentTime: 0 } },
};
// Stored voicings for "C", most common first. Two break the register rules
// and must never be drawn, however common.
const NARROW = [60, 64, 67];        // C4 E4 G4: spans a fifth (< an octave)
const MUDDY = [36, 40, 43, 48];     // C2 E2 G2 C3: a third and a minor third below C3
const C_VOICINGS = [
  { pitches: NARROW, count: 50000 },
  { pitches: MUDDY, count: 40000 },
  { pitches: [48, 55, 64], count: 10000 },     // C3 G3 E4
  { pitches: [52, 60, 67], count: 5000 },      // E3 C4 G4
  { pitches: [48, 52, 55, 60], count: 2000 },  // C3 E3 G3 C4 (exactly an octave)
  { pitches: [55, 64, 72], count: 1000 },      // G3 E4 C5
  { pitches: [36, 43, 60], count: 500 },       // C2 G2 C4: a fifth and more below C3, fine
  { pitches: [60, 67, 76], count: 100 },       // C4 G4 E5
  { pitches: [64, 72, 79], count: 10 },        // E4 C5 G5
];
const ALLOWED = C_VOICINGS.filter(v => v.pitches !== NARROW && v.pitches !== MUDDY);
ctx.fetch = async () => ({ json: async () => [{ pitches: [48, 52, 55, 36], legacy: true }, ...C_VOICINGS] });
vm.createContext(ctx);
vm.runInContext(src, ctx);
const run = code => vm.runInContext(code, ctx);
run(`
  voicingTemperatureInput = { valueAsNumber: 1 };
  midiChordOut = null;
  curSession = { startFrame: 0, noteHistory: [], chordHistory: [], voicingDecisions: new Map() };
`);
let fails = 0;
const check = (name, cond) => { console.log((cond ? 'PASS ' : 'FAIL ') + name); if (!cond) fails++; };
const key = p => JSON.stringify(p);
const block = [48, 52, 55, 36];
const voice = frame => run(`databaseVoicing(${frame}, 'C', true, ${key(block)})`);
const reset = (t, notes = []) => run(`voicingTemperatureInput.valueAsNumber = ${t};` +
  `curSession.voicingDecisions.clear(); curSession.noteHistory = ${key(notes)}`);

(async () => {
  // First call starts the fetch and falls back to the server's block chord
  check('falls back to the block chord while voicings load', key(voice(100)) === key(block));
  await new Promise(resolve => setImmediate(resolve));  // let the fetch finish
  check('voicings loaded', Array.isArray(run("voicingCandidateCache.get('C')")));

  // Register rules
  check('rule: spans at least an octave (C4 E4 G4 out, C3 E3 G3 C4 in)',
    !run(`voicingRegisterOk(${key(NARROW)})`) && run('voicingRegisterOk([48, 52, 55, 60])'));
  check('rule: below C3 at least a fifth apart (C2 E2 G2 C3 out, C2 G2 C4 in)',
    !run(`voicingRegisterOk(${key(MUDDY)})`) && run('voicingRegisterOk([36, 43, 60])'));
  check('rule: nothing below C2 (A1 E2 ... out)', !run('voicingRegisterOk([33, 40, 52, 57])') && run('voicingRegisterOk([36, 43, 52])'));
  check('complete mode: keys below C3 by pitch class, the voicing played higher (2026-11-23)',
    run('JSON.stringify(completionAdded([48, 55, 64], [36, 40], new Set([36, 40]), null))') === '[48,55,64]'
    && run('completionAdded([52, 60, 67], [36, 40], new Set([36, 40]), null)') === null
    && run('JSON.stringify(completionAdded([36, 43, 52, 60], [36, 40], new Set([36, 40]), null))') === '[52,60]'
    && run('completionAdded([48, 55, 64], [36, 40], new Set([36, 40]), 60)') === null);
  check('rule: the performer\'s own keys are exempt (2026-11-23)',
    run(`voicingRegisterOk(${key(MUDDY)}, ${key(MUDDY)})`) && !run(`voicingRegisterOk(${key(MUDDY)}, [36, 40, 43])`)
    && run('voicingRegisterOk([60, 64, 67], [60, 64, 67])') && !run('voicingRegisterOk([60, 64, 67], [60, 64])')
    && run('voicingRegisterOk([33, 45, 52, 60], [33])'));
  check('block chord: a tone within a fifth above a bass below C3 goes up an octave (B: B2 D#3 F#3 B3 -> B2 F#3 B3 D#4)',
    JSON.stringify(run('blockChordInRegister([59, 51, 54, 47])').sort((a, b) => a - b)) === '[47,54,59,63]');
  check('block chord: C (C2 + C3 E3 G3) unchanged', JSON.stringify(run('blockChordInRegister([48, 52, 55, 36])')) === '[48,52,55,36]');
  reset(1);
  const drawn = new Set();
  for (let f = 0; f < 3000; f++) drawn.add(key(voice(10000 + f)));
  check('rules: the narrow and the muddy voicing are never drawn, though the most common',
    !drawn.has(key(NARROW)) && !drawn.has(key(MUDDY)));

  // Temperature 0: the most common voicing that meets the rules
  reset(0);
  check('T=0, no melody: the most common allowed voicing (C3 G3 E4)', key(voice(100)) === key([48, 55, 64]));

  // Melody: lowest note of the last two beats is E4 (64) -> top <= 61 (3 semitones below)
  reset(0, [{ on: true, pitch: 72, frame: 0 }, { on: false, pitch: 72, frame: 2 }, { on: true, pitch: 64, frame: 3 }]);
  nowFrame = 4;
  const v = voice(100);
  check('top at least 3 semitones below the recent melody low (<= 61)', Math.max(...v) <= 61);
  check('T=0 with ceiling: most common allowed that fits (C3 E3 G3 C4)', key(v) === key([48, 52, 55, 60]));

  // Sticky: melody rises -> keep; melody drops below -> redraw
  reset(1, [{ on: true, pitch: 79, frame: 3 }]);
  const high = voice(100);
  run('curSession.noteHistory.push({on: false, pitch: 79, frame: 5}, {on: true, pitch: 84, frame: 5})');
  check('sticky: melody rises, voicing kept', key(voice(100)) === key(high));
  run('curSession.noteHistory.push({on: false, pitch: 84, frame: 6}, {on: true, pitch: 65, frame: 6})');
  check('melody drops below: redrawn under it (top <= 62)', Math.max(...voice(100)) <= 62);

  // Fixed random number per chord: repeated polls give the same choice
  reset(1);
  const draws = new Set();
  for (let i = 0; i < 20; i++) draws.add(key(voice(200)));
  check('same chord, repeated polls: one voicing (no flicker)', draws.size === 1);

  // Temperature shapes the distribution over many chords (allowed voicings only)
  const share = (t, n = 4000) => {
    reset(t);
    let top = 0;
    for (let f = 0; f < n; f++) if (key(voice(1000 + f)) === key([48, 55, 64])) top++;
    return top / n;
  };
  // No melody here: the default ceiling A4 (2026-11-18) leaves the allowed voicings up to A4
  const UNDER_A4 = ALLOWED.filter(v => Math.max(...v.pitches) <= 69);
  const expected = 10000 / UNDER_A4.reduce((a, v) => a + v.count, 0);
  const s05 = share(0.5), s1 = share(1), s3 = share(3);
  console.log(`    share of the most common allowed voicing: T=0.5 ${s05.toFixed(2)}, T=1 ${s1.toFixed(2)}, T=3 ${s3.toFixed(2)} (T=1 expected ${expected.toFixed(2)})`);
  check('T=1 draws in proportion to count', Math.abs(s1 - expected) < 0.03);
  reset(3);
  let aboveA4 = 0;
  for (let f = 0; f < 2000; f++) if (Math.max(...voice(5000 + f)) > 69) aboveA4++;
  check('no recent melody: nothing above A4 (the default ceiling), even at T=3', aboveA4 === 0);
  check('lower T favours the common voicing, higher T spreads', s05 > s1 && s1 > s3);

  // Walking bass line: every note above G3
  run('curSession.songKind = "bass"'); reset(0);
  check('bass line: every note above G3', Math.min(...voice(300)) > 55);
  run('delete curSession.songKind');

  // Nothing fits (melody Bb3 -> top <= 55; every allowed top is C4 or higher): the block chord
  reset(1, [{ on: true, pitch: 58, frame: 499 }]);
  nowFrame = 500;
  check('nothing fits: the block chord (no octave-shifted copies)', key(voice(900)) === key(block));
  reset(1);

  // Hold of the same chord keeps its voicing
  run(`curSession.chordHistory = [{scheduleFrame: 399, symbol: 'C', pitches: [60, 64, 67], eventIDs: []}]`);
  check('hold keeps the held voicing', key(run(`databaseVoicing(400, 'C', false, ${key(block)})`)) === key([60, 64, 67]));
  run('curSession.chordHistory = []');

  // Song playback: the song's own chords, voiced by the same rule (melody known in advance)
  run('voicingTemperatureInput.valueAsNumber = 0');
  const songChords = [
    { symbol: 'C', onset: 0, offset: 4, pitches: block },   // melody E4 over it -> top <= 61
    { symbol: 'C', onset: 4, offset: 8, pitches: block },   // melody C6 -> no tight ceiling
    { symbol: 'C', onset: 8, offset: 12, pitches: block },  // melody Bb3 -> nothing fits -> block
  ];
  const songMelody = [{ pitch: 64, onset: 1, offset: 1.5 }, { pitch: 84, onset: 6.5, offset: 8 },
    { pitch: 58, onset: 9, offset: 11 }];
  const voicedSong = await vm.runInContext(
    `voiceReferenceChords(${JSON.stringify(songChords)}, ${JSON.stringify(songMelody)}, false)`, ctx);
  check('song chords: top 3 below the melody over the chord (C3 E3 G3 C4)', key(voicedSong[0].pitches) === key([48, 52, 55, 60]));
  check('song chords: high melody -> the most common allowed voicing (T=0)', key(voicedSong[1].pitches) === key([48, 55, 64]));
  check('song chords: nothing fits -> the block chord', key(voicedSong[2].pitches) === key(block));
  const voicedBass = await vm.runInContext(`voiceReferenceChords(${JSON.stringify(songChords)}, [], true)`, ctx);
  check('song chords under a bass line: every note above G3', voicedBass.every(c => Math.min(...c.pitches) > 55));

  // With a MIDI output the voicing is final MIDI_DECIDE_AHEAD before the chord
  run('midiChordOut = {}'); reset(0, [{ on: true, pitch: 84, frame: 3 }]);
  nowFrame = 498;  // 2 frames = 0.375 s before frame 500 < 0.55 s
  const fixed = voice(500);
  run('curSession.noteHistory.push({on: false, pitch: 84, frame: 4}, {on: true, pitch: 62, frame: 498})');
  check('MIDI output: no redraw within MIDI_DECIDE_AHEAD', key(voice(500)) === key(fixed));
  console.log(fails ? `${fails} FAILED` : 'ALL PASSED');
  process.exit(fails ? 1 : 0);
})();
