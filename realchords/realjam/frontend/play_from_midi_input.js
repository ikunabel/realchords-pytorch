/**
 * @fileoverview Agent interaction, live session control, piano
 * interaction, audio, and UI.
 *
 * @author Alex Scarlatos, Yusong Wu
 * Modified from https://github.com/lukewys/PianorollVis.js
 */

let visual, chordSynth, melodySynth, metronome, webMIDIInputs, instrumentMap;
let mouseDown = false, compKeyboardOctave = 4, keysToNotes, heldNotes = {};
let metronomeStatus = false, metronomeEvents = [], metronomeFreq = 'beat';
let curSession, lastSession, recorder, curAudioRecording;
let playBtn, metronomeBtn, bpmInput, timeSigInput, metronomeFreqBtn,
  interfaceSelect, liveSessionBtn, temperatureInput, silenceInput,
  lookaheadInput, commitaheadInput, modelSelect, chordInstSelect,
  melodyInstSelect, showChordsCheck, downloadSessionCheck, metronomeCheck,
  customVoicingsCheck, songSearchInput,
  songSearchResults, voicingBrowserChordSelect, voicingBrowserInfo,
  chordTimingSelect, chordTimingInfo, agentGateCheck, voicingTemperatureInput, inputLagInput;
let songCatalogue = [];   // [{dataset, split, id, title, artist, kind}, ...]
let bassSongsOnlyCheck;
let songSearchMatches = [];  // Current fuzzy hits, best first (capped)
let songSearchIndex = 0;     // Keyboard-highlighted row in songSearchMatches
let songSearchLastQuery = '';  // Query songSearchPool was narrowed down to
let songSearchPool = [];       // Every song matching it, not just the shown ones
const songSearchMaxResults = 30;
const songSearchMinChars = 2;
let voicingBrowserAll = [];   // [{pitches, count, num_songs}, ...] for the selected chord
let voicingBrowserList = [];  // the ones passing the browser's filters (voicingBrowserFilter)
let voicingBrowserNotesSelect, voicingBrowserLowInput, voicingBrowserHighInput,
  voicingBrowserLowValue, voicingBrowserHighValue;
let voicingBrowserIndex = -1;
let voicingBrowserActivePitches = [];  // pitches currently sounding/highlighted
let voicingBrowserOffTimeout = null;   // pending noteOff timeout for them

const fpb = 4;  // Frames per beat
const chordVelocity = 0.5;
const melodyVelocity = 0.5;
const metronomeVelocity = 0.5;
/** @enum {number} */
const compKeyOctaveRange = [2, 6];

const DEFAULTS = {
  bpm: window.location.hostname === 'localhost' ? 30 : 80,
  timeSignature: 4,
  temperature: 0.8,
  silence: window.location.hostname === 'localhost' ? 4 : 8,
  lookahead: window.location.hostname === 'localhost' ? 2 : 4,
  commitahead: window.location.hostname === 'localhost' ? 2 : 4,
  chordInstrument: 'Piano (Versilian)',
  melodyInstrument: 'Piano (Versilian)',
  inputLagMs: 30,  // see performerFrame
};

/** @enum {string} */
const pianoNotes = [
  'A0', 'A#0', 'B0', 'C1', 'C#1', 'D1', 'D#1', 'E1', 'F1', 'F#1', 'G1',
  'G#1', 'A1', 'A#1', 'B1', 'C2', 'C#2', 'D2', 'D#2', 'E2', 'F2', 'F#2',
  'G2', 'G#2', 'A2', 'A#2', 'B2', 'C3', 'C#3', 'D3', 'D#3', 'E3', 'F3',
  'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3', 'C4', 'C#4', 'D4', 'D#4', 'E4',
  'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4', 'C5', 'C#5', 'D5', 'D#5',
  'E5', 'F5', 'F#5', 'G5', 'G#5', 'A5', 'A#5', 'B5', 'C6', 'C#6', 'D6',
  'D#6', 'E6', 'F6', 'F#6', 'G6', 'G#6', 'A6', 'A#6', 'B6', 'C7', 'C#7',
  'D7', 'D#7', 'E7', 'F7', 'F#7', 'G7', 'G#7', 'A7', 'A#7', 'B7', 'C8'
];

/** @enum {number} */
const noteToPitch =
  pianoNotes.reduce((prev, note, idx) => ({ ...prev, [note]: idx + 21 }), {});

/** @enum {string} */
const pitchToNote =
  pianoNotes.reduce((prev, note, idx) => ({ ...prev, [idx + 21]: note }), {});

/**
 * Return if two arrays have the same elements
 * @param {Array!} arr1
 * @param {Array!} arr2
 * @return {boolean}
 */
function arraysEqual(arr1, arr2) {
  return arr1.length === arr2.length && arr1.every((el, i) => el === arr2[i]);
}

/**
 * Add input cleaning and an optional callback to a numeric input element
 * @param {Object!} inputEl
 * @param {callback!} callback
 */
function addNumericInputEventListener(inputEl, callback) {
  inputEl.addEventListener('input', event => {
    let floatVal = parseFloat(event.target.value);
    if (isNaN(floatVal)) {
      floatVal = inputEl.min || 0;
    }
    if (inputEl.min) {
      floatVal = Math.max(floatVal, inputEl.min);
    }
    if (inputEl.max) {
      floatVal = Math.min(floatVal, inputEl.max);
    }
    inputEl.value = floatVal;
    if (callback) {
      callback(floatVal);
    }
  });
}

/**
 * Wire a range-input slider to a text element showing its live value.
 * @param {string} inputId - Element ID of the <input type="range">
 * @param {string} valueId - Element ID of the value-display element
 * @param {number} decimals - Decimal places to display
 * @return {Element} The range input element (for further use)
 */
function bindSliderValueDisplay(inputId, valueId, decimals) {
  const inputEl = document.getElementById(inputId);
  const valueEl = document.getElementById(valueId);
  valueEl.textContent = inputEl.valueAsNumber.toFixed(decimals);
  inputEl.addEventListener('input', () => {
    valueEl.textContent = inputEl.valueAsNumber.toFixed(decimals);
  });
  return inputEl;
}

/** Initialize components after loading is done */
function showMainScreen() {
  document.querySelector('.splash').hidden = true;
  document.querySelector('.loaded').hidden = false;

  Tone.context.lookAhead = 0;
  Tone.start();
  enableClickingInputs();
  enableKeyboardInputs();
  document.addEventListener('mousedown', () => {
    mouseDown = true;
  });
  document.addEventListener('mouseup', () => {
    mouseDown = false;
  });

  // Enable WEBMIDI.js and then prepare the input interfaces
  enableMIDI();
}

/**
 * Save the most recent session's JSON to logs/realjam/ on the server
 * (POST /session_log); a browser can't choose a download folder. Falls back
 * to a browser download if the server can't be reached.
 */
async function saveSessionRecording() {
  if (!downloadSessionCheck.checked) {
    return;
  }
  const name = `GenJam Session - ${lastSession.startTime} ${lastSession.description}`;
  const json = JSON.stringify(lastSession);
  try {
    const result = await fetch(`${window.location.origin}/session_log`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: `{"name": ${JSON.stringify(name)}, "session": ${json}}`,
    });
    if (!result.ok) throw new Error(`server answered ${result.status}`);
    console.log('Session saved to', (await result.json()).path);
    return;
  } catch (e) {
    console.warn('Could not save the session on the server, downloading it instead', e);
  }
  const blob = new Blob([json], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  document.body.appendChild(a);
  a.style = 'display: none';
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

/** Start or stop a live session */
function toggleLiveSession() {
  if (curSession) {
    lastSession = curSession;
    lastSession.description = `sic${showChordsCheck.checked}_bpm${bpmInput.value}_met${metronomeStatus}_la${lookaheadInput.value}_com${commitaheadInput.value}_sil${silenceInput.value}_temp${temperatureInput.value}_voic${customVoicingsCheck.checked ? 'db' + voicingTemperatureInput.value : 'block'}_lag${lastSession.inputLagMs}_timing${chordTimingSelect.value}_${lastSession.model || modelSelect.value}`;
    const lookupSelect = document.getElementById('voicing-lookup-select');
    lastSession.voicingLookup = lookupSelect && lookupSelect.value ? lookupSelect.value : null;
    curSession = undefined;
    liveSessionBtn.textContent = 'Start Live Session';
    showChordsCheck.disabled = false;
    bpmInput.disabled = false;
    inputLagInput.disabled = false;
    timeSigInput.disabled = false;
    silenceInput.disabled = false;
    chordTimingSelect.disabled = false;
    Tone.Transport.stop();
    Tone.Transport.cancel();
    stopMetronome();
    pianoNotes.forEach(note => chordSynth.triggerRelease(note));
    midiAllOff();
    visual.clearScheduledNotes(0);
    visual.stopAllNotes();
    recorder.stop();
    updateChordTimingInfo();
  } else {
    startSession();
  }
}

/**
 * Start a fresh session and the Transport clock. Shared by live
 * sessions (toggleLiveSession) and scheduled robot-melody playback
 * (playSongMelody) -- the two differ only in where noteHistory events come
 * from afterward (live MIDI input vs. pre-scheduled playback).
 * @param {{referenceOnly?: boolean}=} options - referenceOnly suppresses all
 *   generation, making the session purely a playback vehicle for a song's own
 *   ground-truth melody and chords (see playSongReference). Reusing the
 *   session machinery this way means the Stop button, metronome and visual
 *   teardown all keep working unchanged.
 */
function startSession(options = {}) {
  Tone.Transport.stop();
  Tone.Transport.start();
  if (metronomeCheck.checked) {
    startMetronome();
  }
  curAudioRecording = [];
  recorder.start();
  curSession = {
    startTime: Date.now(),
    startFrame: undefined,  // Frame relative to Transport start that first
    // note is played
    noteHistory: [],        // All note hits and releases
    chordHistory: [],       // All chord pitch/symbol onsets with frames
    chordTokens: [],        // All chord tokens in frame format
    introSet: false,        // If model has generated intro section
    lastVoicing: null,      // Pitches of the last chord actually played,
                             // for voice-leading continuity across polls
                             // regardless of which decode rule produced it
    loopStarted: false,     // If the generation loop has been kicked off
    spaceChord: false,      // Complete mode: space held made the agent play
    heldChord: null,        // Complete mode: chord currently sustaining,
                             // written into chordTokens by fillChordTokens
    triggeredChord: null,   // Triggered mode: chord sustaining since the
                             // last space press
    voicingDecisions: new Map(),  // Database voicings: "frame|symbol" ->
                                   // {u, pitches}, see databaseVoicing
    model: modelSelect.value,  // the model the chord tokens belong to (sessionModel)
    candidates: null,       // Complete mode: model's ranked chord onsets
    heldKeys: new Set(),    // Complete mode: keys the performer is holding
    onsetGroup: null,       // Complete mode: notes starting within one
                             // gesture window, {notes: [{pitch, frame}]}
    chordKeys: new Set(),   // Complete mode: held keys that form the chord
    melodyKeys: new Set(),  // Complete mode: held keys read as melody
    completion: null,       // Complete mode: agent's notes sounding
    completionHistory: [],  // Complete mode: {frame, held, symbol, prob, rank, share, choices,
                             // pitches, soundedAt} per chord gesture;
                             // pitches/soundedAt only once the agent plays it
    gestureHistory: [],     // Complete mode: {frame, pitches, kind} per onset
                             // group, kind 'chord' or 'melody'
    gateHistory: [],        // Complete mode: {frame, agentIn} per change of
                             // the left-pedal / space gate
    triggerHistory: [],     // Triggered mode: {frame, symbol, pitches} per
                             // press, frame relative to session start
    playedNotes: [],        // Every note that sounds, see logPlayedNote
    playedChords: [],       // Every chord as it sounds, see logPlayedChord
    pedalLog: [],           // Every incoming pedal value, see logPedal
    referenceOnly: !!options.referenceOnly,  // Playing ground truth, no model
    inputLagMs: inputLagMs(),  // taken off the performer's key times, see performerFrame
    audioLatency: audioLatencyInfo(),  // {base, output} in s, as the browser reports them
  };
  liveSessionBtn.textContent = 'Stop Live Session';
  showChordsCheck.disabled = true;
  bpmInput.disabled = true;
  inputLagInput.disabled = true;  // one lag per session (logged)
  timeSigInput.disabled = true;
  silenceInput.disabled = true;
  chordTimingSelect.disabled = true;

  // Complete mode drives chords independently of the melody, so it doesn't
  // wait for a first note the way the auto loop does (see playNote).
  if (isCompleteMode()) loadChordVocab();
  if (isCompleteMode() && !curSession.referenceOnly) {
    startGenerationLoop();
  }
  updateChordTimingInfo();
}

/**
 * Length of one play-through, in frames. Uses the song's own length from the
 * data (num_beats) rather than where its last note ends: a melody that rests
 * through its final beats would otherwise restart early and drift off the bar
 * grid on every repetition. Never shorter than the notes themselves, in case
 * num_beats is missing or too small.
 * @param {Array<{offset: number}>} events - Anything with a beat-unit offset
 * @param {?number} numBeats
 * @return {number}
 */
function loopLengthFrames(events, numBeats) {
  const lastOffset = events.reduce(
    (latest, e) => Math.max(latest, Math.round(e.offset * fpb)), 0);
  return Math.max(numBeats ? numBeats * fpb : 0, lastOffset);
}

/**
 * Play a song's ground-truth melody on a schedule (not live input), feeding
 * it into the same session/generation pipeline as live playing so the model
 * generates chords in response -- lets voicing settings be tested hands-free
 * against real melodies. Loops indefinitely until the session is stopped
 * some other way (Stop Live Session, or picking a different song).
 * @param {Array<{pitch: number, onset: number, offset: number}>} notes
 *   Melody notes in quarter-note (beat) units, as returned by the
 *   /songs/.../melody endpoint.
 * @param {?number} numBeats - The song's length, for the loop period
 */
function playSongMelody(notes, numBeats) {
  if (!notes.length) {
    console.warn('Selected song has no melody notes; nothing to play.');
    return;
  }
  if (curSession) {
    toggleLiveSession();  // stop whatever session (live or robot) is running
  }
  startSession();

  // Anchor session frame 0 to "now", before scheduling anything relative to it.
  getSessionCurrentFrame();
  scheduleMelodyLoop(notes, curSession.startFrame,
    loopLengthFrames(notes, numBeats));

  // Kick off chord generation immediately, same as a live session's first
  // note would (see playNote).
  startGenerationLoop();
}

/**
 * Play a song's ground-truth melody *and* its ground-truth chords, with no
 * model in the loop at all -- the reference recording to A/B the generated
 * accompaniment against.
 *
 * Runs as a session so the Stop button and visual teardown work as usual,
 * but flagged referenceOnly so no /play or /chord_candidates request is ever
 * made (see startGenerationLoop).
 * @param {Array<{pitch: number, onset: number, offset: number}>} melody
 * @param {Array<{pitches: Array<number>, onset: number, offset: number,
 *   symbol: string}>} chords
 * @param {?number} numBeats - The song's length, for the loop period
 */
function playSongReference(melody, chords, numBeats) {
  if (!melody.length && !chords.length) {
    console.warn('Selected song has no reference content; nothing to play.');
    return;
  }
  if (curSession) {
    toggleLiveSession();
  }
  startSession({ referenceOnly: true });
  getSessionCurrentFrame();  // Anchor frame 0 before scheduling against it
  scheduleReferenceLoop(melody, chords, curSession.startFrame,
    loopLengthFrames([...melody, ...chords], numBeats));
}

/**
 * Play a song's ground-truth chords while a bass model plays the melody lane
 * (Song Playback: "Ground-truth chords"). The chords loop as in ground-truth
 * playback; the bass line is requested from the server in chunks ahead of
 * the playhead (/bass), generated against the same looping chords, and
 * played -- piano or laptop -- and drawn as it arrives. The first chunk is
 * fetched before playback starts so the opening notes are on time.
 * @param {Object} song - Catalogue entry
 * @param {Array<Object>} melody - The song's melody, only for its length
 * @param {Array<Object>} chords - The song's chords, with pitches
 * @param {?number} numBeats
 */
async function playSongWithBass(song, melody, chords, numBeats) {
  if (!chords.length) {
    console.warn('Selected song has no chords to play a bass line against.');
    return;
  }
  const model = modelSelect.value;
  if (!bassModelNames.includes(model)) {
    console.warn('No bass model selected.');
    return;
  }
  const loopFrames = loopLengthFrames([...melody, ...chords], numBeats);
  const bass = { song, model, loopFrames, tokens: [], open: null, last: null, busy: false };
  const chunk = Math.max(getLookaheadFrames(), BASS_MIN_CHUNK);
  const first = await fetchBassChunk(bass, 0, chunk);
  if (!first) return;

  if (curSession) {
    toggleLiveSession();
  }
  startSession({ referenceOnly: true });
  getSessionCurrentFrame();  // Anchor frame 0 before scheduling against it
  curSession.songKind = song.kind || 'melody';
  curSession.bass = bass;
  bass.base = curSession.startFrame;
  scheduleReferenceLoop([], chords, bass.base, loopFrames);
  addBassFrames(bass, first);
  bassLoop(curSession);
}
const BASS_MIN_CHUNK = 8;     // frames per request, at least
const BASS_AHEAD_FRAMES = 16;  // keep this much generated beyond the lookahead

/** Ask the server for `frames` bass frames from `frame` on; null on failure */
async function fetchBassChunk(bass, frame, frames) {
  try {
    const result = await fetch(`${window.location.origin}/bass`, {
      'method': 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: bass.model,
        dataset: bass.song.dataset,
        split: bass.song.split,
        id: bass.song.id,
        loopFrames: bass.loopFrames,
        frame,
        frames,
        temperature: temperatureInput.valueAsNumber,
        bassTokens: bass.tokens,
      })
    });
    if (!result.ok) throw new Error(`server answered ${result.status}`);
    return await result.json();
  } catch (e) {
    console.warn('Could not get bass frames', e);
    return null;
  }
}

/** Keep the bass line generated ahead of the playhead, until the session ends */
async function bassLoop(session) {
  if (curSession !== session || !session.bass) return;
  const bass = session.bass;
  const ahead = bass.tokens.length - getSessionCurrentFrame();
  if (!bass.busy && ahead < getLookaheadFrames() + BASS_AHEAD_FRAMES) {
    bass.busy = true;
    const chunk = Math.max(getLookaheadFrames(), BASS_MIN_CHUNK);
    const result = await fetchBassChunk(bass, bass.tokens.length, chunk);
    bass.busy = false;
    if (curSession !== session) return;
    if (result) addBassFrames(bass, result);
  }
  setTimeout(() => bassLoop(session), 100);
}

/**
 * Turn newly generated bass frames into notes. A note opens at an onset (or
 * a new pitch) and stays open while the frames hold it; it is scheduled once
 * it closes, when the next note is known, so a repeat of the same key gets
 * its release.
 */
function addBassFrames(bass, result) {
  result.frames.forEach(({ pitch, on }, k) => {
    const frame = bass.tokens.length;
    bass.tokens.push(result.tokens[k]);
    const continues = bass.open && pitch === bass.open.pitch && !on;
    if (continues) return;
    const next = pitch === null ? null : { pitch, start: frame };
    if (bass.open) scheduleBassNote(bass, bass.open, frame, next);
    bass.open = next;
    if (next) {
      // Show it falling in as soon as it's known; its release is set on close
      visual.scheduleNoteOn(pitch, bass.base + frame, 'orange', 1.0);
    }
  });
}

/** Schedule a closed bass note [start, end) on the piano or laptop */
function scheduleBassNote(bass, note, end, next) {
  const absOn = bass.base + note.start, absOff = bass.base + end;
  scheduleRobotNote(note.pitch, true, absOn);   // laptop sound, key highlight
  scheduleRobotNote(note.pitch, false, absOff);
  visual.scheduleNoteOff(note.pitch, absOff, true);
  scheduleMidiMelody(
    [{ pitch: note.pitch, onset: note.start / fpb, offset: end / fpb }],
    bass.base, undefined, false,
    next ? [{ pitch: next.pitch, onset: next.start / fpb, offset: next.start / fpb + 1 }] : []);
}

/**
 * Schedule one play-through of a song's ground-truth melody and chords, then
 * schedule the next right after it -- loops until the session is stopped,
 * same as scheduleMelodyLoop.
 * @param {Array<Object>} melody
 * @param {Array<Object>} chords
 * @param {number} baseFrame - Absolute frame this play-through starts at
 * @param {number} loopFrames - Period between play-throughs
 */
function scheduleReferenceLoop(melody, chords, baseFrame, loopFrames) {
  const nextFrame = baseFrame + loopFrames;

  melody.forEach(({ pitch, onset, offset }) => {
    const onFrame = baseFrame + Math.round(onset * fpb);
    const offFrame = baseFrame + Math.round(offset * fpb);
    scheduleRobotNote(pitch, true, onFrame);
    scheduleRobotNote(pitch, false, offFrame);
  });
  const midiReleases = scheduleMidiMelody(melody, baseFrame, loopFrames, true);
  // The whole melody is known in advance here, so it can fall towards the
  // keys like the chords do. (Not in a live session with a robot melody:
  // processAgentAction clears every scheduled note on each model response.)
  // All onsets before any release, so a repeated pitch's release attaches
  // to the right note (scheduleNoteOff picks the latest earlier onset).
  // With a MIDI output each block ends at its real note-off, re-strike gap
  // included, so the waterfall shows what the piano is actually sent.
  melody.forEach(({ pitch, onset }) => visual.scheduleNoteOn(
    pitch, baseFrame + Math.round(onset * fpb), 'orange', 1.0));
  if (midiReleases) {
    midiReleases.forEach(({ pitch, offFrame }) =>
      visual.scheduleNoteOff(pitch, offFrame, true));
  } else {
    melody.forEach(({ pitch, offset }) => visual.scheduleNoteOff(
      pitch, baseFrame + Math.round(offset * fpb)));
  }

  chords.forEach(({ pitches, onset, offset, symbol }) => {
    const onFrame = baseFrame + Math.round(onset * fpb);
    const offFrame = baseFrame + Math.round(offset * fpb);
    // Same scheduling path the generated chords use, so the two sound
    // identical apart from the choice of chord
    scheduleChordPitches(
      pitches, onFrame, frameToTransportTime(onFrame), true, 1.0, symbol);
    scheduleChordPitches(
      pitches, offFrame, frameToTransportTime(offFrame), false);
  });

  if (midiChordOut) {
    // Each annotated chord is a fresh strike. A release only goes out where
    // the music actually rests: where the next chord starts at the same
    // frame, releasing and pressing at once would defeat the piano's action.
    const onsetFrames = new Set(
      chords.map(({ onset }) => baseFrame + Math.round(onset * fpb)));
    chords.forEach(({ pitches, onset, offset }, i) => {
      const onFrame = baseFrame + Math.round(onset * fpb);
      const offFrame = baseFrame + Math.round(offset * fpb);
      // Previous chord, wrapping to the last one of the previous pass
      const prevOnFrame = i > 0 ?
        baseFrame + Math.round(chords[i - 1].onset * fpb) :
        baseFrame - loopFrames + Math.round(chords[chords.length - 1].onset * fpb);
      const spf = Tone.Time('16n').toSeconds();
      // The gap is decided shortly before the re-strike, from the settings
      // then; if it differs from what was drawn, move the drawn releases of
      // the previous chord's shared keys to match.
      const prev = chords[i > 0 ? i - 1 : chords.length - 1];
      const prevOffFrame = prevOnFrame + Math.round((prev.offset - prev.onset) * fpb);
      const drawnGap = midiRestrikeGapFor((onFrame - prevOnFrame) * spf);
      scheduleMidiChordLive(pitches, onFrame, prevOnFrame, gap => {
        if (!showChordsCheck.checked || gap === drawnGap || prevOffFrame !== onFrame) return;
        prev.pitches.filter(p => pitches.includes(p)).forEach(p =>
          visual.scheduleNoteOff(p, onFrame - gap / spf, true));
      });
      // Draw the releases the piano is sent: keys the next chord shares are
      // lifted a re-strike gap before it, the rest at the change itself
      if (showChordsCheck.checked) {
        const next = chords[i + 1] ||
          (chords[0] && { ...chords[0], onset: chords[0].onset + loopFrames / fpb });
        const nextOnFrame = next && baseFrame + Math.round(next.onset * fpb);
        pitches.forEach(pitch => {
          const lifted = nextOnFrame === offFrame && next.pitches.includes(pitch);
          const gapFrames = lifted
            ? midiRestrikeGapFor((nextOnFrame - onFrame) * spf) / spf : 0;
          visual.scheduleNoteOff(pitch, offFrame - gapFrames, true);
        });
      }
      if (!onsetFrames.has(offFrame)) {
        scheduleMidiChord([], offFrame, false);
      }
    });
  }

  // Schedule the next play-through ahead of the boundary rather than at it,
  // so its opening notes are already falling into view, and so MIDI events
  // sent early (Send Early) for its first beats aren't already stale.
  const aheadFrames = Math.min(
    Math.floor(loopFrames / 2), getLookaheadFrames() + 2 * fpb);
  Tone.Transport.scheduleOnce(() => {
    if (curSession && curSession.referenceOnly) {
      scheduleReferenceLoop(melody, chords, nextFrame, loopFrames);
    }
  }, frameToTransportTime(nextFrame - aheadFrames));
}

/**
 * Schedule one play-through of a song's notes starting at baseFrame, then
 * schedule another play-through right after it ends -- repeats indefinitely
 * as long as the session started by playSongMelody is still the active one.
 * @param {Array<{pitch: number, onset: number, offset: number}>} notes
 * @param {number} baseFrame - Absolute (Transport-relative) frame this
 *   play-through starts at
 * @param {number} loopFrames - Period between play-throughs
 */
function scheduleMelodyLoop(notes, baseFrame, loopFrames) {
  const nextFrame = baseFrame + loopFrames;
  notes.forEach(({ pitch, onset, offset }) => {
    const onFrame = baseFrame + Math.round(onset * fpb);
    const offFrame = baseFrame + Math.round(offset * fpb);
    scheduleRobotNote(pitch, true, onFrame);
    scheduleRobotNote(pitch, false, offFrame);
  });
  scheduleMidiMelody(notes, baseFrame, loopFrames);

  Tone.Transport.scheduleOnce(() => {
    // Stops the loop once the session has been ended some other way
    // (Stop Live Session button, or a different song/live session started --
    // both replace curSession, and starting a new one also cancels this via
    // Tone.Transport.cancel() in toggleLiveSession's stop branch).
    if (curSession) {
      scheduleMelodyLoop(notes, nextFrame, loopFrames);
    }
  }, frameToTransportTime(nextFrame));
}

/**
 * Schedule a single robot-melody note-on/off event at an absolute
 * (Transport-relative) frame -- the scheduled-playback analogue of
 * playNote/releaseNote, which push to noteHistory immediately for live input.
 * @param {number} pitch - Raw MIDI pitch
 * @param {boolean} on - Note-on (true) or note-off (false)
 * @param {number} absFrame - Absolute frame (relative to Transport start) to
 *   fire at
 */
function scheduleRobotNote(pitch, on, absFrame) {
  const note = pitchToNote[pitch];
  Tone.Transport.scheduleOnce(() => {
    if (!curSession) {
      return;
    }
    curSession.noteHistory.push(
      { on, pitch, frame: absFrame - curSession.startFrame });
    // With a MIDI output the piano plays the melody (scheduleMidiMelody);
    // this event still feeds the model and the piano roll.
    if (on) {
      if (!midiChordOut) {
        melodySynth.triggerAttack(note, '+0', melodyVelocity);
        logPlayedNote('melody', pitch, true, melodyVelocity);
      }
      visual.noteOn(pitch);
    } else {
      if (!midiChordOut) {
        melodySynth.triggerRelease(note);
        logPlayedNote('melody', pitch, false);
      }
      visual.noteOff(pitch);
    }
  }, frameToTransportTime(absFrame));
}

/**
 * Set the lookahead value in beats
 * @param {number} lookahead
 */
function setLookahead(lookahead) {
  visual.setVisibleFrames(lookahead * fpb);
  commitaheadInput.max = lookahead;
  if (commitaheadInput.valueAsNumber > lookahead) {
    commitaheadInput.value = lookahead;
  }
}

/**
 * Get the number of lookahead frames
 * @return {number}
 */
function getLookaheadFrames() {
  return lookaheadInput.valueAsNumber * fpb;
}

/**
 * Get the number of commitahead frames
 * @return {number}
 */
function getCommitaheadFrames() {
  return commitaheadInput.valueAsNumber * fpb;
}

/**
 * Get the number of initial frames of silence
 * @return {number}
 */
function getSilenceFrames() {
  return silenceInput.valueAsNumber * fpb;
}

/**
 * Get the current frame of the session
 * Also set the start frame on the first call
 * @return {number}
 */
function getSessionCurrentFrame() {
  // Get current frame
  const softFrame = getTransportFrame();
  const frame = Math.round(softFrame);

  // Set start frame if needed, subtract modulo to always start on the beat.
  // Compare with undefined: a start frame of 0 is valid, and a falsy test
  // would re-anchor it on the next call, a beat or more later.
  if (curSession.startFrame === undefined) {
    curSession.startFrame = frame - (frame % fpb);
  }

  return frame - curSession.startFrame;
}

/** The input lag setting, in ms (0-150; the default if the page has none) */
function inputLagMs() {
  const v = inputLagInput && inputLagInput.valueAsNumber;
  return Number.isFinite(v) ? Math.max(0, Math.min(v, 150)) : DEFAULTS.inputLagMs;
}

/**
 * Session frame of a key the performer presses or releases now: the
 * transport frame minus the input lag, rounded to the 16th (2026-11-18).
 * Keys played to the click arrive late against the transport -- the click's
 * audio output delay plus the Disklavier's MIDI input delay: in the logged
 * sessions a median +30 ms, in the first session after the chord-frame fix
 * +53 ms, with on-beat presses up to ~100 ms late counted for the 16th after
 * the beat (journal/reference/REALJAM_CHORD_TIMING.md, Part 6). The lag is
 * taken off before rounding. Never later than the current frame. Pedal and
 * space keep the current frame (pedal triggers were measured on the grid).
 * @return {number}
 */
function performerFrame() {
  const now = getSessionCurrentFrame();  // also anchors the session's frame 0
  const lag = inputLagMs() / 1000 / Tone.Time('16n').toSeconds();
  const frame = Math.round(getTransportFrame() - lag) - curSession.startFrame;
  return Math.max(0, Math.min(frame, now));
}

/** The browser's reported audio latencies (s), for the session log */
function audioLatencyInfo() {
  try {
    const raw = Tone.context.rawContext;
    return { base: raw.baseLatency ?? null, output: raw.outputLatency ?? null };
  } catch (e) {
    return null;
  }
}

/**
 * If chords follow auto mode's schedule but only sound when the performer
 * presses space. The model runs exactly as in auto mode -- same polling,
 * lookahead, commit and chord history, never told about the presses -- so
 * the schedule is the one auto mode would play; a press plays whatever that
 * schedule has sounding at that moment. See triggerScheduledChord.
 * @return {boolean}
 */
function isTriggeredChordMode() {
  return (chordTimingSelect.value === 'triggered' ||
    chordTimingSelect.value === 'triggered-history') &&
    !(curSession && curSession.referenceOnly);
}

/**
 * The triggered-mode variant in which the model hears what actually sounded
 * (2026-11-17, for trying out): before every request the past of the chord
 * history is rewritten from the presses (rewriteTriggeredHistory) -- the
 * pressed chord from its press frame on, silence after a release -- instead
 * of keeping the model's own schedule there. The future (its plan) is kept.
 * @return {boolean}
 */
function isTriggeredHistoryMode() {
  return chordTimingSelect.value === 'triggered-history' && isTriggeredChordMode();
}

/**
 * Onset and hold token of the chord the plan has at session frame `frame`,
 * or null (silence / nothing planned). Needs the token ranges the server
 * sends with each plan (processAgentAction).
 * @return {?{onset: number, hold: number}}
 */
function plannedChordTokens(frame) {
  const tok = curSession.chordTokens[frame];
  const range = curSession.chordHoldRange, offset = curSession.chordOnsetOffset;
  if (tok === undefined || tok < 0 || !range || offset === undefined) return null;
  const isHold = t => t >= range[0] && t <= range[1];
  if (isHold(tok)) return { onset: tok + offset, hold: tok };
  if (isHold(tok - offset)) return { onset: tok, hold: tok - offset };
  return null;
}

/**
 * Rewrite the past of the chord history (frames before now) to what sounded:
 * from each press the pressed chord (onset at the press frame, then holds),
 * after a release silence (-1, which the server fills with SILENCE).
 */
function rewriteTriggeredHistory() {
  const now = getSessionCurrentFrame();
  const presses = curSession.triggerHistory;
  let current = null, next = 0;
  for (let f = 0; f < now; f++) {
    while (next < presses.length && presses[next].frame <= f) {
      const press = presses[next++];
      current = press.release || !press.tokens ? null : press;
    }
    curSession.chordTokens[f] = !current ? -1
      : f === current.frame ? current.tokens.onset : current.tokens.hold;
  }
}

/**
 * Chord completion: the performer plays some notes and the model completes
 * them into a chord. All played notes are constraints, not melody -- the
 * model's melody input stays silent and it relies on the chord history it
 * builds up. The model's ranked candidates are fetched ahead of time (see
 * candidateLoop), so a gesture is completed without a round trip.
 * @return {boolean}
 */
function isCompleteMode() {
  return (chordTimingSelect.value === 'complete' || chordTimingSelect.value === 'input') &&
    !(curSession && curSession.referenceOnly);
}

/**
 * Input mode (2026-11-17, for testing): complete mode's reading of the
 * performer's playing -- chord vs. melody, the chord's top note as melody,
 * the model's chord from the held notes, all written into its history -- but
 * the agent never plays. Shows the chord the held notes form exactly in the
 * vocabulary (exactChordFor) next to what the model reads from them.
 * @return {boolean}
 */
function isInputMode() {
  return chordTimingSelect.value === 'input' && isCompleteMode();
}

// Chord vocabularies by model (/chord_vocab?model=), loaded once each: models
// differ (2,821 or 2,969 chords, different token ids -- journal/reference/
// TOKENIZERS.md, section 0), so every chord token the page holds is in the
// vocabulary of the session's model (sessionModel)
const chordVocabs = new Map();

/**
 * The model chords are requested from and token ids belong to: the session's
 * (set at its start, changed only by onModelChanged, which remaps the
 * history first), else the one selected.
 * @return {string}
 */
function sessionModel() {
  return (curSession && curSession.model) || modelSelect.value;
}

/** Load a model's chord vocabulary (once per model) */
async function loadChordVocab(model = sessionModel()) {
  if (chordVocabs.has(model)) return chordVocabs.get(model);
  try {
    const result = await fetch(
      `${window.location.origin}/chord_vocab?model=${encodeURIComponent(model)}`);
    const vocab = await result.json();
    chordVocabs.set(model, vocab);
    return vocab;
  } catch (e) {
    console.warn('Could not load the chord vocabulary', e);
    return null;
  }
}

/**
 * The model menu changed (2026-11-19). Outside a session nothing to do. In
 * one, the history's chord tokens (and the held chord's) are remapped by name
 * into the new model's vocabulary (/chord_token_map; a chord it lacks becomes
 * silence) and only then does the session switch, in one synchronous step;
 * candidate lists of the old model are dropped, and responses still on their
 * way for it are ignored (they check sessionModel).
 */
async function onModelChanged() {
  const session = curSession;
  const to = modelSelect.value;
  if (!session || !session.model || session.model === to) {
    if (session) session.model = to;
    return;
  }
  const from = session.model;
  let map = null;
  try {
    const result = await fetch(`${window.location.origin}/chord_token_map?from=` +
      `${encodeURIComponent(from)}&to=${encodeURIComponent(to)}`);
    map = result.ok ? await result.json() : null;
  } catch (e) {
    console.warn('Could not remap the chord history', e);
  }
  await loadChordVocab(to);
  if (curSession !== session || session.model !== from || modelSelect.value !== to) return;
  if (!map) {  // keep the old model rather than feed it the wrong ids
    modelSelect.value = from;
    return;
  }
  const remap = t => (typeof t === 'number' && t >= 0 && t < map.length ? map[t] : t);
  session.chordTokens = session.chordTokens.map(remap);
  if (session.heldChord) {
    session.heldChord.onsetToken = remap(session.heldChord.onsetToken);
    session.heldChord.holdToken = remap(session.heldChord.holdToken);
  }
  session.candidates = null;
  session.candidateLists = new Map();
  session.chordHoldRange = undefined;  // the next plan brings the new model's
  session.chordOnsetOffset = undefined;
  session.model = to;
  if (!session.modelSwitches) session.modelSwitches = [];
  session.modelSwitches.push({ frame: getSessionCurrentFrame(), from, to });
  updateChordTimingInfo();
}

/**
 * music21's reading of played notes, mapped onto the model's vocabulary
 * (POST /recognize_chord, chord_recognition.py): {symbol, root, commonName,
 * how: 'exact' | 'incomplete' | 'nearest'}, or null.
 * @param {Array<number>} pitches
 */
async function recognizeWithMusic21(pitches) {
  try {
    const result = await fetch(`${window.location.origin}/recognize_chord`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pitches, model: sessionModel() }),  // a chord of its vocabulary
    });
    return result.ok ? await result.json() : null;
  } catch (e) {
    console.warn('music21 recognition failed', e);
    return null;
  }
}

/**
 * The ranked candidate list from the vocabulary and the model's probability
 * for each of its chords (same order): most probable first.
 * @param {Array<Object>} vocab - /chord_vocab entries
 * @param {Array<number>} probs
 * @return {Array<Object>}
 */
function candidatesFromProbs(vocab, probs) {
  return vocab.map((chord, i) => ({ ...chord, prob: probs[i] }))
    .sort((a, b) => b.prob - a.prob);
}

/**
 * How sure the model is among `candidates`, their probabilities renormalised
 * to sum to 1: exp(entropy) = the effective number of equally likely chords
 * (the measure of journal experiments/ONSET_TIMING.md: about 2 on a downbeat,
 * dozens between beats), and the most probable chord's share.
 * @param {Array<Object>} candidates - with prob, most probable first
 * @return {?{choices: number, top: string, topShare: number, total: number}}
 */
function onsetCertainty(candidates) {
  const total = candidates.reduce((a, c) => a + (c.prob || 0), 0);
  if (!(total > 0)) return null;
  let entropy = 0;
  candidates.forEach(c => {
    const p = (c.prob || 0) / total;
    if (p > 0) entropy -= p * Math.log(p);
  });
  return { choices: Math.exp(entropy), top: candidates[0].symbol,
    topShare: candidates[0].prob / total, total };
}

/** "63%" / "0.4%" */
function formatShare(share) {
  const pct = 100 * share;
  return pct >= 10 ? `${Math.round(pct)}%` : `${pct.toFixed(1)}%`;
}

/** "≈2" / "≈37" */
function formatChoices(choices) {
  return `≈${choices < 10 ? choices.toFixed(1).replace(/\.0$/, '') : Math.round(choices)}`;
}

/** A chord symbol's bass pitch class: the slash note if any, else its root */
function chordBassPc(chord) {
  const slash = chord.symbol.lastIndexOf('/');
  const bass = slash >= 0 ? chord.symbol.slice(slash + 1) : '';
  const m = /^([A-G])([b#]*)$/.exec(bass);
  if (!m) return chord.root;
  const letter = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 }[m[1]];
  const shift = [...m[2]].reduce((a, c) => a + (c === '#' ? 1 : -1), 0);
  return ((letter + shift) % 12 + 12) % 12;
}

/**
 * The vocabulary chords whose pitch classes are exactly the held notes', best
 * first: the one whose bass is the lowest held note (C/E for E3 G3 C4, C for
 * C3 E3 G3), then the rest. Empty if none (or the vocabulary isn't loaded).
 * @param {Array<number>} held
 * @return {Array<string>} symbols
 */
function exactChordsFor(held) {
  const chordVocab = chordVocabs.get(sessionModel());
  if (!chordVocab || !held.length) return [];
  const pcs = new Set(held.map(p => p % 12));
  const lowestPc = Math.min(...held) % 12;
  const exact = chordVocab.filter(c => c.pitchClasses.length === pcs.size &&
    c.pitchClasses.every(pc => pcs.has(pc)));
  exact.sort((a, b) => (chordBassPc(b) === lowestPc) - (chordBassPc(a) === lowestPc));
  return exact.map(c => c.symbol);
}

/**
 * Kick off whichever generation loop the current chord-timing mode uses.
 * Idempotent -- called from several places that can each be the first to
 * happen in a session (first melody note, robot playback, session start in
 * complete mode, or a space press before any of those).
 */
function startGenerationLoop() {
  if (!curSession || curSession.loopStarted || curSession.referenceOnly) {
    return;
  }
  curSession.loopStarted = true;
  getSessionCurrentFrame();  // Anchor session frame 0 before scheduling
  if (isCompleteMode()) {
    candidateLoop();
  } else {
    syncWithServer();
  }
}

/** Send session history as context to model to get new chord predictions */
async function syncWithServer() {
  // Exit loop if current session ended (in case ended during timeout)
  if (!curSession) {
    return;
  }

  // Send current frame and context to server for generation
  const curFrame = getSessionCurrentFrame();
  if (isTriggeredHistoryMode()) rewriteTriggeredHistory();
  const model = sessionModel();
  const result = await fetch(`${window.location.origin}/play`, {
    'method': 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model,
      notes: curSession.noteHistory,
      chordTokens: curSession.chordTokens,
      frame: curFrame + 1,  // Request chords to play at the next frame
      lookahead: getLookaheadFrames(),
      commitahead: getCommitaheadFrames(),
      silenceTill: getSilenceFrames(),
      temperature: temperatureInput.valueAsNumber,
      introSet: curSession.introSet,
      // Database voicings are sampled here in the browser (databaseVoicing);
      // the server then just sends its block chord as a fallback
      useCustomVoicings: false,
      bassLine: !customVoicingsCheck.checked && inputIsBassLine(),
      prevVoicing: curSession.lastVoicing,
      vlWeight: CUSTOM_VOICING_VL_WEIGHT,
      regWeight: CUSTOM_VOICING_REG_WEIGHT,
    })
  });
  const json = await result.json();

  // Exit loop if current session ended (in case ended during fetch)
  if (!curSession) {
    return;
  }

  // Schedule chords based on agent response -- unless the model was switched
  // meanwhile: its tokens are in the old vocabulary
  if (model === sessionModel()) processAgentAction(json);

  // Send next request to server, wait until next frame if haven't advanced yet
  if (curFrame === getSessionCurrentFrame()) {
    const fps = fpb * Tone.Transport.bpm.value / 60;
    setTimeout(syncWithServer, 1000 / fps);
  } else {
    syncWithServer();
  }
}

/**
 * Bring curSession.chordTokens up to the current frame, so the model sees an
 * accurate account of what has been sounding. Frames with no chord are left
 * as -1, which the server fills with SILENCE.
 */
function fillChordTokens() {
  const curFrame = getSessionCurrentFrame();
  for (let i = 0; i <= curFrame; i++) {
    if (curSession.chordTokens[i] === undefined) {
      curSession.chordTokens[i] = -1;
    }
  }
  const held = curSession.heldChord;
  if (held) {
    // Everything after the onset frame is the same chord still ringing
    for (let i = held.startFrame + 1; i <= curFrame; i++) {
      curSession.chordTokens[i] = held.holdToken;
    }
  }
}

// Custom-voicing scoring weights; were sliders, fixed at their old defaults
const CUSTOM_VOICING_VL_WEIGHT = 0.5;
const CUSTOM_VOICING_REG_WEIGHT = 0.2;

/**
 * The current melody note: the highest one sounding now, else the last one
 * played, else middle C (tells a bass line from a tune, inputIsBassLine).
 * @return {number}
 */
function currentMelodyPitch() {
  const active = new Map();  // pitch -> still held
  let last = null;
  for (const { on, pitch } of curSession.noteHistory) {
    if (on) {
      active.set(pitch, true);
      last = pitch;
    } else {
      active.delete(pitch);
    }
  }
  if (active.size) return Math.max(...active.keys());
  return last !== null ? last : 60;
}

// ---------------------------------------------------------------------------
// Database voicings (2026-11-16), auto and triggered mode. Each chord gets a
// voicing pianists actually played: one of the stored voicings of the voicing
// database (/voicings, at their recorded pitches), drawn among those that fit
// -- spanning at least an octave, notes below C3 at least a fifth apart, top
// note at least 3 semitones below the lowest melody note of the last two
// beats, or every note above G3 under a walking bass line -- with probability
// proportional to count^(1/T), T the Voicing Temperature. If none fits, the
// block chord. Every model response re-plans the future, so a
// chord's voicing is re-checked about five times a second while it is still
// ahead: it is sticky (kept while it still fits) and redrawn only when the
// melody has dropped below it, with the chord's own random number (drawn
// once), so repeated checks never flicker. The waterfall always shows the
// current choice; with a MIDI output it is final MIDI_DECIDE_AHEAD before the
// chord, when its MIDI is decided. Holds keep the voicing of the chord they hold.
// ---------------------------------------------------------------------------
const VOICING_MELODY_WINDOW_BEATS = 2;  // the melody's "recent range"
const VOICING_MELODY_GAP = 3;  // semitones: chord top at least this far below it
const VOICING_DEFAULT_CEILING = 69;  // A4: the ceiling with no recent melody (as a melody at C5)
// Two register rules on the stored voicings (user, 2026-11-16):
const VOICING_MIN_SPAN = 12;  // lowest to highest note at least an octave (no close triads)
const VOICING_LOW_REGISTER = 48;  // below C3 ...
const VOICING_LOW_MIN_INTERVAL = 7;  // ... adjacent notes at least a fifth apart
const VOICING_LOWEST = 36;  // C2: nothing lower (2026-11-18)
const VOICING_PRELOAD = 8;  // complete mode: the top candidates' voicings fetched ahead
const BASS_LINE_VOICING_FLOOR = 55;     // G3, as on the server
const voicingCandidateCache = new Map();  // symbol -> [{pitches, count}] or 'loading'

/**
 * The chord's stored voicings, or null while they are being fetched (the
 * first time a chord comes up; the server's block chord stands in until then).
 * @param {string} symbol
 * @return {?Array<{pitches: Array<number>, count: number}>}
 */
function voicingCandidates(symbol) {
  const cached = voicingCandidateCache.get(symbol);
  if (Array.isArray(cached)) return cached;
  if (!cached) loadVoicingCandidates(symbol);
  return null;
}

/**
 * Fetch a chord's stored voicings into the cache (once).
 * @param {string} symbol
 * @return {Promise<Array<{pitches: Array<number>, count: number}>>}
 */
function loadVoicingCandidates(symbol) {
  const cached = voicingCandidateCache.get(symbol);
  if (Array.isArray(cached)) return Promise.resolve(cached);
  if (cached && cached.then) return cached;
  const pending = fetch(`${window.location.origin}/voicings?chord=${encodeURIComponent(symbol)}`)
    .then(result => result.json())
    .then(list => {
      const voicings = list.filter(v => !v.legacy && v.count > 0)
        .map(v => ({ pitches: v.pitches, count: v.count }));
      voicingCandidateCache.set(symbol, voicings);
      return voicings;
    })
    .catch(() => {
      voicingCandidateCache.delete(symbol);
      return [];
    });
  voicingCandidateCache.set(symbol, pending);
  return pending;
}

/**
 * Database voicings for a song's own chords (song playback), by the same rule
 * as the model's chords: stored voicings that fit, drawn by count^(1/T), the
 * block chord if none fits. The melody is known in advance here, so the
 * ceiling is exact: 3 semitones below the lowest melody note sounding during
 * the chord or begun in the two beats before it. Under a walking bass line
 * (a bass song's own, or the bass model's) every note above G3 instead.
 * Each chord is drawn once, when the song starts.
 * @param {Array<{symbol: string, pitches: Array<number>, onset: number,
 *   offset: number}>} chords - Beat units; pitches = the block chord
 * @param {Array<{pitch: number, onset: number, offset: number}>} melody -
 *   What plays with them ([] if no melody plays)
 * @param {boolean} bassLine
 */
async function voiceReferenceChords(chords, melody, bassLine) {
  const symbols = [...new Set(chords.flatMap(c => [c.symbol, c.symbol.split('/')[0]]))];
  await Promise.all(symbols.map(loadVoicingCandidates));
  return chords.map(chord => {
    let candidates = voicingCandidateCache.get(chord.symbol) || [];
    if (!candidates.length && chord.symbol.includes('/')) {
      candidates = voicingCandidateCache.get(chord.symbol.split('/')[0]) || [];
    }
    let constraint = { ceiling: null, floor: null };
    if (bassLine) {
      constraint = { ceiling: null, floor: BASS_LINE_VOICING_FLOOR };
    } else {
      const over = melody.filter(n => n.onset < chord.offset &&
        n.offset > chord.onset - VOICING_MELODY_WINDOW_BEATS);
      if (over.length) {
        constraint = { ceiling: Math.min(...over.map(n => n.pitch)) - VOICING_MELODY_GAP, floor: null };
      }
    }
    const pitches = drawVoicing(voicingPool(candidates, constraint), Math.random());
    return { ...chord, pitches: pitches || blockChordInRegister(chord.pitches) };
  });
}

/** Lowest melody note sounding now or begun in the last two beats, else null */
function recentMelodyLow() {
  if (!curSession || !curSession.noteHistory.length) return null;
  const now = getSessionCurrentFrame();
  const since = now - VOICING_MELODY_WINDOW_BEATS * fpb;
  const sounding = new Set();
  let low = null;
  curSession.noteHistory.forEach(({ on, pitch, frame }) => {
    if (on) sounding.add(pitch); else sounding.delete(pitch);
    if (on && frame >= since) low = low === null ? pitch : Math.min(low, pitch);
  });
  sounding.forEach(pitch => { low = low === null ? pitch : Math.min(low, pitch); });
  return low;
}

/** What a voicing must satisfy now: {ceiling} (max allowed top) and/or {floor} */
function voicingConstraint() {
  if (inputIsBassLine()) return { ceiling: null, floor: BASS_LINE_VOICING_FLOOR };
  const low = recentMelodyLow();
  // No recent melody: as under a melody at C5 (2026-11-18). Without a ceiling
  // any stored voicing could be drawn, up to C6 (space chords with no melody
  // in complete mode, where chord keys are not melody)
  if (low === null) return { ceiling: VOICING_DEFAULT_CEILING, floor: null };
  return { ceiling: low - VOICING_MELODY_GAP, floor: null };
}

function voicingFits(pitches, { ceiling, floor }) {
  return (ceiling === null || Math.max(...pitches) <= ceiling) &&
    (floor === null || Math.min(...pitches) > floor);
}

/**
 * The fixed-octave block chord (chord tones C3-B3, the bass C2-B2) under the
 * register rule below C3: a chord tone less than a fifth above a bass below
 * C3 goes up an octave (B2 D#3 F#3 B3 -> B2 F#3 B3 D#4). 2026-11-18.
 * @param {Array<number>} pitches
 * @return {Array<number>}
 */
function blockChordInRegister(pitches) {
  if (!pitches || !pitches.length) return pitches;
  const bass = Math.min(...pitches);
  if (bass >= VOICING_LOW_REGISTER) return pitches;
  const lifted = pitches.map(p => (p !== bass && p - bass < VOICING_LOW_MIN_INTERVAL ? p + 12 : p));
  return [...new Set(lifted)];
}

/**
 * The voicing-lookup menu (2026-11-23): lists the server's lookups and
 * switches the active one. The server's lookup is global, so the page's
 * cached voicings are dropped on a switch.
 */
async function setupVoicingLookupSelect() {
  const select = document.getElementById('voicing-lookup-select');
  if (!select) return;
  try {
    const { lookups, active } = await (await fetch(`${window.location.origin}/voicing_lookups`)).json();
    select.innerHTML = '';
    lookups.forEach(name => {
      const option = document.createElement('option');
      option.value = option.textContent = name;
      select.appendChild(option);
    });
    if (active) select.value = active;
    select.disabled = lookups.length < 2;
  } catch (e) {
    select.disabled = true;
    return;
  }
  select.addEventListener('change', async () => {
    select.disabled = true;
    try {
      const result = await fetch(`${window.location.origin}/voicing_lookup`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: select.value }),
      });
      if (!result.ok) throw new Error(await result.text());
      voicingCandidateCache.clear();
    } catch (e) {
      console.warn('voicing lookup switch failed', e);
    }
    select.disabled = false;
    select.blur();  // so space keeps triggering chords
  });
}

/**
 * The register rules every database voicing must meet: nothing below C2, it
 * spans at least an octave, and where the lower of two adjacent notes is
 * below C3 they are at least a fifth apart. The rules judge only what the
 * agent adds (2026-11-23: a performer playing close intervals below C3 got no
 * completion at all): notes among `keys` -- the performer's own -- are exempt,
 * i.e. a note below C2 or a close low pair is fine if the performer plays it,
 * and a voicing narrower than an octave is fine if it adds nothing. Must match
 * realjam/voicing_fit.py (register_exempt_notes).
 * @param {Array<number>} pitches
 * @param {Array<number>=} keys - the performer's keys (complete mode)
 * @return {boolean}
 */
function voicingRegisterOk(pitches, keys = []) {
  const p = [...new Set(pitches)].sort((a, b) => a - b);
  const own = new Set(keys);
  const need = new Set(p.filter(x => x < VOICING_LOWEST));
  for (let i = 1; i < p.length; i++) {
    if (p[i - 1] < VOICING_LOW_REGISTER && p[i] - p[i - 1] < VOICING_LOW_MIN_INTERVAL) {
      need.add(p[i - 1]);
      need.add(p[i]);
    }
  }
  if (p.length && p[p.length - 1] - p[0] < VOICING_MIN_SPAN) p.forEach(x => need.add(x));
  return [...need].every(x => own.has(x));
}

/**
 * The candidates a chord's voicing is drawn from, most common first: its
 * stored voicings that meet the register rules and fit the melody /
 * bass-line constraint, at their recorded pitches (no shifting). Empty if
 * none does; the block chord is then played.
 * @return {Array<{pitches: Array<number>, count: number}>}
 */
function voicingPool(candidates, constraint) {
  return candidates.filter(v => voicingRegisterOk(v.pitches) && voicingFits(v.pitches, constraint));
}

/**
 * Draw from the pool with probability proportional to count^(1/T), T the
 * Voicing Temperature, by inverse CDF with the chord's own uniform number `u`
 * (the pool keeps the database's order, most common first, so a nearby `u`
 * means a similar rank).
 * @return {?Array<number>}
 */
function drawVoicing(pool, u) {
  if (!pool.length) return null;
  const tValue = voicingTemperatureInput.valueAsNumber;
  if (!(tValue > 0)) return pool[0].pitches;
  // In log space: count^(1/T) overflows for small T
  const logMax = Math.log(Math.max(...pool.map(v => v.count)));
  const weights = pool.map(v => Math.exp((Math.log(v.count) - logMax) / tValue));
  let r = u * weights.reduce((a, b) => a + b, 0);
  for (let i = 0; i < pool.length; i++) {
    r -= weights[i];
    if (r <= 0) return pool[i].pitches;
  }
  return pool[pool.length - 1].pitches;
}

/**
 * Database voicing for one planned chord (see the comment block above). The
 * voicing is kept while it is still in the chord's pool -- the pool changes
 * when the melody drops below it -- and redrawn otherwise.
 * @param {number} frame - Absolute Transport frame of the chord
 * @param {string} symbol
 * @param {boolean} on - Onset (vs. hold)
 * @param {Array<number>} pitches - The server's block chord (fallback)
 * @return {Array<number>}
 */
function databaseVoicing(frame, symbol, on, pitches) {
  if (!symbol || !pitches || !pitches.length) return pitches;  // silence
  const before = getScheduledChordAt(frame - 1);
  if (!on && before && before.symbol === symbol && before.pitches.length) {
    return before.pitches;  // a hold keeps the chord it holds
  }
  let candidates = voicingCandidates(symbol);
  if (candidates && !candidates.length && symbol.includes('/')) {
    // A slash chord with no stored voicings: voice its chord
    candidates = voicingCandidates(symbol.split('/')[0]);
  }
  if (!candidates || !candidates.length) return blockChordInRegister(pitches);
  const key = `${frame}|${symbol}`;
  let decision = curSession.voicingDecisions.get(key);
  if (!decision) {
    decision = { frame, u: Math.random(), pitches: null };
    curSession.voicingDecisions.set(key, decision);
  }
  const secondsAhead = (frame - getTransportFrame()) * Tone.Time('16n').toSeconds();
  const final = midiChordOut && secondsAhead < MIDI_DECIDE_AHEAD;
  if (decision.pitches && final) return decision.pitches;
  const pool = voicingPool(candidates, voicingConstraint());
  const current = decision.pitches && decision.pitches.join(',');
  if (!current || !pool.some(v => v.pitches.join(',') === current)) {
    decision.pitches = drawVoicing(pool, decision.u) || blockChordInRegister(pitches);
  }
  return decision.pitches;
}

/**
 * The schedule's entry sounding at an absolute Transport frame (latest
 * chordHistory entry at or before it), or null before the first chord.
 * @param {number} targetFrame
 * @return {?Object}
 */
function getScheduledChordAt(targetFrame) {
  for (let i = curSession.chordHistory.length - 1; i >= 0; i--) {
    if (curSession.chordHistory[i].scheduleFrame <= targetFrame) {
      return curSession.chordHistory[i];
    }
  }
  return null;
}

/**
 * Play the chord the schedule has sounding right now, releasing the one
 * held since the last press. Bound to space in triggered mode. A press
 * before the scheduled change re-strikes the current chord; the schedule
 * itself is untouched, so the model never hears about the presses.
 */
function triggerScheduledChord(source = 'space') {
  const entry = getScheduledChordAt(Math.floor(getTransportFrame()));
  // Exactly the voicing the waterfall shows (the current database voicing,
  // see databaseVoicing), so what falls into view is what sounds
  const pitches = entry ? entry.pitches || [] : [];

  if (curSession.triggeredChord) {
    curSession.triggeredChord.forEach(pitch => {
      if (!midiChordOut) {
        chordSynth.triggerRelease(pitchToNote[pitch]);
        logPlayedNote('chord', pitch, false);
      }
      visual.noteOff(pitch);
    });
  }
  pitches.forEach(pitch => {
    if (!midiChordOut) {
      chordSynth.triggerAttack(pitchToNote[pitch], '+0', chordVelocity);
      logPlayedNote('chord', pitch, true, chordVelocity);
    }
    visual.noteOn(pitch, 'blue');
  });
  midiPlayNow(pitches);
  logPlayedChord(entry ? entry.symbol : '', pitches);

  curSession.triggeredChord = pitches;
  const frame = getSessionCurrentFrame();
  curSession.triggerHistory.push({
    frame,
    symbol: entry ? entry.symbol : '',
    pitches,
    source,  // 'space' or 'pedal'
    // The pressed chord's tokens, for the variant that writes what sounded
    // into the history (rewriteTriggeredHistory)
    tokens: plannedChordTokens(frame),
  });
  updateChordTimingInfo();
}

const COMPLETION_WINDOW_MS = 50;  // notes this close together form one gesture
const CANDIDATE_LISTS_KEPT = 4;  // candidate lists kept by frame, for a gesture's own frame
const MIN_CHORD_NOTES = 2;        // a gesture with fewer notes is melody
// A chord gesture whose top note is at or above middle C also plays that
// note as melody (user, 2026-11-17): E3 G3 C4 = chord C and melody C4
const CHORD_TOP_MELODY_FROM = 60;  // C4
// A single note at least this far below a held melody note is a bass note (a
// chord gesture); closer, it is a legato melody step (2026-11-18: in the
// session logs, 88% of single notes 1-2 semitones below a held key and 65% at
// 3-4 came with that key released within 150 ms; from a fourth down, 24-36%)
const BASS_BELOW_MELODY = 5;  // a fourth
// Label colour of a chord the model inferred from the held notes (not exactly them)
const CHORD_INFERRED_LABEL_COLOR = '#ffeb3b';

// The gate that lets the agent play in complete mode: the left (soft) pedal,
// which the Disklavier sends as continuous CC 67, with hysteresis so a
// pedal hovering at one value can't flicker; or space held (laptop, or a
// USB foot switch that types space).
const AGENT_PEDAL_CC = 67;
const AGENT_PEDAL_ON = 40;
const AGENT_PEDAL_OFF = 20;
let agentGateHeld = false;

/*
 * Complete mode reads what the performer plays by timing, not register.
 * Notes starting within COMPLETION_WINDOW_MS of each other form one onset
 * group. A group of at least MIN_CHORD_NOTES is a chord gesture, wherever
 * its notes lie on the keyboard; a single note is melody, which goes to the
 * model's melody input and never starts or holds a chord. The chord lasts
 * while any of its own keys is held; melody keys don't sustain it.
 *
 * The agent plays only while the gate is open (isAgentIn). With the gate
 * closed, chord gestures are still recognised -- the most probable
 * compatible candidate -- and written into the chord history without
 * sounding, so the model keeps following the performer's progression and
 * continues it once let in. Opening the gate while a chord is held makes the
 * agent complete it right away; closing it releases the agent's notes.
 */

/** If the agent may play: gate open, or the gate is switched off */
function isAgentIn() {
  if (isInputMode()) return false;  // input mode: the agent never plays
  return agentGateHeld || !(agentGateCheck && agentGateCheck.checked);
}

/** Open or close the gate (pedal crossing a threshold, space down/up) */
function setAgentGate(held) {
  if (held === agentGateHeld) return;
  const wasIn = isAgentIn();
  agentGateHeld = held;
  if (curSession && isCompleteMode() && isAgentIn() !== wasIn) agentGateChanged();
  updateChordTimingInfo();
}

/**
 * The agent was let in or out (the left pedal, or the checkbox), complete
 * mode. The pedal only gives permission: going in plays nothing, not even
 * over chord notes already held (2026-11-18: completing those at the press
 * surprised the performer); the agent completes the next chord gesture, or
 * plays when space is pressed (spaceDown). Out: the agent's notes stop, a
 * chord of its own ends.
 */
function agentGateChanged() {
  const session = curSession;
  session.gateHistory.push({ frame: getSessionCurrentFrame(), agentIn: isAgentIn() });
  if (isAgentIn()) return;
  session.spaceChord = false;
  if (session.heldChord && session.heldChord.free) endChord();
  else silenceCompletion();
}

/**
 * Space pressed in complete mode, the agent in (2026-11-18): it plays at once,
 * like triggered mode -- over chord notes held and not yet completed, it
 * completes them (picked again with the agent in); with none held, a chord of
 * its own (playAgentChord). Space up ends what it started (spaceUp). Agent
 * out: nothing.
 */
function spaceDown() {
  const session = curSession;
  session.gateHistory.push({ frame: getSessionCurrentFrame(), space: true });
  if (!isAgentIn() || session.completion) return;
  session.spaceChord = true;
  if (session.chordKeys.size) startChord([...session.chordKeys]);
  else playAgentChord();
}

function spaceUp() {
  const session = curSession;
  session.gateHistory.push({ frame: getSessionCurrentFrame(), space: false });
  if (!session.spaceChord) return;
  session.spaceChord = false;
  if (session.heldChord && session.heldChord.free) endChord();
  else silenceCompletion();
}

/**
 * The agent's chord of its own (space, no chord notes held): drawn from the
 * model's whole candidate list for this moment (nothing to filter by), by
 * probability^(1/Temperature), voiced like auto mode's database voicings
 * (under the recent melody), else the fixed-octave block chord.
 */
async function playAgentChord() {
  const session = curSession;
  // The chord starts at the frame of the space press, picked from the
  // model's list for that frame (2026-11-18; was the latest list, usually for
  // the next 16th -- see candidatesForFrame)
  const frame = getSessionCurrentFrame();
  const { candidates: list, refetched } = await candidatesForFrame(session, frame);
  const stillWanted = () => curSession === session && session.spaceChord && isAgentIn() &&
    !session.heldChord && !session.chordKeys.size;
  if (!list || !stillWanted()) return;
  const picked = pickCandidate(list, [], false);
  if (!picked) return;
  const { chord, rank, share, choices } = picked;
  const candidates = await loadVoicingCandidates(chord.symbol);
  if (!stillWanted()) return;
  const pitches = drawVoicing(voicingPool(candidates, voicingConstraint()), Math.random()) ||
    blockChordInRegister([...chord.pitchClasses.map(pc => 48 + pc), 36 + chord.root]);  // the block chord
  fillChordTokens();
  session.chordTokens[frame] = chord.onsetToken;
  session.completionHistory.push({ frame, readAt: getSessionCurrentFrame(), refetched,
    held: [], symbol: chord.symbol, prob: chord.prob, rank, share, choices, match: 'agent' });
  session.heldChord = { ...chord, rank, share, choices, startFrame: frame, free: true, drawn: [],
    match: 'agent', entry: session.completionHistory.length - 1 };
  fillChordTokens();  // frames since the press: this chord's holds
  const sorted = [...pitches].sort((a, b) => a - b);
  const drawFrame = getTransportFrame();
  sorted.forEach(pitch => {
    if (!midiChordOut) {
      chordSynth.triggerAttack(pitchToNote[pitch], '+0', chordVelocity);
      logPlayedNote('chord', pitch, true, chordVelocity);
    }
    visual.noteOn(pitch, 'lightBlue');
    visual.scheduleNoteOn(pitch, drawFrame, 'lightBlue', 1.0,
      pitch === sorted[0] ? chord.symbol : '', true);
  });
  midiStrikeNow(sorted);
  logPlayedChord(chord.symbol, sorted);
  session.completion = { pitches: sorted };
  session.chordHistory.push({ scheduleFrame: frame, pitches: sorted, symbol: chord.symbol, eventIDs: [] });
  Object.assign(session.completionHistory[session.completionHistory.length - 1],
    { pitches: sorted, soundedAt: frame });
  updateChordTimingInfo();
}

/** Left-pedal value 0-127 from MIDI, with hysteresis */
let leftPedalDown = false;

/**
 * Left-pedal value 0-127 from MIDI, with hysteresis. Triggered mode: like
 * space, down plays the scheduled chord and up stops it. Otherwise it is the
 * complete-mode agent gate.
 */
function onAgentPedal(value) {
  let down = leftPedalDown;
  if (!leftPedalDown && value >= AGENT_PEDAL_ON) down = true;
  else if (leftPedalDown && value <= AGENT_PEDAL_OFF) down = false;
  if (down === leftPedalDown) return;
  leftPedalDown = down;
  if (curSession && isTriggeredChordMode()) {
    if (down) triggerScheduledChord('pedal'); else releaseTriggeredChord('pedal');
    return;
  }
  setAgentGate(down);
}

/** Fetch the model's ranked chord onsets for the next frame. One request at
 * a time: a call while one is in flight asks for one more afterwards and
 * returns the pending one. */
function fetchCandidates() {
  const session = curSession;
  if (!session) return Promise.resolve();
  if (session.candidatePromise) {
    session.candidateRefetch = true;
    return session.candidatePromise;
  }
  session.candidatePromise = requestCandidates(session).finally(() => {
    session.candidatePromise = null;
    if (curSession === session && session.candidateRefetch) {
      session.candidateRefetch = false;
      fetchCandidates();
    }
  });
  return session.candidatePromise;
}

async function requestCandidates(session, frame = getSessionCurrentFrame() + 1) {
  fillChordTokens();
  const notesBefore = melodyEventsBefore(session, frame);
  try {
    // All chords: the vocabulary once per session, then only the model's
    // probabilities in its order (format "probs") -- 2026-11-17, was the top
    // 200, which for 6% of typical hand positions held no fitting chord
    const model = sessionModel();
    const vocab = await loadChordVocab(model);
    if (!vocab) throw new Error('chord vocabulary not loaded');
    const result = await fetch(`${window.location.origin}/chord_candidates`, {
      'method': 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        notes: session.noteHistory,  // melody notes only, see classifyOnsetGroup
        chordTokens: session.chordTokens,
        frame,
        format: 'probs',
      })
    });
    if (!result.ok) throw new Error(`server answered ${result.status}`);
    const answer = await result.json();
    if (model !== sessionModel()) return null;  // switched meanwhile: the old model's
    const candidates = Array.isArray(answer) ? answer  // an older server: the ranked top 200
      : candidatesFromProbs(vocab, answer.probs);
    session.candidates = candidates;
    session.certainty = onsetCertainty(candidates);
    // Kept by the frame they predict, for a gesture on that frame
    // (candidatesForFrame); only the last few frames are needed
    if (!session.candidateLists) session.candidateLists = new Map();
    session.candidateLists.set(frame, { candidates, notesBefore });
    for (const f of session.candidateLists.keys()) {
      if (f < frame - CANDIDATE_LISTS_KEPT) session.candidateLists.delete(f);
    }
    // The completion's voicings: the likely chords' loaded ahead (cached)
    candidates.slice(0, VOICING_PRELOAD).forEach(c => loadVoicingCandidates(c.symbol));
    session.candidateError = null;
    if (curSession === session) updateChordTimingInfo();
    return candidates;
  } catch (e) {
    console.warn('Could not fetch chord candidates', e);
    session.candidateError = String(e.message || e);
  }
  if (curSession === session) updateChordTimingInfo();
  return null;
}

/** How many melody events (on and off) lie before `frame`: what a candidate
 * list for that frame saw of the melody (the server cuts the melody there) */
function melodyEventsBefore(session, frame) {
  return session.noteHistory.filter(e => e.frame < frame).length;
}

/**
 * The model's chord candidates for a chord starting at `frame` (2026-11-18).
 * A gesture is read 50 ms after its first key, and the list refreshed every
 * 16th is for the frame after the one it was requested in, so the latest list
 * was often for the wrong 16th: in the logged sessions a third of the chords
 * went in a 16th after their first key, and the model's prediction differs a
 * lot between a beat and the 16th after it (journal/reference/
 * REALJAM_CHORD_TIMING.md, Part 6). A kept list for `frame` that saw the same
 * melody is used as is; otherwise the model is asked for `frame` now (one
 * round trip). Falls back to the latest list if that fails.
 * @return {{candidates: Array|null, refetched: boolean}}
 */
async function candidatesForFrame(session, frame) {
  const kept = session.candidateLists && session.candidateLists.get(frame);
  if (kept && kept.notesBefore === melodyEventsBefore(session, frame)) {
    return { candidates: kept.candidates, refetched: false };
  }
  const fresh = await requestCandidates(session, frame);
  return { candidates: fresh || session.candidates, refetched: true };
}

/**
 * Keep the candidates fresh while waiting for gestures: every frame (16th),
 * as often as auto mode asks for its plan (2026-11-17; was once per beat plus
 * after every melody note), so a new melody note is in the list within a 16th.
 * One request in flight at a time (fetchCandidates).
 */
async function candidateLoop() {
  if (!curSession || !isCompleteMode()) return;
  await fetchCandidates();
  if (!curSession || !isCompleteMode()) return;
  setTimeout(candidateLoop, 60000 / Tone.Transport.bpm.value / fpb);
}

/** Performer pressed a key in complete mode: it joins the open onset group */
function onCompletionNoteDown(pitch) {
  const frame = performerFrame();
  curSession.heldKeys.add(pitch);
  if (!curSession.onsetGroup) {
    curSession.onsetGroup = { notes: [] };
    setTimeout(classifyOnsetGroup, COMPLETION_WINDOW_MS);
  }
  curSession.onsetGroup.notes.push({ pitch, frame });
}

/** Performer released a key in complete mode */
function onCompletionNoteUp(pitch) {
  const frame = performerFrame();
  curSession.heldKeys.delete(pitch);
  // A chord's top note can be melody too: end both
  const wasMelody = curSession.melodyKeys.delete(pitch);
  if (wasMelody) curSession.noteHistory.push({ on: false, pitch, frame });
  if (curSession.chordKeys.delete(pitch)) {
    if (!curSession.chordKeys.size) endChord();
    return;
  }
  if (wasMelody) return;
  // Released inside the gesture window: noted, classified when it closes
  const group = curSession.onsetGroup;
  const note = group && group.notes.find(n => n.pitch === pitch && n.offFrame === undefined);
  if (note) note.offFrame = frame;
}

/**
 * A single note low enough to be a bass note rather than melody: a fourth or
 * more below the lowest melody key held, or, with none held, below middle C.
 * @param {number} pitch
 * @return {boolean}
 */
function isBassNote(pitch) {
  const melody = [...curSession.melodyKeys].filter(p => curSession.heldKeys.has(p) && p !== pitch);
  return melody.length ? pitch <= Math.min(...melody) - BASS_BELOW_MELODY
    : pitch < CHORD_TOP_MELODY_FROM;
}

/** The gesture window closed: chord gesture or melody? */
async function classifyOnsetGroup() {
  const session = curSession;
  if (!session || !session.onsetGroup) return;
  const notes = session.onsetGroup.notes;
  session.onsetGroup = null;
  let kind = notes.length >= MIN_CHORD_NOTES ? 'chord' : 'melody';
  // Complete and input mode: a single new note
  // below the top of the held chord's body (its keys as struck, without a
  // melody top note) changes the chord -- the keys still held plus it --
  // instead of being melody. Above the chord body it stays melody. "As
  // struck": replacing the top note (G3 released, F3 pressed) still counts.
  const single = notes.length === 1 && notes[0].offFrame === undefined ? notes[0] : null;
  if (single && isCompleteMode() && session.heldChord && session.chordKeys.size &&
      !session.heldChord.loneBass) {
    const struck = session.heldChord.drawn || [...session.chordKeys];
    const body = struck.filter(p => p !== session.chordMelodyTop);
    if (body.length && single.pitch < Math.max(...body)) kind = 'chord change';
  }
  // A bass note (2026-11-18): a single note well below the melody --
  // a fourth or more below a held melody note, or, with none held, below
  // middle C -- is a chord gesture too. Over a held chord it joins it; alone
  // (or after another lone bass note, which it replaces: a moving bass line)
  // it is a chord of its own, completed by the agent.
  if (single && kind === 'melody' && isCompleteMode() && isBassNote(single.pitch)) {
    kind = session.heldChord && session.chordKeys.size && !session.heldChord.loneBass
      ? 'chord change' : 'bass';
  }
  if (kind === 'chord change' || kind === 'bass') {
    session.gestureHistory.push({ frame: single.frame, pitches: [single.pitch], kind, melody: null });
    visual.recolorNote(single.pitch, 'blue');
    // A change above the bass (2026-11-18): a voice moving inside the
    // chord (a suspension resolving, F3 Bb3 Eb4 -> F3 A3 Eb4) is read anew
    // but is no new chord prompt -- the agent keeps sounding (startChord's
    // voiceMove). A new lowest note is a new prompt.
    const voiceMove = kind === 'chord change' && session.chordKeys.size &&
      single.pitch > Math.min(...session.chordKeys);
    if (voiceMove) session.gestureHistory[session.gestureHistory.length - 1].voiceMove = true;
    await startChord(kind === 'bass' ? [single.pitch] : [...session.chordKeys, single.pitch],
      { voiceMove, frame: single.frame });
    return;
  }
  // A chord gesture reaching middle C or higher: its top note is melody too
  const top = notes.reduce((a, b) => (b.pitch > a.pitch ? b : a));
  const topIsMelody = kind === 'chord' && top.pitch >= CHORD_TOP_MELODY_FROM;
  session.gestureHistory.push({ frame: notes[0].frame, pitches: notes.map(n => n.pitch), kind,
    melody: topIsMelody ? top.pitch : null });
  if (kind === 'chord') session.chordMelodyTop = topIsMelody ? top.pitch : null;
  if (kind === 'melody') {
    notes.forEach(n => {
      session.noteHistory.push({ on: true, pitch: n.pitch, frame: n.frame });
      if (n.offFrame !== undefined) {
        session.noteHistory.push({ on: false, pitch: n.pitch, frame: n.offFrame });
      } else {
        session.melodyKeys.add(n.pitch);
      }
    });
    updateChordTimingInfo();  // the next per-frame refresh includes the new melody
    return;
  }
  if (topIsMelody) {
    session.noteHistory.push({ on: true, pitch: top.pitch, frame: top.frame });
    if (top.offFrame !== undefined) {
      session.noteHistory.push({ on: false, pitch: top.pitch, frame: top.offFrame });
    } else {
      session.melodyKeys.add(top.pitch);
    }
  }
  // Show the reading on the keys: chord keys blue, melody stays orange (a
  // chord's top note read as melody too stays orange)
  notes.forEach(n => {
    if (n.offFrame === undefined && !(topIsMelody && n === top)) visual.recolorNote(n.pitch, 'blue');
  });
  const gestureKeys = notes.filter(n => n.offFrame === undefined).map(n => n.pitch);
  const gestureFrame = Math.min(...notes.map(n => n.frame));  // its first key's
  // A bass struck before the chord (read as melody then, alone)
  // joins it if still held below the chord: its melody note ends here
  if (isCompleteMode() && gestureKeys.length) {
    const lowest = Math.min(...gestureKeys);
    const absorbed = [...session.melodyKeys].filter(p => p < lowest && session.heldKeys.has(p));
    const frame = gestureFrame;
    absorbed.forEach(p => {
      session.melodyKeys.delete(p);
      session.noteHistory.push({ on: false, pitch: p, frame });
      visual.recolorNote(p, 'blue');
    });
    // A held lone bass note (its own chord until now) joins too
    if (session.heldChord && session.heldChord.loneBass) {
      absorbed.push(...[...session.chordKeys].filter(p => p < lowest && session.heldKeys.has(p)));
    }
    if (absorbed.length) session.gestureHistory[session.gestureHistory.length - 1].absorbed = absorbed;
    gestureKeys.push(...absorbed);
  }
  await startChord(gestureKeys, { frame: gestureFrame });
}

/**
 * A chord gesture: pick the chord, write it into the history, and let the
 * agent complete it if it's in. A new gesture while a chord is held
 * replaces that chord (a legato chord change).
 * @param {Array<number>} held - The gesture's keys still down
 * @param {{voiceMove: boolean}=} options - voiceMove: one note changed above
 *   the bass. The chord is read anew (history, label) but the agent is not
 *   prompted: its notes that belong to the new chord keep sounding, the
 *   rest are released, nothing is struck (2026-11-18).
 */
async function startChord(held, { voiceMove = false, frame: onsetFrame } = {}) {
  const session = curSession;
  if (!held.length) return;  // the whole gesture was released in the window
  // The chord starts at its first key's frame, not when the gesture was read
  // (50 ms later, often the next 16th) -- 2026-11-18, see candidatesForFrame
  const now = getSessionCurrentFrame();
  const frame = onsetFrame === undefined ? now : Math.min(onsetFrame, now);
  // With the agent in, the chord is also weighted by how well the keys fit
  // its playable stored voicings (pickCandidate), fetched alongside
  const wantFit = isAgentIn() && !isInputMode() && !voiceMove;
  const [{ candidates, refetched }, fit] = await Promise.all([
    candidatesForFrame(session, frame), wantFit ? voicingFitFor(held) : Promise.resolve(null)]);
  if (curSession !== session || !candidates) return;
  held = held.filter(p => session.heldKeys.has(p));  // released while waiting
  if (!held.length) return;
  const picked = pickCandidate(candidates, held, !isAgentIn(), fit);
  if (!picked) return;
  const kept = voiceMove ? session.completion : null;
  if (kept) session.completion = null;  // endChord must not silence it
  if (session.heldChord) endChord(false);
  const { chord, rank, share, choices } = picked;
  held.forEach(p => session.chordKeys.add(p));

  fillChordTokens();
  session.chordTokens[frame] = chord.onsetToken;
  session.completionHistory.push({ frame, readAt: getSessionCurrentFrame(), refetched,
    held, symbol: chord.symbol, prob: chord.prob, rank, share, choices,
    fit: picked.fit, fitAsked: !!fit });
  session.heldChord = { ...chord, rank, share, choices, startFrame: frame,
    loneBass: held.length === 1,  // a bass note on its own (classifyOnsetGroup)
    entry: session.completionHistory.length - 1 };
  fillChordTokens();  // frames since the onset: this chord's holds
  if (voiceMove) {
    session.completionHistory[session.heldChord.entry].voiceMove = true;
    keepCompletion(kept, chord);
  }
  // Recognised (the held notes are exactly the chord) or inferred (the model
  // added notes, or -- the fallback -- left some of the held ones out)
  const heldPcs = new Set(held.map(p => p % 12));
  const chordPcs = new Set(chord.pitchClasses);
  const missing = [...heldPcs].filter(pc => !chordPcs.has(pc));
  session.heldChord.match = missing.length ? 'partial'
    : heldPcs.size === chordPcs.size ? 'exact' : 'inferred';
  session.heldChord.missing = missing;
  session.completionHistory[session.completionHistory.length - 1].match = session.heldChord.match;
  // In the waterfall like a chord in auto mode: a block per key, the symbol at
  // the lowest note, until the chord ends (endChord); white when recognised
  // exactly, yellow when inferred
  const keys = [...session.chordKeys].sort((a, b) => a - b);
  const labelFrame = getTransportFrame();
  let labelColor = session.heldChord.match === 'exact' ? 'white' : CHORD_INFERRED_LABEL_COLOR;
  let label = chord.symbol;
  if (isCompleteMode() && !isAgentIn()) {
    // The agent is silent (input mode; complete mode with the pedal up): the
    // history gets what was played -- music21's reading of the held notes;
    // input mode also shows the exact vocabulary name. With the agent in, the
    // history keeps the chord that sounds (the model's completion).
    const exact = isInputMode() ? exactChordsFor(keys) : [];
    session.heldChord.exact = exact;
    session.completionHistory[session.completionHistory.length - 1].exact = exact[0] || null;
    const thisChord = session.heldChord;
    const m21 = await recognizeWithMusic21(keys);
    if (curSession !== session || session.heldChord !== thisChord) return;  // replaced meanwhile
    session.heldChord.music21 = m21;
    session.completionHistory[session.completionHistory.length - 1].music21 = m21;
    if (m21) {
      // The history gets music21's chord -- what was played -- instead of the
      // model's reading (kept as modelSymbol for comparison); a vocabulary
      // token, so the model knows it
      session.heldChord.modelSymbol = chord.symbol;
      Object.assign(session.heldChord, { symbol: m21.symbol,
        onsetToken: m21.onsetToken, holdToken: m21.holdToken });
      session.chordTokens[frame] = m21.onsetToken;
      fillChordTokens();  // holds since the onset, with music21's chord
      session.completionHistory[session.completionHistory.length - 1].historySymbol = m21.symbol;
      label = m21.display; labelColor = 'white';
    } else if (exact.length) { label = exact[0]; labelColor = 'white'; }
  }
  keys.forEach(p => visual.scheduleNoteOn(p, labelFrame, 'blue', 1.0,
    p === keys[0] ? label : '', true, labelColor));
  session.heldChord.drawn = keys;
  if (isAgentIn() && !voiceMove) soundCompletion();  // a voice move is no prompt
  updateChordTimingInfo();
}

/**
 * The agent plays its part of the held chord (if not already sounding): a
 * stored voicing of the chord that contains the performer's keys at their
 * pitches (the chord body: without a top note read as melody), drawn like
 * every database voicing (register rules, by count^(1/T)); the agent plays
 * its other notes, none of them above the melody ceiling (3 semitones under
 * the recent melody low) or on a key the performer holds. If no stored
 * voicing does, the agent adds nothing (2026-11-18; replaced a constructed
 * placement around the hand, completeAround).
 */
async function soundCompletion() {
  const session = curSession;
  const chord = session && session.heldChord;
  if (!chord || session.completion || !session.chordKeys.size) return;
  const voicing = await completionVoicing(chord.symbol);
  if (curSession !== session || session.heldChord !== chord || session.completion ||
    !session.chordKeys.size || !isAgentIn()) return;
  const pitches = voicing ? voicing.added : [];
  const drawFrame = getTransportFrame();
  pitches.forEach(pitch => {
    if (!midiChordOut) {
      chordSynth.triggerAttack(pitchToNote[pitch], '+0', chordVelocity);
      logPlayedNote('chord', pitch, true, chordVelocity);
    }
    visual.noteOn(pitch, 'lightBlue');  // the agent's part; your chord keys are blue
    visual.scheduleNoteOn(pitch, drawFrame, 'lightBlue', 1.0, '', true);
  });
  midiStrikeNow(pitches);
  // The chord that sounds: the performer's chord keys and the agent's notes
  logPlayedChord(chord.symbol, [...session.chordKeys, ...pitches]);
  const frame = getSessionCurrentFrame();
  session.completion = { pitches, voicing: voicing && voicing.pitches,
    fitting: voicing ? voicing.fitting : 0 };
  session.chordHistory.push(
    { scheduleFrame: frame, pitches, symbol: chord.symbol, eventIDs: [] });
  const entry = session.completionHistory[chord.entry];
  if (entry && !entry.pitches) {
    Object.assign(entry, { pitches, soundedAt: frame, voicing: session.completion.voicing,
      fittingVoicings: session.completion.fitting });
  }
  updateChordTimingInfo();
}

/**
 * What a completion is built around (completionVoicing, voicingFitFor): the
 * keys its voicing must contain -- the chord body, i.e. the chord keys
 * without a top note read as melody (all of them if that leaves none) --, the
 * keys held now (its other notes may not lie on them) and the melody ceiling.
 * @param {Array<number>} chordKeys
 * @return {{keys: Array<number>, held: Set<number>, ceiling: ?number}}
 */
function completionContext(chordKeys) {
  const session = curSession;
  const body = chordKeys.filter(p => !session.melodyKeys.has(p));
  const low = recentMelodyLow();
  return { keys: body.length ? body : [...chordKeys], held: session.heldKeys,
    ceiling: low === null ? null : low - VOICING_MELODY_GAP };
}

/**
 * The voicing fit of every chord for a chord gesture (server /voicing_fit,
 * realjam/voicing_fit.py): the share of its stored voicings, by count, that
 * contain the keys at their pitches and that the completion could play
 * (completionVoicing's rule). Null if it can't be had.
 * @param {Array<number>} chordKeys
 * @return {Promise<?Map<string, number>>} symbol -> fit
 */
async function voicingFitFor(chordKeys) {
  const { keys, held, ceiling } = completionContext(chordKeys);
  const model = sessionModel();
  try {
    const vocab = await loadChordVocab(model);
    const result = await fetch(`${window.location.origin}/voicing_fit`, {
      'method': 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ keys, held: [...held], ceiling, model }),
    });
    if (!result.ok) throw new Error(`server answered ${result.status}`);
    const { fit } = await result.json();
    if (!vocab || !Array.isArray(fit) || fit.length !== vocab.length) return null;
    return new Map(vocab.map((chord, i) => [chord.symbol, fit[i]]));
  } catch (e) {
    console.warn('Could not fetch the voicing fit', e);
    return null;
  }
}

/**
 * The completion's voicing for `symbol` under the keys held now.
 * @param {string} symbol
 * @return {Promise<?{pitches: Array<number>, added: Array<number>, fitting: number}>}
 *   the stored voicing drawn, the notes the agent adds, and how many stored
 *   voicings fitted; null if none did
 */
async function completionVoicing(symbol) {
  const session = curSession;
  let candidates = await loadVoicingCandidates(symbol);
  if (!candidates.length && symbol.includes('/')) {
    candidates = await loadVoicingCandidates(symbol.split('/')[0]);  // as databaseVoicing
  }
  const { keys, held, ceiling } = completionContext([...session.chordKeys]);
  const pool = candidates.filter(v => {
    if (!keys.every(k => v.pitches.includes(k)) || !voicingRegisterOk(v.pitches, keys)) return false;
    const added = v.pitches.filter(p => !keys.includes(p));
    return added.every(p => !held.has(p) && (ceiling === null || p <= ceiling));
  });
  const pitches = drawVoicing(pool, Math.random());
  if (!pitches) return null;
  return { pitches, added: pitches.filter(p => !keys.includes(p)).sort((a, b) => a - b),
    fitting: pool.length };
}

/**
 * After a voice move: the agent's sounding notes that belong to the new
 * chord keep sounding, the others are released; nothing is struck. Without
 * sounding notes the agent stays silent (the move is no prompt).
 * @param {?Object} kept - The completion sounding before the move
 * @param {Object} chord - The new chord (pitchClasses)
 */
function keepCompletion(kept, chord) {
  const session = curSession;
  if (!kept) return;
  const pcs = new Set(chord.pitchClasses);
  const stay = kept.pitches.filter(p => pcs.has(p % 12));
  const endFrame = getTransportFrame();
  kept.pitches.filter(p => !pcs.has(p % 12)).forEach(pitch => {
    if (!midiChordOut) {
      chordSynth.triggerRelease(pitchToNote[pitch]);
      logPlayedNote('chord', pitch, false);
    }
    visual.noteOff(pitch);
    visual.scheduleNoteOff(pitch, endFrame, true);
  });
  midiApplyChord(stay, Tone.context.currentTime);
  session.completion = { ...kept, pitches: stay, kept: true };
  const entry = session.completionHistory[session.completionHistory.length - 1];
  if (entry) Object.assign(entry, { pitches: stay, keptFrom: kept.pitches });
}

/** The agent's notes stop; the chord itself stays in the history */
function silenceCompletion() {
  const session = curSession;
  if (!session || !session.completion) return;
  const endFrame = getTransportFrame();
  session.completion.pitches.forEach(pitch => {
    if (!midiChordOut) {
      chordSynth.triggerRelease(pitchToNote[pitch]);
      logPlayedNote('chord', pitch, false);
    }
    visual.noteOff(pitch);
    visual.scheduleNoteOff(pitch, endFrame, true);
  });
  midiStrikeToken++;           // cancel a re-press midiStrikeNow may have pending
  midiApplyChord([], Tone.context.currentTime);
  session.completion = null;
  updateChordTimingInfo();
}

/**
 * The chord ends: its keys were all let go, or a new gesture replaces it.
 * @param {boolean=} refetch - Ask for new candidates (the history changed);
 *   a replacing gesture picks from the current ones instead
 */
function endChord(refetch = true) {
  fillChordTokens();           // the chord held up to now
  const drawn = curSession.heldChord && curSession.heldChord.drawn;
  if (drawn) {
    const endFrame = getTransportFrame();
    drawn.forEach(p => visual.scheduleNoteOff(p, endFrame, true));
  }
  curSession.heldChord = null;
  curSession.chordKeys.clear();
  silenceCompletion();
  if (refetch) fetchCandidates();
  updateChordTimingInfo();
}

/**
 * Sample a chord among the candidates containing every pitch class held (or,
 * if none does, those sharing the most), weighted by the model's
 * probability sharpened or flattened by the Temperature field: p^(1/T).
 * Temperature 0, or greedy (used to recognise a chord the agent won't play),
 * always takes the most probable.
 *
 * With `fit` (complete mode with the agent in, 2026-11-18): each chord's
 * weight is also multiplied by how well the keys, at their pitches, fit its
 * stored voicings the completion could play (voicingFitFor), so the hand's
 * shape counts and only a chord the agent can complete is chosen. If none of
 * the compatible chords has such a voicing, the plain rule applies.
 * @param {boolean=} greedy
 * @param {?Map<string, number>=} fit - symbol -> voicing fit (0-1)
 * @return {?{chord: Object, rank: number, share: number, choices: number,
 *   fit: ?number}}
 *   share: the chord's probability among the compatible chords; choices:
 *   their effective number (onsetCertainty) -- how sure the model was;
 *   fit: the chosen chord's voicing fit if it weighted the choice, else null
 */
function pickCandidate(candidates, held, greedy = false, fit = null) {
  const picked = pickCompatible(candidates, held, greedy, fit);
  if (!picked) return null;
  const certainty = onsetCertainty(picked.compatible.map(c => c.chord));
  return { chord: picked.chord, rank: picked.rank,
    share: certainty ? picked.chord.prob / certainty.total : NaN,
    choices: certainty ? certainty.choices : NaN,
    fit: picked.fitted ? fit.get(picked.chord.symbol) : null };
}

function pickCompatible(candidates, held, greedy, fit = null) {
  const need = new Set(held.map(p => p % 12));
  const scored = candidates.map((chord, rank) => ({
    chord, rank, overlap: chord.pitchClasses.filter(pc => need.has(pc)).length,
  }));
  const bestOverlap = Math.max(...scored.map(c => c.overlap));
  const compatible = scored.filter(c => c.overlap === bestOverlap);
  if (!compatible.length) return null;
  // The voicing fit, if any compatible chord can be completed (else ignored)
  const fitOf = c => (fit && fit.get(c.chord.symbol)) || 0;
  const fitting = fit ? compatible.filter(c => fitOf(c) > 0) : [];
  const fitted = fitting.length > 0;
  const pool = fitted ? fitting : compatible;
  const weightOf = (c, temperature) => Math.pow(c.chord.prob, 1 / temperature) * (fitted ? fitOf(c) : 1);
  const temperature = temperatureInput.valueAsNumber;
  if (greedy || !(temperature > 0)) {
    // the most probable (sorted), with the fit: the largest p * fit
    const best = fitted ? pool.reduce((a, b) => (weightOf(b, 1) > weightOf(a, 1) ? b : a)) : pool[0];
    return { ...best, compatible, fitted };
  }
  const weights = pool.map(c => weightOf(c, temperature));
  let r = Math.random() * weights.reduce((a, b) => a + b, 0);
  for (let i = 0; i < pool.length; i++) {
    r -= weights[i];
    if (r <= 0) return { ...pool[i], compatible, fitted };
  }
  return { ...pool[pool.length - 1], compatible, fitted };
}


/** Space let go in triggered mode: the chord stops */
function releaseTriggeredChord(source = 'space') {
  if (!curSession.triggeredChord) return;
  curSession.triggeredChord.forEach(pitch => {
    if (!midiChordOut) {
      chordSynth.triggerRelease(pitchToNote[pitch]);
      logPlayedNote('chord', pitch, false);
    }
    visual.noteOff(pitch);
  });
  midiStrikeToken++;  // cancel a re-press midiStrikeNow may still have pending
  midiApplyChord([], Tone.context.currentTime);
  curSession.triggeredChord = null;
  curSession.triggerHistory.push(
    { frame: getSessionCurrentFrame(), symbol: '', pitches: [], release: true, source });
  updateChordTimingInfo();
}

/** Show what is sounding and what space will play next */
function updateChordTimingInfo() {
  if (chordTimingSelect.value === 'complete' || chordTimingSelect.value === 'input') {
    if (!curSession) {
      chordTimingInfo.textContent =
        'Start a session, then play a few notes: the model completes the chord';
      return;
    }
    if (curSession.candidateError && !curSession.candidates) {
      chordTimingInfo.textContent =
        `No chord candidates from the server (${curSession.candidateError}); ` +
        'check the server log';
      return;
    }
    const gate = isInputMode() ? 'Input' : `Agent ${isAgentIn() ? 'IN' : 'out'}`;
    const lastGesture = curSession.gestureHistory[curSession.gestureHistory.length - 1];
    const read = lastGesture ? `   Last read as ${lastGesture.kind}` +
      (lastGesture.melody ? ` + melody ${pitchToNote[lastGesture.melody]}` : '') : '';
    const held = curSession.heldChord;
    let chord;
    if (held) {
      const pcName = pc => pitchToNote[60 + pc].replace(/\d+$/, '');
      const played = (held.drawn || []).map(p => pcName(p % 12))
        .filter((n, i, a) => a.indexOf(n) === i).join(' ');
      const how = held.match === 'exact' ? 'Recognised' : 'Inferred';
      const detail = held.match === 'exact' ? '(exactly your notes)'
        : held.match === 'partial'
          ? `(from ${played}; ${held.missing.map(pcName).join(' ')} not in the chord)`
          : `(from ${played})`;
      const sure = `${formatShare(held.share)}, of ${formatChoices(held.choices)} likely`;
      const agent = curSession.completion && curSession.completion.kept
        ? ', a voice moved: the agent keeps its notes that fit'
        : !curSession.completion ? ', agent silent'
        : !curSession.completion.voicing
          ? ` (model's #${held.rank + 1} choice: ${sure}); no stored voicing contains your keys, nothing added`
          : `, completed (model's #${held.rank + 1} choice: ${sure} among the chords that fit; ` +
            `voicing 1 of ${curSession.completion.fitting} containing your keys)`;
      chord = `${how}: ${held.symbol} ${detail}${agent}`;
      if (held.free) {
        chord = `Agent's chord: ${held.symbol} (space, no chord notes held; ` +
          `model's #${held.rank + 1} choice: ${sure})`;
      } else if (!isInputMode() && held.music21 && !curSession.completion) {
        chord = `Your notes ${played}: ${held.music21.display} (music21, in the history), agent silent`;
      }
      if (isInputMode()) {
        const exact = held.exact || [];
        const m21 = held.music21;
        chord = `Your notes ${played}: ` +
          (m21 ? `${m21.display} (music21, in the history)` : 'music21: - (model reading in the history)') +
          `   Exact name: ${exact.length ? exact.slice(0, 2).join(' / ') : 'none'}` +
          `   Model reads: ${held.modelSymbol || held.symbol}`;
      }
    } else {
      chord = !curSession.candidates ? 'thinking...'
        : isAgentIn() && !isInputMode()
          ? 'Play a chord (2+ notes at once): the agent completes it; or hold space: it plays its own'
          : 'Play a chord (2+ notes at once)';
    }
    const c = curSession.certainty;
    const now = c ? `   Model now: ${formatChoices(c.choices)} likely chords` +
      ` (top ${c.top} ${formatShare(c.topShare)})` : '';
    chordTimingInfo.textContent = `${gate}   ${chord}${read}${now}`;
    return;
  }
  if (chordTimingSelect.value === 'triggered' || chordTimingSelect.value === 'triggered-history') {
    if (!curSession) {
      chordTimingInfo.textContent =
        'Start a session; hold space or the left pedal to play the chord scheduled at that moment';
      return;
    }
    const last = curSession.triggerHistory[curSession.triggerHistory.length - 1];
    const now = getScheduledChordAt(Math.floor(getTransportFrame()));
    chordTimingInfo.textContent =
      `Playing: ${last && last.symbol || '--'}   Space plays: ${now && now.symbol || '--'}`;
    return;
  }
  chordTimingInfo.textContent =
    'Chords play on the model\'s own predicted timing';
}

/**
 * Get chord pitches being played or held at the given frame
 * @param {number} targetFrame
 * @return {Array<number>!}
 */
function getChordPitchesAtFrame(targetFrame) {
  for (let i = curSession.chordHistory.length - 1; i >= 0; i--) {
    const { pitches, scheduleFrame } = curSession.chordHistory[i];
    if (scheduleFrame <= targetFrame) {
      return pitches;
    }
  }
  return [];
}

/**
 * Schedule to play chord pitches in the future
 * @param {Array<number>!} pitches
 * @param {number} frame - Frame (relative to ToneJS start) to schedule at
 * @param {number} time - ToneJS time to schedule at
 * @param {boolean} on - If the pitches should hit or release
 * @param {number=} alpha - Opacity of incoming chords on grid
 * @param {string=} symbol - Chord symbol to draw next to incoming chord on grid
 * @return {Array<number>!} ToneJS IDs of scheduled events
 */
function scheduleChordPitches(
  pitches, frame, time, on, alpha = 1.0, symbol = '') {
  let eventIDs = [];
  const bassPitch =
    pitches.reduce((lowest, pitch) => Math.min(lowest, pitch), 1000);
  if (on && symbol && pitches.length && !isTriggeredChordMode()) {
    eventIDs.push(Tone.Transport.scheduleOnce(t => logPlayedChord(symbol, pitches, t), time));
  }
  pitches.forEach(pitch => {
    // In triggered mode the schedule only falls into view; space plays it
    if (!isTriggeredChordMode()) {
      eventIDs.push(scheduleNote(pitchToNote[pitch], on, time));
    }
    if (showChordsCheck.checked) {
      if (on) {
        visual.scheduleNoteOn(
          pitch, frame, 'lightBlue', alpha,
          pitch === bassPitch ? symbol : '');
      } else {
        visual.scheduleNoteOff(pitch, frame);
      }
    }
  });
  return eventIDs;
}

/**
 * Process response from chord model
 * @param {Object!} json
 * newChords - New chord predictions starting at frame
 * newChordTokens - Per-frame tokens for new chords
 * introChordTokens - Per-frame tokens for chords at beginning of session
 * frame - Frame new chords start at
 */
function processAgentAction({ newChords, newChordTokens, introChordTokens,
  frame, chordHoldRange, chordOnsetOffset }) {
  // Token ranges, for the triggered variant that rewrites the history
  if (chordHoldRange) {
    curSession.chordHoldRange = chordHoldRange;
    curSession.chordOnsetOffset = chordOnsetOffset;
  }
  const curFrame = getTransportFrame();
  const targetFrame = frame + curSession.startFrame;
  const silenceFrame = getSilenceFrames() + curSession.startFrame;

  // Fill in new chord tokens and fill possible gaps. Frames that passed while
  // the request was in flight keep what actually sounded (the previous plan):
  // the scheduling below skips them, so writing the new plan there would put
  // chords that were never heard into the history the next request is
  // conditioned on (journal/reference/REALJAM_CHORD_TIMING.md, Known gaps 6).
  const firstOpen = Math.max(frame, curFrame - curSession.startFrame);
  for (let i = Math.min(frame, curSession.chordTokens.length);
    i < frame + newChordTokens.length; i++) {
    if (i < frame) {
      curSession.chordTokens[i] = -1;
    } else if (i < firstOpen) {
      if (curSession.chordTokens[i] === undefined) {
        curSession.chordTokens[i] = -1;
      }
    } else {
      curSession.chordTokens[i] = newChordTokens[i - frame];
    }
  }
  // Where the new plan starts on top of sounded frames, a hold of a different
  // chord than the one sounding must become that chord's onset, or the
  // history is not a valid token sequence
  if (firstOpen > frame && firstOpen < frame + newChordTokens.length &&
    chordHoldRange) {
    const isHold = t => t >= chordHoldRange[0] && t <= chordHoldRange[1];
    const chordOf = t => isHold(t) ? t :
      (t - chordOnsetOffset >= chordHoldRange[0] &&
        t - chordOnsetOffset <= chordHoldRange[1]) ? t - chordOnsetOffset : null;
    const tok = curSession.chordTokens[firstOpen];
    const prev = curSession.chordTokens[firstOpen - 1];
    if (isHold(tok) && chordOf(prev) !== tok) {
      curSession.chordTokens[firstOpen] = tok + chordOnsetOffset;
    }
  }

  // Fill intro chord tokens when model passes them back
  //  (after post-hoc filling of initial silence frames)
  if (introChordTokens) {
    curSession.introSet = true;
    for (let i = 0; i < introChordTokens.length; i++) {
      curSession.chordTokens[i] = introChordTokens[i];
    }
  }

  // Cancel all scheduled events since will be replaced with fresher actions
  // Make sure to not clear before current frame or target frame
  const clearFrame = Math.max(targetFrame, curFrame);
  visual.clearScheduledNotes(clearFrame);
  for (let i = curSession.chordHistory.length - 1; i >= 0; i--) {
    if (curSession.chordHistory[i].scheduleFrame < clearFrame) {
      curSession.chordHistory = curSession.chordHistory.slice(0, i + 1);
      break;
    }
    curSession.chordHistory[i].eventIDs.forEach(
      eventID => Tone.Transport.clear(eventID));
  }

  if (!curSession.voicingDecisions) curSession.voicingDecisions = new Map();
  curSession.voicingDecisions.forEach((d, key) => {
    if (d.frame < curFrame - 64) curSession.voicingDecisions.delete(key);
  });

  // MIDI chord changes for the frames being re-planned. Each poll re-plans
  // every future frame, usually identically; MIDI events go out early, so
  // cancelling and rescheduling an unchanged chord could double or delay it.
  // Keep the existing event when the plan for its frame hasn't changed.
  const midiReplanned = new Map();
  midiScheduledByFrame.forEach((entry, frame) => {
    if (frame >= clearFrame) {
      midiReplanned.set(frame, entry);
    }
    if (frame >= clearFrame || frame < curFrame - 64) {
      midiScheduledByFrame.delete(frame);
    }
  });

  // Schedule note hits and releases for sent frames
  newChords.forEach(([symbol, pitches, on], frameOffset) => {
    const scheduleFrame = targetFrame + frameOffset;
    const time = frameToTransportTime(scheduleFrame);
    console.log(
      'Play', symbol, on, 'at frame', scheduleFrame, ', cur frame', curFrame);

    // Don't schedule/record chords for frames that already happened
    // Otherwise could cause issues with keeping track of currently held pitches
    // Also don't schedule until after silence frames
    if (scheduleFrame < curFrame || scheduleFrame < silenceFrame) {
      return;
    }
    if (customVoicingsCheck.checked) {
      pitches = databaseVoicing(scheduleFrame, symbol, on, pitches);
    }

    const isCommitted = frameOffset < getCommitaheadFrames();
    const prevFramePitches = getChordPitchesAtFrame(scheduleFrame - 1);
    // Custom voicings are chosen per frame (they follow the melody), so a
    // hold of the same chord can come back with different pitches. That's
    // not a change of mind -- keep the voicing the chord started with.
    const prevEntry = getScheduledChordAt(scheduleFrame - 1);
    const sameChordHeld = !on && prevEntry && prevEntry.symbol === symbol;

    // Onsets - release previous chord and play new chord (or rest)
    // Holds - release previous and play chord if different than previous
    //  (indicates model changed its mind about what chord to play)
    if (on || (!sameChordHeld && !arraysEqual(pitches, prevFramePitches))) {
      const offEventIDs =
        scheduleChordPitches(prevFramePitches, scheduleFrame, time, false);
      const onEventIDs = scheduleChordPitches(
        pitches, scheduleFrame, time, true, isCommitted ? 1.0 : 0.5, symbol);
      curSession.chordHistory.push({
        scheduleFrame,
        pitches,
        symbol,
        eventIDs: offEventIDs.concat(onEventIDs)
      });

      if (midiChordOut && !isTriggeredChordMode()) {
        const key = `${on}|${pitches.join(',')}`;
        const prior = midiReplanned.get(scheduleFrame);
        if (prior && prior.key === key) {
          midiReplanned.delete(scheduleFrame);
          midiScheduledByFrame.set(scheduleFrame, prior);
        } else {
          // Latest earlier change on the piano, i.e. when the keys about to
          // be re-struck were last pressed
          const prevFrame = [...midiScheduledByFrame.keys()]
            .filter(f => f < scheduleFrame)
            .reduce((a, b) => Math.max(a, b), -Infinity);
          midiScheduledByFrame.set(scheduleFrame, {
            key, ids: scheduleMidiChord(pitches, scheduleFrame, on,
              prevFrame === -Infinity ? undefined : prevFrame)
          });
        }
      }
    }
  });

  // Whatever wasn't re-planned identically is no longer part of the plan
  midiReplanned.forEach(({ ids }) => ids.forEach(id => Tone.Transport.clear(id)));

  // The latest planned voicing, sent back as prevVoicing next poll for
  // voice-leading continuity across separate /play calls. Taken from what
  // is actually scheduled, so a re-voiced hold that was kept out above
  // doesn't become the reference.
  for (let i = curSession.chordHistory.length - 1; i >= 0; i--) {
    const { pitches } = curSession.chordHistory[i];
    if (pitches && pitches.length) {
      curSession.lastVoicing = pitches;
      break;
    }
  }

  if (isTriggeredChordMode()) {
    updateChordTimingInfo();
  }
}

/**
 * Get necessary info from server (model names) and then enable live sessions
 */
function establishServerConnection() {
  Promise.all([
    fetch(`${window.location.origin}/models`).then(response => response.json()),
    fetch(`${window.location.origin}/bass_models`)
      .then(response => response.json()).catch(() => []),
  ]).then(([chordModels, bassModels]) => {
    chordModelNames = chordModels;
    bassModelNames = bassModels;
    refreshModelOptions();
    console.log('Interactive agent is ready!');
    liveSessionBtn.disabled = false;
  });
  loadSongCatalogue();
  loadVoicingBrowserChords();
}

let chordModelNames = [];  // models that generate chords (all modes)
let bassModelNames = [];   // models that generate the melody lane from chords

/**
 * What the song itself plays in Song Playback: 'melody' (the model plays
 * chords), 'chords' (a bass model plays the melody lane), 'both' (no model),
 * or 'melody-only' / 'chords-only' (just that lane, no model).
 * @return {string}
 */
function getGtMode() {
  const checked = document.querySelector('input[name="gt-mode"]:checked');
  return checked ? checked.value : 'melody';
}

/**
 * Fill the model menu with the models that fit: bass models when the song
 * supplies the chords, chord models otherwise. Keeps the selection if it
 * still applies.
 */
function refreshModelOptions() {
  const names = getGtMode() === 'chords' ? bassModelNames : chordModelNames;
  const previous = modelSelect.value;
  modelSelect.innerHTML = '';
  names.forEach(model => {
    const option = document.createElement('option');
    option.append(document.createTextNode(model));
    option.value = model;
    modelSelect.appendChild(option);
  });
  if (!names.length) {
    const option = document.createElement('option');
    option.append(document.createTextNode('(no bass model available)'));
    option.value = '';
    modelSelect.appendChild(option);
  }
  if (names.includes(previous)) modelSelect.value = previous;
}

/** Fetch the robot-melody song catalogue for fuzzy searching */
async function loadSongCatalogue() {
  const result = await fetch(`${window.location.origin}/songs`);
  songCatalogue = prepareSongCatalogue(await result.json());
}

/**
 * Precompute the lowercased forms the search scans, so a keystroke doesn't
 * re-lowercase and re-concatenate all 30k songs (that alone was most of the
 * per-keystroke cost).
 * @param {Array<Object>!} songs
 * @return {Array<Object>!} The same entries, annotated in place
 */
function prepareSongCatalogue(songs) {
  songs.forEach(song => {
    song.titleLower = song.title.toLowerCase();
    song.artistLower = song.artist.toLowerCase();
  });
  return songs;
}

// Characters a match can start right after and still count as beginning a
// word: space, hyphen, underscore, slash, open paren, comma, apostrophe.
// Compared by char code so scoring never has to slice out a character.
const wordBoundaryCharCodes = new Set([32, 45, 95, 47, 40, 44, 39]);

// How many query characters the last fuzzyMatchScore call got through before
// giving up. Returned out of band because it's only needed on the rare path
// where a query spans both title and artist, and packing it into the return
// value would allocate for every song scanned.
let fuzzyMatchConsumed = 0;

/**
 * Split search text into the characters a match must contain, in order.
 * Done once per keystroke rather than once per song: iterating the query
 * string directly inside the scorer allocates a character per song scanned,
 * which dominated the cost across a 30k-song catalogue.
 * @param {string} query - Lowercased search text
 * @return {Array<string>!} Characters to match, spaces dropped
 */
function toQueryChars(query) {
  // Spaces are separators between what the user typed, not characters to find
  return query.split('').filter(c => c !== ' ');
}

/**
 * Score how well a fuzzy query matches some text, higher being better.
 * Query characters must appear in order but not contiguously, so "empstate"
 * or "emp state" both find "Empire State Of Mind". Bonuses for word-start
 * and consecutive hits keep the meaningful matches above incidental letter
 * scatter across a long title.
 * @param {Array<string>!} queryChars - From toQueryChars
 * @param {string} lower - Text to score against, already lowercased (see
 *   prepareSongCatalogue -- lowercasing here would dominate the cost)
 * @param {number=} startIndex - First query character to match, so a caller
 *   can match the remainder of a query without slicing the array per song
 * @return {number} Score, or -Infinity if the query isn't a subsequence of
 *   the text. A real match can score negative, so the sentinel has to sit
 *   below any achievable score rather than at some fixed threshold. Also
 *   sets fuzzyMatchConsumed.
 */
function fuzzyMatchScore(queryChars, lower, startIndex = 0) {
  let score = 0;
  let pos = 0;         // Where in the text to resume scanning
  let prevMatch = -2;  // Index of the previously matched character
  for (let i = startIndex; i < queryChars.length; i++) {
    const found = lower.indexOf(queryChars[i], pos);
    if (found === -1) {
      fuzzyMatchConsumed = i;
      return -Infinity;
    }
    score += 1;
    if (found === prevMatch + 1) {
      score += 6;
    }
    if (found === 0 || wordBoundaryCharCodes.has(lower.charCodeAt(found - 1))) {
      score += 10;
    }
    score -= Math.min(found - pos, 4);  // Penalise skipping over characters
    prevMatch = found;
    pos = found + 1;
  }
  fuzzyMatchConsumed = queryChars.length;
  // Among equally good matches, prefer the tighter text
  return score - Math.min(lower.length / 10, 5);
}

/**
 * Best fuzzy score for a song, matching the query against its title, its
 * artist, or both together -- so "creep" finds the song, "radiohead" finds
 * everything by them, and "creep radiohead" finds the one song.
 * @param {Array<string>!} queryChars - From toQueryChars
 * @param {Object!} song - Catalogue entry with title and artist
 * @return {number} Score, or -Infinity if nothing matched
 */
function songMatchScore(queryChars, song) {
  let best = fuzzyMatchScore(queryChars, song.titleLower);
  const titleConsumed = fuzzyMatchConsumed;

  // Nudge artist hits below title hits, so typing a song name still ranks
  // that song above other work by an artist whose name happens to match
  const artistScore = fuzzyMatchScore(queryChars, song.artistLower);
  if (artistScore > -Infinity) {
    best = Math.max(best, artistScore - 2);
  }

  // Neither field matched the whole query, so try it as "title artist"
  // ("creep radiohead"): the title took a prefix, the artist must take the
  // rest. Scanning only the leftover characters here, rather than matching
  // the query against a combined string, keeps this off the hot path -- it
  // was costing more than the two real passes combined.
  if (best === -Infinity && titleConsumed > 0) {
    best = fuzzyMatchScore(queryChars, song.artistLower, titleConsumed);
  }
  return best;
}

/** Recompute fuzzy matches for the current search text and show them */
function onSongSearchInput() {
  const query = songSearchInput.value.trim().toLowerCase();
  songSearchIndex = 0;
  const bassOnly = bassSongsOnlyCheck && bassSongsOnlyCheck.checked;
  // Walking bass lines only, nothing typed: list all of them to browse
  if (bassOnly && !query) {
    songSearchMatches = songCatalogue.filter(song => song.kind === 'bass');
    songSearchPool = [];
    songSearchLastQuery = '';
    renderSongSearchResults();
    return;
  }
  // A single character matches most of the catalogue, so the results wouldn't
  // mean anything and it's the most expensive scan there is
  if (query.length < songSearchMinChars) {
    songSearchMatches = [];
    songSearchPool = [];
    songSearchLastQuery = '';
    renderSongSearchResults();
    return;
  }

  // Extending a query can only narrow the results -- if a song's text didn't
  // contain the old query as a subsequence, it can't contain a longer one --
  // so typing forward rescores just the previous hits instead of all 30k.
  // Only scoring against the full catalogue when the query isn't an extension
  // (first character, a backspace, or a paste) keeps every keystroke cheap,
  // since the full scan then only happens for short, fast-to-reject queries.
  const pool = (songSearchLastQuery && query.startsWith(songSearchLastQuery))
    ? songSearchPool
    : (bassOnly ? songCatalogue.filter(song => song.kind === 'bass') : songCatalogue);

  // Collect every match, but only ever rank the handful actually shown. A
  // short query matches most of the catalogue, and sorting tens of thousands
  // of those per keystroke costs far more than the matching itself. The pool
  // stays in catalogue order so equal scores always break ties the same way.
  const queryChars = toQueryChars(query);
  const nextPool = [];
  const top = [];
  let worstShown = -Infinity;
  pool.forEach(song => {
    const score = songMatchScore(queryChars, song);
    if (score === -Infinity) {
      return;
    }
    nextPool.push(song);
    if (top.length === songSearchMaxResults && score <= worstShown) {
      return;
    }
    let i = top.length;
    while (i > 0 && top[i - 1].score < score) {
      i--;
    }
    top.splice(i, 0, { song, score });
    if (top.length > songSearchMaxResults) {
      top.pop();
    }
    worstShown = top[top.length - 1].score;
  });

  songSearchPool = nextPool;
  songSearchLastQuery = query;
  songSearchMatches = top.map(entry => entry.song);
  renderSongSearchResults();
}

const SONG_KIND_LABELS = { bass: 'bass line', demo: 'demo' };
/**
 * If the line the chords accompany is a bass line rather than a melody,
 * judged from the line itself (not from the model: a bass-input model may
 * still be played a tune): a walking-bass song is playing; a melody song is
 * not; when the performer plays, a note below BASS_LINE_BELOW is a bass line.
 * @return {boolean}
 */
function inputIsBassLine() {
  if (curSession && curSession.songKind) return curSession.songKind === 'bass';
  return !!curSession && currentMelodyPitch() < BASS_LINE_BELOW;
}
const BASS_LINE_BELOW = 55;  // G3

/** Draw the current match list, marking the keyboard-highlighted row */
function renderSongSearchResults() {
  songSearchResults.innerHTML = '';
  songSearchMatches.forEach((song, i) => {
    const row = document.createElement('div');
    row.className = i === songSearchIndex ? 'search-result active' : 'search-result';
    row.textContent = song.title;
    const artist = document.createElement('span');
    artist.className = 'artist';
    artist.textContent = `  ${song.artist}`;
    row.appendChild(artist);
    if (song.kind && song.kind !== 'melody') {
      // A walking bass line (WJD, FiloBass) or a calibration demo, not a tune
      const badge = document.createElement('span');
      badge.className = `kind-badge ${song.kind}`;
      badge.textContent = SONG_KIND_LABELS[song.kind] || song.kind;
      badge.title = song.kind === 'bass'
        ? 'Its "melody" is a walking bass line (jazz bass, low register), not a tune'
        : 'Synthetic calibration song for the Disklavier';
      row.appendChild(badge);
    }
    row.addEventListener('click', () => playSongFromSearch(i));
    songSearchResults.appendChild(row);
    if (i === songSearchIndex && row.scrollIntoView) {
      row.scrollIntoView({ block: 'nearest' });  // keep the arrow-key pick in view
    }
  });
}

/** Arrow keys move the highlight, Enter plays it, Escape clears the search */
function onSongSearchKeydown(event) {
  if (event.key === 'Escape') {
    songSearchInput.value = '';
    onSongSearchInput();
    return;
  }
  if (!songSearchMatches.length) {
    return;
  }
  if (event.key === 'ArrowDown') {
    event.preventDefault();
    songSearchIndex = Math.min(songSearchIndex + 1, songSearchMatches.length - 1);
    renderSongSearchResults();
  } else if (event.key === 'ArrowUp') {
    event.preventDefault();
    songSearchIndex = Math.max(songSearchIndex - 1, 0);
    renderSongSearchResults();
  } else if (event.key === 'Enter') {
    event.preventDefault();
    playSongFromSearch(songSearchIndex);
  }
}

/**
 * Fetch the chosen song's melody and start robot playback.
 * @param {number} index - Row in songSearchMatches
 */
async function playSongFromSearch(index) {
  const song = songSearchMatches[index];
  if (!song) {
    return;
  }
  songSearchIndex = index;
  renderSongSearchResults();
  songSearchInput.blur();  // Hand the keyboard back to playing

  const base = `${window.location.origin}/songs/` +
    `${song.dataset}/${song.split}/${encodeURIComponent(song.id)}`;

  const gtMode = getGtMode();
  // The song's chords come as block chords; with Database voicings on they
  // are voiced here by the same rule as the model's (voiceReferenceChords).
  // A demo song's chords are exact keys and stay as they are.
  const voiced = (chords, melody, bassLine) =>
    customVoicingsCheck.checked && song.kind !== 'demo'
      ? voiceReferenceChords(chords, melody, bassLine) : chords;
  if (gtMode === 'chords') {
    // The song's chords; a bass model plays the melody lane (a walking bass)
    const result = await fetch(`${base}/reference`);
    const { melody, chords, num_beats } = await result.json();
    await playSongWithBass(song, melody, await voiced(chords, [], true), num_beats);
    return;
  }
  if (gtMode === 'both' || gtMode === 'melody-only' || gtMode === 'chords-only') {
    // The song's own material, no model involved: both lanes, or just one
    const result = await fetch(`${base}/reference`);
    const { melody, chords, num_beats } = await result.json();
    const playMelody = gtMode === 'chords-only' ? [] : melody;
    const playChords = gtMode === 'melody-only' ? [] :
      await voiced(chords, playMelody, song.kind === 'bass' && playMelody.length > 0);
    playSongReference(playMelody, playChords, num_beats);
    if (curSession) curSession.songKind = song.kind || 'melody';
    return;
  }

  const result = await fetch(`${base}/melody`);
  const { melody, num_beats } = await result.json();
  playSongMelody(melody, num_beats);
  // On the session it started, so it ends with it
  if (curSession) curSession.songKind = song.kind || 'melody';
}

/**
 * Voicing browser: audition individual stored voicings for a chord symbol,
 * completely independent of any live/robot session -- no noteHistory, no
 * Transport scheduling, no server /play calls. Select a chord, then use
 * Left/Right arrow keys to step through its stored voicings; each plays for
 * exactly one measure at the current tempo/time signature.
 */

/** Fetch every chord symbol covered by the voicing lookup and populate the dropdown */
async function loadVoicingBrowserChords() {
  const result = await fetch(`${window.location.origin}/voicings/chords`);
  const chords = await result.json();
  chords.forEach(chord => {
    const option = document.createElement('option');
    option.value = chord;
    option.textContent = chord;
    voicingBrowserChordSelect.appendChild(option);
  });
}

/** Handle chord selection: fetch its voicings and audition the first one */
async function onVoicingBrowserChordSelected() {
  const chord = voicingBrowserChordSelect.value;
  if (!chord) {
    voicingBrowserAll = [];
    voicingBrowserList = [];
    voicingBrowserIndex = -1;
    return;
  }
  const result = await fetch(
    `${window.location.origin}/voicings?chord=${encodeURIComponent(chord)}`);
  voicingBrowserAll = await result.json();
  applyVoicingBrowserFilter(true);
}

/**
 * The browser's filters (2026-11-19): number of notes, lowest note at least,
 * highest note at most. The list is narrowed to the voicings passing them
 * and starts again at the first; it plays only when a chord was just chosen
 * (`play`), never while a filter is being changed (a slider drag would
 * strike the piano over and over).
 * @param {boolean} play
 */
function applyVoicingBrowserFilter(play) {
  const notes = parseInt(voicingBrowserNotesSelect.value, 10);
  const low = parseInt(voicingBrowserLowInput.value, 10);
  const high = parseInt(voicingBrowserHighInput.value, 10);
  voicingBrowserList = voicingBrowserAll.filter(v => (!notes || v.pitches.length === notes) &&
    Math.min(...v.pitches) >= low && Math.max(...v.pitches) <= high);
  voicingBrowserIndex = voicingBrowserList.length ? 0 : -1;
  if (play && voicingBrowserList.length) {
    playBrowsedVoicing();
  } else if (voicingBrowserChordSelect.value) {
    voicingBrowserInfo.textContent = voicingBrowserList.length
      ? `${voicingBrowserList.length} of ${voicingBrowserAll.length} voicings match -- ` +
        'use \u2190/\u2192 to play them'
      : `No stored voicing of ${voicingBrowserChordSelect.value} matches ` +
        `(${voicingBrowserAll.length} stored)`;
  }
}

/** A filter changed: show the slider values (low never above high), then filter */
function onVoicingBrowserFilterChanged(event) {
  let low = parseInt(voicingBrowserLowInput.value, 10);
  let high = parseInt(voicingBrowserHighInput.value, 10);
  if (low > high) {
    if (event && event.target === voicingBrowserLowInput) voicingBrowserHighInput.value = high = low;
    else voicingBrowserLowInput.value = low = high;
  }
  voicingBrowserLowValue.textContent = pitchToNote[low];
  voicingBrowserHighValue.textContent = pitchToNote[high];
  applyVoicingBrowserFilter(false);
}

/**
 * Left/Right arrow keys on the browser's drop-downs (chord, notes in voicing)
 * step through the voicings. They never change the drop-down itself, also
 * when no voicing matches (2026-11-19: they used to switch to the next chord
 * then, and changed the notes filter when it had focus). Up/Down still
 * choose the chord.
 */
function onVoicingBrowserKeydown(event) {
  if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return;
  event.preventDefault();
  if (!voicingBrowserList.length) return;
  voicingBrowserIndex = event.key === 'ArrowRight'
    ? Math.min(voicingBrowserIndex + 1, voicingBrowserList.length - 1)
    : Math.max(voicingBrowserIndex - 1, 0);
  playBrowsedVoicing();
}

/** After a filter change, back to the chord menu: arrows then browse again */
function refocusVoicingBrowser() {
  if (voicingBrowserChordSelect && voicingBrowserChordSelect.focus) voicingBrowserChordSelect.focus();
}

/**
 * Play the currently-selected browsed voicing for one measure, and update
 * the info readout. Uses chordSynth directly (no Transport scheduling, no
 * session) so this works whether or not a live/robot session is running.
 */
function playBrowsedVoicing() {
  const entry = voicingBrowserList[voicingBrowserIndex];
  if (!entry) {
    return;
  }

  // Cut off whatever the previous voicing was doing immediately, rather than
  // letting it ring/stay highlighted until its own bar finishes -- otherwise
  // rapidly stepping through voicings overlaps with the new one.
  if (voicingBrowserOffTimeout !== null) {
    clearTimeout(voicingBrowserOffTimeout);
    voicingBrowserOffTimeout = null;
  }
  if (voicingBrowserActivePitches.length) {
    if (!midiChordOut) {
      chordSynth.triggerRelease(voicingBrowserActivePitches.map(pitch => pitchToNote[pitch]));
    }
    voicingBrowserActivePitches.forEach(pitch => visual.noteOff(pitch));
    voicingBrowserActivePitches = [];
  }

  const beatsPerMeasure = Tone.Transport.timeSignature;
  const barSeconds = (60 / Tone.Transport.bpm.value) * beatsPerMeasure;
  const noteNames = entry.pitches.map(pitch => pitchToNote[pitch]);
  if (midiChordOut) {
    // Replaces the previous voicing on the piano, re-pressing shared keys
    midiStrikeNow(entry.pitches);
  } else {
    chordSynth.triggerAttackRelease(noteNames, barSeconds);
  }

  // Highlight on the piano roll for the same duration as the audio -- plain
  // setTimeout (not Tone.Transport) since this tool runs outside any
  // session/Transport scheduling.
  entry.pitches.forEach(pitch => visual.noteOn(pitch));
  voicingBrowserActivePitches = entry.pitches;
  voicingBrowserOffTimeout = setTimeout(() => {
    entry.pitches.forEach(pitch => visual.noteOff(pitch));
    midiApplyChord([], Tone.context.currentTime);
    voicingBrowserActivePitches = [];
    voicingBrowserOffTimeout = null;
  }, barSeconds * 1000);

  const detail = entry.legacy
    ? '(ReaLchords default fixed-octave voicing)'
    : `(seen ${entry.count} times across ${entry.num_songs} songs)`;
  const filtered = voicingBrowserList.length < voicingBrowserAll.length
    ? ` (of ${voicingBrowserAll.length} stored)` : '';
  voicingBrowserInfo.textContent =
    `Voicing ${voicingBrowserIndex + 1} / ${voicingBrowserList.length}${filtered} -- ` +
    `${noteNames.join(', ')} ${detail}`;
}


/**
 * Play melody note, add to history and start live session loop if needed
 * @param {string} note
 * @param {number} velocity
 */
function playNote(note, velocity, fromLaptop = false) {
  const keyIndex = noteToPitch[note];
  const completing = curSession && isCompleteMode();
  if (curSession && !completing) {
    curSession.noteHistory.push(
      { on: true, pitch: keyIndex, frame: performerFrame() });
    startGenerationLoop();
  }
  // With a MIDI output the piano sounds the melody. Notes played on the
  // laptop (keys or mouse) are pressed on the piano; notes coming from a MIDI
  // input are not sent back, since on a Disklavier that input *is* the piano
  // and the key is already down.
  logPlayedNote('performer', keyIndex, true, velocity);
  if (!midiChordOut) {
    melodySynth.triggerAttack(note, '+0', velocity);
  } else if (fromLaptop) {
    midiKeyDown(keyIndex, 'laptop', midiMelodyVelocity(), Tone.context.currentTime);
  }
  visual.noteOn(keyIndex);
  if (completing) onCompletionNoteDown(keyIndex);
}

/**
 * Release melody note and add to history
 * @param {string} note
 */
function releaseNote(note, fromLaptop = false) {
  const keyIndex = noteToPitch[note];
  const completing = curSession && isCompleteMode();
  if (curSession && !completing) {
    curSession.noteHistory.push(
      { on: false, pitch: keyIndex, frame: performerFrame() });
  }
  logPlayedNote('performer', keyIndex, false);
  if (!midiChordOut) {
    melodySynth.triggerRelease(note);
  } else if (fromLaptop) {
    midiKeyUp(keyIndex, 'laptop', Tone.context.currentTime);
  }
  visual.noteOff(keyIndex);
  if (completing) onCompletionNoteUp(keyIndex);
}

// ---------------------------------------------------------------------------
// Session log of what actually sounds (2026-11-16): every note as it goes out
// to the laptop synth or the piano (curSession.playedNotes) and every chord
// with its symbol and voiced pitches as it sounds (curSession.playedChords),
// in every mode, song playback included. Times are Transport seconds and
// session frames (16ths from the session's frame 0). With a MIDI output,
// scheduled notes are sent Send Early ahead of the beat they belong to and
// logged when sent; the frame is the beat they sound on.
// ---------------------------------------------------------------------------

/** Transport seconds at an audio-context time (now if omitted) */
function transportSecondsAt(audioTime) {
  const now = Tone.context.currentTime;
  return Tone.Transport.seconds + ((audioTime === undefined ? now : audioTime) - now);
}

/** Session frame at Transport seconds */
function sessionFrameAt(seconds) {
  return seconds / Tone.Time('16n').toSeconds() - (curSession.startFrame || 0);
}

/**
 * Log one note going out.
 * @param {string} part - 'performer' (your playing), 'melody' (the song's
 *   melody or the bass model's line), 'chord' (model, song or completion)
 * @param {number} pitch
 * @param {boolean} on
 * @param {number=} velocity - MIDI 1-127, or 0-1 (laptop synth), stored as 1-127
 * @param {number=} audioTime - When it goes out (audio-context time); now if omitted
 * @param {number=} lead - Seconds it is sent ahead of its beat (MIDI Send Early)
 */
function logPlayedNote(part, pitch, on, velocity, audioTime, lead = 0) {
  if (!curSession || !curSession.playedNotes) return;
  const t = transportSecondsAt(audioTime);
  const vel = typeof velocity === 'number'
    ? Math.round(velocity <= 1 ? velocity * 127 : velocity) : null;
  curSession.playedNotes.push({
    t: Math.round(t * 1000) / 1000,
    frame: Math.round(sessionFrameAt(t + lead) * 100) / 100,
    pitch, on, velocity: on ? vel : 0, part,
    out: midiChordOut ? 'midi' : 'synth',
  });
}

/** Log an incoming pedal value (CC 64 damper, 66 sostenuto, 67 soft) */
function logPedal(cc, value) {
  if (!curSession || !curSession.pedalLog || ![64, 66, 67].includes(cc)) return;
  const t = transportSecondsAt();
  curSession.pedalLog.push({ t: Math.round(t * 1000) / 1000,
    frame: Math.round(sessionFrameAt(t) * 100) / 100, cc, value });
}

/** Log a chord as it sounds: its symbol and the pitches it is voiced with */
function logPlayedChord(symbol, pitches, audioTime) {
  if (!curSession || !curSession.playedChords || !pitches || !pitches.length) return;
  const t = transportSecondsAt(audioTime);
  curSession.playedChords.push({
    t: Math.round(t * 1000) / 1000,
    frame: Math.round(sessionFrameAt(t) * 100) / 100,
    symbol, pitches: [...pitches].sort((a, b) => a - b),
  });
}

/**
 * Schedule note to play in the future
 * @param {string} note
 * @param {boolean} on - If the note is a hit or release
 * @param {string} playTime - ToneJS time for scheduled event
 * @return {number} ID of scheduled ToneJS event
 */
function scheduleNote(note, on, playTime) {
  const keyIndex = noteToPitch[note];
  let eventID;
  // With a MIDI chord output the piano makes the sound (see the MIDI chord
  // output section below); these events then only drive the piano roll.
  if (on) {
    eventID = Tone.Transport.scheduleOnce(time => {
      if (!midiChordOut) {
        chordSynth.triggerAttack(note, '+0', chordVelocity);
        logPlayedNote('chord', keyIndex, true, chordVelocity, time);
      }
      Tone.Draw.schedule(() => {
        visual.noteOn(keyIndex, 'blue');
      }, time);
    }, playTime);
  } else {
    eventID = Tone.Transport.scheduleOnce(time => {
      if (!midiChordOut) {
        chordSynth.triggerRelease(note);
        logPlayedNote('chord', keyIndex, false, 0, time);
      }
      Tone.Draw.schedule(() => {
        visual.noteOff(keyIndex);
      }, time);
    }, playTime);
  }
  return eventID;
}

/*
 * MIDI chord output (e.g. a Disklavier)
 *
 * The browser-side part of what Aria-Duet does in demo_mlx.py's stream_midi:
 * the piano's own MIDI IN Delay buffer (~500 ms) is meant to be switched off,
 * and its mechanical latency is covered here by sending every chord change
 * `midiLeadMs` early. All chords use one velocity (the MIDI Chord Velocity
 * control), so one lead suffices -- but softer notes sound later, so the lead
 * has to be re-tuned when the velocity changes. Note-ons and note-offs share
 * the lead, which keeps their order intact.
 *
 * Exactly one chord sounds at a time, so the piano is driven as a state:
 * midiApplyChord releases whatever isn't in the new chord and attacks what
 * isn't sounding yet. Pitches shared by consecutive chords are held rather
 * than released and re-pressed in the same instant, which the action can't
 * do; resending the same chord is a no-op; and a re-planned chord can never
 * leave notes stuck. A chord *onset* is a genuine re-strike, so its shared
 * pitches are lifted early (midiRestrikeGapFor) to let the action reset.
 */
let chordOutputSelect, midiLeadInput, midiVelocityInput, midiMelodyVelocityInput;
let midiChordOut = null;               // WebMidi Output; null = laptop synth
const midiSounding = new Set();        // pitches currently held on the piano
let midiScheduledByFrame = new Map();  // frame -> {key, ids}, see processAgentAction
/**
 * How long a key is released before it is struck again, by how far apart
 * the two strikes are. Calibrated per note length on the ENSPIRE U1 at
 * 80 BPM with the disklavier_fifths-sweep demos: the hold is whatever is
 * left of the note (spacing - release). A long hold lets the hammer settle
 * on the backcheck, which then needs a long release to reset; after a short
 * hold it is still rebounding and resets quickly. Anchors are in ms (the
 * mechanics don't care about tempo); spacings between them are blended
 * linearly, beyond them the nearest value is used.
 */
// Calibrated 2026-10-05: piano volume 2.5/5, velocity 40 (chords and melody),
// 80 BPM; hold = spacing - release: 117.5 / 225 / 450 / 1200 ms.
const MIDI_RELEASE_ANCHORS = [
  { id: 'midi-release-16-input', spacingMs: 187.5, fallback: 70 },   // 1/16 at 80 BPM
  { id: 'midi-release-8-input', spacingMs: 375, fallback: 150 },     // 1/8
  { id: 'midi-release-4-input', spacingMs: 750, fallback: 300 },     // 1/4
  { id: 'midi-release-2-input', spacingMs: 1500, fallback: 300 },    // 1/2 and longer
];

/** Release (ms) set for anchor i */
function midiReleaseMs(i) {
  const anchor = MIDI_RELEASE_ANCHORS[i];
  const ms = anchor.input ? anchor.input.valueAsNumber : NaN;
  return Number.isNaN(ms) ? anchor.fallback : Math.max(0, ms);
}

/** Release (ms) after a long hold, for strikes whose spacing is unknown */
function midiRestrikeGapMs() {
  return midiReleaseMs(MIDI_RELEASE_ANCHORS.length - 1);
}

/**
 * Release time before re-striking a key that was pressed `spacingSec`
 * earlier, from the release table. Never longer than the spacing itself
 * (the note would get no hold at all).
 * @param {number=} spacingSec - Time since the key was last pressed;
 *   undefined when unknown (the long-hold value is used)
 * @return {number} Seconds
 */
function midiRestrikeGapFor(spacingSec) {
  if (spacingSec === undefined) return midiRestrikeGapMs() / 1000;
  const ms = spacingSec * 1000;
  const anchors = MIDI_RELEASE_ANCHORS;
  let release;
  if (ms <= anchors[0].spacingMs) {
    release = midiReleaseMs(0);
  } else if (ms >= anchors[anchors.length - 1].spacingMs) {
    release = midiReleaseMs(anchors.length - 1);
  } else {
    const i = anchors.findIndex(a => a.spacingMs >= ms);
    const lo = anchors[i - 1], hi = anchors[i];
    const t = (ms - lo.spacingMs) / (hi.spacingMs - lo.spacingMs);
    release = midiReleaseMs(i - 1) + t * (midiReleaseMs(i) - midiReleaseMs(i - 1));
  }
  return Math.min(release, ms) / 1000;
}
const MIDI_STALE_MS = 50;              // later than this, a note-on is skipped

/**
 * Turn an audio-clock event time into a Web MIDI timestamp, so the message
 * goes out exactly then even if the callback ran a little early.
 * @param {number} audioTime - AudioContext time from a Transport callback
 * @return {number} performance.now()-based timestamp
 */
function midiTimestamp(audioTime) {
  const aheadMs = (audioTime - Tone.context.currentTime) * 1000;
  return performance.now() + Math.max(0, aheadMs);
}

/** Chord velocity on the piano, i.e. its loudness (1-127; 0 would be a note-off). */
function midiVelocity() {
  return Math.min(127, Math.max(1, Math.round(midiVelocityInput.valueAsNumber || 40)));
}

/** Robot-melody velocity on the piano (1-127). */
function midiMelodyVelocity() {
  return Math.min(127, Math.max(1, Math.round(midiMelodyVelocityInput.valueAsNumber || 40)));
}

/*
 * Key arbiter. Chords and the robot melody play the same 88 keys, so each
 * key records who is holding it: it goes down for its first holder and comes
 * up only once nobody holds it. A chord change can then never cut off a
 * melody note on a shared key, and vice versa. A note that lands on a key
 * already held by the other part keeps sounding rather than being re-struck.
 */
const midiKeyHolders = new Map();  // pitch -> Set of 'chord' | 'melody'

function midiKeyDown(pitch, holder, velocity, audioTime) {
  const holders = midiKeyHolders.get(pitch) || new Set();
  if (holders.size === 0) {
    midiChordOut.send([0x90, pitch, velocity], { time: midiTimestamp(audioTime) });
    if (holder !== 'laptop') {  // the performer's laptop keys: logged in playNote
      logPlayedNote(holder === 'chord' ? 'chord' : 'melody', pitch, true, velocity, audioTime,
        (midiLeadInput.valueAsNumber || 0) / 1000);
    }
  }
  holders.add(holder);
  midiKeyHolders.set(pitch, holders);
}

function midiKeyUp(pitch, holder, audioTime) {
  const holders = midiKeyHolders.get(pitch);
  if (!holders || !holders.delete(holder)) return;
  if (holders.size === 0) {
    midiChordOut.send([0x80, pitch, 0], { time: midiTimestamp(audioTime) });
    if (holder !== 'laptop') {
      logPlayedNote(holder === 'chord' ? 'chord' : 'melody', pitch, false, 0, audioTime,
        (midiLeadInput.valueAsNumber || 0) / 1000);
    }
    midiKeyHolders.delete(pitch);
    midiReleasedAt.set(pitch, audioTime);
  }
}
/** pitch -> audio time its key was last released, for midiStrikeNow */
const midiReleasedAt = new Map();

function midiNoteOn(pitch, audioTime) {
  midiKeyDown(pitch, 'chord', midiVelocity(), audioTime);
  midiSounding.add(pitch);
}

function midiNoteOff(pitch, audioTime) {
  midiKeyUp(pitch, 'chord', audioTime);
  midiSounding.delete(pitch);
}

/**
 * Schedule the robot melody on the piano for one loop pass.
 *
 * A note is pressed `lead` early. Its release is worked out when it is
 * pressed, not when the pass is scheduled, so the MIDI release settings
 * and the melody velocity can be changed while a loop plays: where the same
 * pitch comes back, the release is pulled earlier by midiRestrikeGapFor
 * (like Aria's _adjust_previous_off_time, never before the note's own
 * onset). The next pass's notes count as "coming back" too (`loopFrames`),
 * so a loop restarting on the same pitch gets its gap like any other
 * repeat. Cleared with everything else by Tone.Transport.cancel() when the
 * session stops.
 * @param {Array<{pitch: number, onset: number, offset: number}>} notes
 * @param {number} baseFrame - Transport frame the loop starts at
 * @param {number=} loopFrames - Loop period, if the notes repeat
 * @param {boolean=} redraw - Move a note's drawn release if the settings
 *   changed after it was drawn (ground-truth playback draws the melody)
 * @param {Array<Object>=} following - Notes known to come next but not
 *   scheduled by this call (same format, same baseFrame), so a repeat of the
 *   same key across calls still gets its release
 * @return {?Array<{pitch: number, offFrame: number}>} each note's release
 *   with the current settings, for drawing (null without a MIDI output)
 */
function scheduleMidiMelody(notes, baseFrame, loopFrames, redraw = false, following = []) {
  if (!midiChordOut) return null;
  const lead = (midiLeadInput.valueAsNumber || 0) / 1000;
  const now = Tone.Transport.seconds;
  const spf = Tone.Time('16n').toSeconds();
  const toSec = beats => Tone.Time(frameToTransportTime(
    baseFrame + Math.round(beats * fpb))).toSeconds();

  const timed = notes
    .map(({ pitch, onset, offset }) => ({ pitch, on: toSec(onset), off: toSec(offset) }))
    .sort((a, b) => a.on - b.on);
  const later = following.map(({ pitch, onset, offset }) =>
    ({ pitch, on: toSec(onset), off: toSec(offset) }));
  const upcoming = loopFrames
    ? [...timed, ...timed.map(n => ({ ...n, on: n.on + loopFrames * spf })), ...later]
    : [...timed, ...later];

  /** When `note` is released under the current settings, in seconds */
  const releaseOf = note => {
    const next = upcoming.find(n => n.pitch === note.pitch && n.on > note.on);
    if (!next) return note.off;
    const gap = midiRestrikeGapFor(next.on - note.on);
    return next.on - note.off < gap ? Math.max(note.on, next.on - gap) : note.off;
  };

  const releases = [];
  timed.forEach(note => {
    const { pitch, on } = note;
    const drawnOff = releaseOf(note);
    releases.push({ pitch, offFrame: drawnOff / spf });
    const onAt = on - lead;
    if (onAt < now - MIDI_STALE_MS / 1000) return;  // too late to sound on time
    Tone.Transport.scheduleOnce(t => {
      midiKeyDown(pitch, 'melody', midiMelodyVelocity(), t);
      const off = releaseOf(note);
      if (redraw && off !== drawnOff) visual.scheduleNoteOff(pitch, off / spf, true);
      // A hair early, so a release at the same moment as the next press of
      // this key (gap 0) is still sent before it
      // (Transport time, not `t`: callbacks get audio-context time, whose
      // clock has been running since the page loaded.)
      Tone.Transport.scheduleOnce(t2 => midiKeyUp(pitch, 'melody', t2),
        Math.max(off - lead - 0.0005, Tone.Transport.seconds + 0.0005));
    }, Math.max(onAt, now + 0.002));
  });
  return releases;
}

/** Move the piano from whatever it's holding to exactly `pitches`. */
function midiApplyChord(pitches, audioTime) {
  if (!midiChordOut) return;
  const target = new Set(pitches);
  [...midiSounding].forEach(p => { if (!target.has(p)) midiNoteOff(p, audioTime); });
  target.forEach(p => { if (!midiSounding.has(p)) midiNoteOn(p, audioTime); });
}

/** Lift any of `pitches` currently held, ahead of a re-strike. */
function midiLift(pitches, audioTime) {
  if (!midiChordOut) return [];
  const lifted = pitches.filter(p => midiSounding.has(p));
  lifted.forEach(p => midiNoteOff(p, audioTime));
  return lifted;
}

/**
 * Schedule a re-strike: lift the keys of `pitches` still held at `liftAt`,
 * press the chord at `sendAt`. When the lift came too late to leave the full
 * release before `sendAt` (a chord decided at the last moment), only the
 * keys actually lifted wait out their release; the rest of the chord still
 * sounds on time.
 * @return {Array<number>} Transport event IDs
 */
function scheduleMidiRestrike(pitches, liftAt, sendAt, gap) {
  let lifted = [];
  const ids = [Tone.Transport.scheduleOnce(t => { lifted = midiLift(pitches, t); }, liftAt)];
  const repressAt = liftAt + gap;
  if (repressAt <= sendAt) {
    ids.push(Tone.Transport.scheduleOnce(t => midiApplyChord(pitches, t), sendAt));
  } else {
    ids.push(Tone.Transport.scheduleOnce(
      t => midiApplyChord(pitches.filter(p => !lifted.includes(p)), t), sendAt));
    ids.push(Tone.Transport.scheduleOnce(t => midiApplyChord(pitches, t), repressAt));
  }
  return ids;
}

/**
 * Schedule a chord change on the piano for `frame`, `midiLeadMs` early.
 * @param {Array<number>} pitches - New chord; [] releases everything
 * @param {number} frame - Transport frame the chord should *sound* at
 * @param {boolean} restrike - Re-press pitches already held (a chord onset)
 * @param {number=} prevFrame - Frame the previous chord was pressed at,
 *   which sets how long its keys were held (see midiRestrikeGapFor)
 * @return {Array<number>} Transport event IDs, cancellable
 */
function scheduleMidiChord(pitches, frame, restrike, prevFrame) {
  const lead = (midiLeadInput.valueAsNumber || 0) / 1000;
  const now = Tone.Transport.seconds;
  const soundAt = Tone.Time(frameToTransportTime(frame)).toSeconds();
  const gap = midiRestrikeGapFor(prevFrame === undefined ? undefined :
    (frame - prevFrame) * Tone.Time('16n').toSeconds());  // a frame is a 16th
  let sendAt = soundAt - lead;
  const ids = [];

  if (sendAt < now - MIDI_STALE_MS / 1000 && pitches.length) {
    // Learned about this chord too late to sound it on time. Skip it, as
    // Aria does, rather than play it late; the next change resyncs.
    return ids;
  }
  sendAt = Math.max(sendAt, now + 0.002);
  if (restrike && pitches.length) {
    // Lift first: with no time to spare both share a time, and insertion
    // order holds
    return ids.concat(scheduleMidiRestrike(
      pitches, Math.min(Math.max(sendAt - gap, now + 0.002), sendAt), sendAt, gap));
  }
  ids.push(Tone.Transport.scheduleOnce(t => midiApplyChord(pitches, t), sendAt));
  return ids;
}

/**
 * scheduleMidiChord for pre-planned chords that are never re-planned
 * (ground-truth playback), with the re-strike gap decided just before it is
 * needed rather than when the chord is scheduled, so the gap settings can be
 * changed while a loop plays. The decision runs MIDI_DECIDE_AHEAD before the
 * chord is sent -- more than the largest gap the setting allows -- and then
 * schedules the lift and the press. Not used for auto mode, whose events
 * must stay cancellable by id when the model re-plans.
 * @param {Array<number>} pitches
 * @param {number} frame - Transport frame the chord should sound at
 * @param {number=} prevFrame - Frame the previous chord was pressed at
 * @param {function(number)=} onGap - Told the gap (s) once decided
 */
function scheduleMidiChordLive(pitches, frame, prevFrame, onGap) {
  const lead = (midiLeadInput.valueAsNumber || 0) / 1000;
  const now = Tone.Transport.seconds;
  const spf = Tone.Time('16n').toSeconds();
  const sendAt = Tone.Time(frameToTransportTime(frame)).toSeconds() - lead;
  if (sendAt < now - MIDI_STALE_MS / 1000 && pitches.length) return;  // too late
  Tone.Transport.scheduleOnce(() => {
    const nowT = Tone.Transport.seconds;
    const gap = midiRestrikeGapFor(
      prevFrame === undefined ? undefined : (frame - prevFrame) * spf);
    const pressAt = Math.max(sendAt, nowT + 0.002);
    scheduleMidiRestrike(
      pitches, Math.min(Math.max(sendAt - gap, nowT + 0.002), pressAt), pressAt, gap);
    if (onGap) onGap(gap);
  }, Math.max(sendAt - MIDI_DECIDE_AHEAD, now + 0.001));
}
const MIDI_DECIDE_AHEAD = 0.55;  // s; the gap setting goes up to 500 ms

let midiStrikeToken = 0;

/** Strike a chord right now, outside the Transport schedule (key presses). */
function midiStrikeNow(pitches) {
  if (!midiChordOut) return;
  const token = ++midiStrikeToken;
  const now = Tone.context.currentTime;
  const gap = midiRestrikeGapMs() / 1000;
  const held = pitches.filter(p => midiSounding.has(p));
  // Keys released less than a gap ago (e.g. space let go and tapped again)
  // haven't reset yet either: they wait out the rest of their gap.
  const recent = pitches.filter(p => !held.includes(p) &&
    midiReleasedAt.has(p) && midiReleasedAt.get(p) > now - gap);
  // New pitches sound at once and the old chord is released, including the
  // pitches it shares with the new one; those are pressed again once the
  // action has had time to reset.
  midiApplyChord(pitches.filter(p => !held.includes(p) && !recent.includes(p)), now);
  const wait = Math.max(held.length ? gap : 0,
    ...recent.map(p => midiReleasedAt.get(p) + gap - now));
  if (wait > 0) {
    setTimeout(() => {
      if (token === midiStrikeToken) {  // not superseded by a later press
        midiApplyChord(pitches, Tone.context.currentTime);
      }
    }, wait * 1000);
  }
}

/**
 * Sound a chord the performer just triggered (triggered mode)
 * with no re-strike gap: keys it shares with the chord already down are
 * simply held, new keys are pressed and the rest released, all at once.
 * Nothing waits for the action to reset, so a chord is never delayed --
 * at the price that a shared or just-released key isn't struck again.
 */
function midiPlayNow(pitches) {
  if (!midiChordOut) return;
  midiStrikeToken++;  // drop any re-press an earlier strike left pending
  midiApplyChord(pitches, Tone.context.currentTime);
}

/** Release everything on the piano, including the sustain pedal. */
function midiAllOff() {
  if (!midiChordOut) return;
  midiStrikeToken++;  // drop any re-press still waiting on its gap
  midiScheduledByFrame.forEach(({ ids }) => ids.forEach(id => Tone.Transport.clear(id)));
  midiScheduledByFrame = new Map();
  [...midiKeyHolders.keys()].forEach(p =>
    midiChordOut.send([0x80, p, 0], { time: midiTimestamp(Tone.context.currentTime) }));
  midiKeyHolders.clear();
  midiSounding.clear();
  midiChordOut.send([0xB0, 123, 0]);  // all notes off
  midiChordOut.send([0xB0, 64, 0]);   // sustain pedal up
}

/** Rebuild the chord output list, keeping the current choice if possible. */
function refreshMIDIOutputs() {
  const previous = chordOutputSelect.value;
  chordOutputSelect.innerHTML = '<option value="">Laptop (instruments)</option>';
  WebMidi.outputs.forEach(output => {
    const option = document.createElement('option');
    option.textContent = output.name;
    option.value = output.name;
    chordOutputSelect.appendChild(option);
  });
  if (WebMidi.outputs.some(o => o.name === previous)) {
    chordOutputSelect.value = previous;
  } else {
    selectChordOutput('');
  }
}

/** @param {string} name - MIDI output name, or '' for the laptop synth */
function selectChordOutput(name) {
  midiAllOff();
  chordSynth.releaseAll();
  midiChordOut = name ? WebMidi.getOutputByName(name) || null : null;
  chordOutputSelect.value = midiChordOut ? name : '';
}

/**
 * Load sample instrument
 * @param {string} instrument
 * @return {Promise!}
 */
function loadInstrument(instrument) {
  return new Promise((resolve) => {
    const sampler = SampleLibrary.load({
      instruments: instrument,
      baseUrl: 'https://nbrosowsky.github.io/tonejs-instruments/samples/',
      onload: () => {
        resolve(sampler);
      },
    });
  });
}

/**
 * Load Salamander piano instrument
 * @return {Promise!}
 */
function loadSalamander() {
  const urls = {};
  for (const baseNote of ['A', 'C', 'Ds', 'Fs']) {
    for (const pitch of Array(7).keys()) {
      urls[`${baseNote.replace('s', '#')}${pitch + 1}`] =
        `${baseNote}${pitch + 1}.mp3`;
    }
  }
  return new Promise((resolve) => {
    const sampler = new Tone.Sampler({
      urls,
      baseUrl: 'https://tonejs.github.io/audio/salamander/',
      onload: () => {
        resolve(sampler);
      },
    });
  });
}

/**
 * Load ToneJS synth instrument
 * @param {Object!} constructor
 * @return {Promise!}
 */
function loadSynth(constructor) {
  return new Promise(resolve => resolve(new Tone.PolySynth(constructor)));
}

/**
 * Load all available instruments
 * @param {Object!} dest - ToneJS context destination
 * @return {Object!} mapping of instrument name to ToneJS instrument
 */
async function loadInstruments(dest) {
  const instrumentNames = [
    'Piano (Salamander)', 'Piano (Versilian)', 'AM Synth', 'FM Synth', 'Harp',
    'Acoustic Guitar'
  ];
  const loadFns = [
    loadSalamander(), loadInstrument('piano'), loadSynth(Tone.AMSynth),
    loadSynth(Tone.FMSynth), loadInstrument('harp'),
    loadInstrument('guitar-acoustic')
  ];
  const instruments = await Promise.all(loadFns);
  return instrumentNames.reduce(
    (prev, name, idx) =>
      ({ ...prev, [name]: instruments[idx].connect(dest).toDestination() }),
    {});
}

/** Create instrument dropdown options and callbacks */
function setupInstrumentSelection() {
  for (const selectEl of [chordInstSelect, melodyInstSelect]) {
    for (const name of Object.keys(instrumentMap)) {
      const option = document.createElement('option');
      option.append(name);
      option.value = name;
      selectEl.appendChild(option);
    }
  }
  chordInstSelect.addEventListener('change', event => {
    chordSynth.releaseAll();
    chordSynth = instrumentMap[event.target.value];
    chordInstSelect.blur();
  });
  melodyInstSelect.addEventListener('change', event => {
    melodySynth.releaseAll();
    melodySynth = instrumentMap[event.target.value];
    melodyInstSelect.blur();
  });
  chordInstSelect.value = DEFAULTS.chordInstrument;
  melodyInstSelect.value = DEFAULTS.melodyInstrument;
}

/** Create mouse events for note playing */
function enableClickingInputs() {
  const keys = document.querySelectorAll('rect');
  let index, note;
  keys.forEach(key => {
    const playKey = () => {
      index = key.getAttribute('data-index');
      note = pianoNotes[index];
      playNote(note, melodyVelocity, true);
    };

    const releaseKey = () => {
      index = key.getAttribute('data-index');
      note = pianoNotes[index];
      releaseNote(note, true);
    };

    key.addEventListener('mousedown', playKey);
    key.addEventListener('mouseenter', () => {
      if (mouseDown) {
        playKey();
      }
    });
    key.addEventListener('mouseup', releaseKey);
    key.addEventListener('mouseleave', releaseKey);
  });
}

/** Set computer key to note mapping and reset necessary globals */
function setKeysToNotes() {
  keysToNotes = {
    'a': `C${compKeyboardOctave}`,
    'w': `C#${compKeyboardOctave}`,
    's': `D${compKeyboardOctave}`,
    'e': `D#${compKeyboardOctave}`,
    'd': `E${compKeyboardOctave}`,
    'f': `F${compKeyboardOctave}`,
    't': `F#${compKeyboardOctave}`,
    'g': `G${compKeyboardOctave}`,
    'y': `G#${compKeyboardOctave}`,
    'h': `A${compKeyboardOctave}`,
    'u': `A#${compKeyboardOctave}`,
    'j': `B${compKeyboardOctave}`,
    'k': `C${compKeyboardOctave + 1}`,
    'o': `C#${compKeyboardOctave + 1}`,
    'l': `D${compKeyboardOctave + 1}`,
    'p': `D#${compKeyboardOctave + 1}`,
    ';': `E${compKeyboardOctave + 1}`,
    '\'': `F${compKeyboardOctave + 1}`,
  };
  visual.setComputerKeyOctave(compKeyboardOctave);
  enableClickingInputs();  // Add listeners on new keys since old ones removed
}

/**
 * If a keystroke is meant for a form field rather than for playing. Without
 * this, typing a song title into the search box would play melody notes and
 * (in triggered chord mode) every space would fire a chord.
 * @param {Event!} event
 * @return {boolean}
 */
function isTypingTarget(event) {
  const tag = event.target && event.target.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
}

/** Create key events for playing notes with the computer keyboard */
function enableKeyboardInputs() {
  setKeysToNotes();
  document.addEventListener('keydown', event => {
    if (isTypingTarget(event)) {
      return;
    }
    // Space plays the scheduled chord in triggered mode. Swallow it so it
    // doesn't also activate whatever button happens to have focus.
    if (event.code === 'Space' && curSession && isTriggeredChordMode()) {
      event.preventDefault();
      if (!event.repeat) {
        triggerScheduledChord();
      }
      return;
    }
    // Complete mode: space makes the agent play now (the pedal must be down)
    if (event.code === 'Space' && curSession && isCompleteMode()) {
      event.preventDefault();
      if (!event.repeat && !isInputMode()) spaceDown();
      return;
    }
    const note = keysToNotes[event.key];
    if (note && !heldNotes[note]) {
      heldNotes[note] = true;
      playNote(note, melodyVelocity, true);
    }
  });
  document.addEventListener('keyup', event => {
    if (isTypingTarget(event)) {
      return;
    }
    // In triggered mode the chord lasts while space is held
    if (event.code === 'Space' && curSession && isTriggeredChordMode()) {
      event.preventDefault();
      releaseTriggeredChord();
      return;
    }
    if (event.code === 'Space' && curSession && isCompleteMode()) {
      event.preventDefault();
      if (!isInputMode()) spaceUp();
      return;
    }
    const note = keysToNotes[event.key];
    if (note) {
      heldNotes[note] = false;
      releaseNote(note, true);
    } else if (event.key === 'z') {
      compKeyboardOctave =
        Math.max(compKeyboardOctave - 1, compKeyOctaveRange[0]);
      setKeysToNotes();
    } else if (event.key === 'x') {
      compKeyboardOctave =
        Math.min(compKeyboardOctave + 1, compKeyOctaveRange[1]);
      setKeysToNotes();
    }
  });
}

/** Create MIDI input dropdown options */
/**
 * Show a single disabled status line in the MIDI interface dropdown, so it
 * says why it's empty instead of silently showing nothing.
 * @param {string} text
 */
function showMIDIStatus(text) {
  interfaceSelect.innerHTML = '';
  const option = document.createElement('option');
  option.textContent = text;
  option.disabled = true;
  option.selected = true;
  interfaceSelect.appendChild(option);
}

/** Request MIDI access, then keep the input list in sync with hot-plugging */
function enableMIDI() {
  if (!window.isSecureContext) {
    // Web MIDI only exists on https:// or localhost, not on 0.0.0.0 or a LAN IP
    showMIDIStatus('MIDI needs http://localhost');
    console.warn('Web MIDI unavailable: open the app via http://localhost:<port>');
    midiMonitorLine('no Web MIDI: page is not a secure context (use https or localhost)');
    return;
  }
  // Chrome asks for MIDI permission via an icon in the address bar; until
  // it's answered, enable() stays pending
  showMIDIStatus('Waiting for MIDI permission…');
  midiMonitorLine('MIDI monitor on; waiting for MIDI permission');
  WebMidi.enable()
    .then(() => {
      midiMonitorLine('MIDI enabled; inputs: ' +
        (WebMidi.inputs.map(i => `"${i.name}"`).join(', ') || 'none'));
      refreshMIDIInputs();
      refreshMIDIOutputs();
      WebMidi.addListener('connected', () => {
        refreshMIDIInputs();
        refreshMIDIOutputs();
      });
      WebMidi.addListener('disconnected', () => {
        refreshMIDIInputs();
        refreshMIDIOutputs();
      });
    })
    .catch(err => {
      showMIDIStatus('MIDI unavailable');
      console.error('WebMidi.enable failed:', err);
      midiMonitorLine(`MIDI unavailable: ${err && err.message || err}`);
    });
}

/**
 * Rebuild the input dropdown. Keeps the current choice if it's still
 * connected; otherwise prefers a real device over virtual ports such as
 * Linux's "Midi Through", which ALSA always lists first.
 */
function refreshMIDIInputs() {
  const previous = interfaceSelect.value;
  webMIDIInputs = WebMidi.inputs;
  if (webMIDIInputs.length < 1) {
    showMIDIStatus('No MIDI device found');
    return;
  }
  interfaceSelect.innerHTML = '';
  webMIDIInputs.forEach(input => {
    const option = document.createElement('option');
    option.textContent = input.name;
    option.value = input.name;
    interfaceSelect.appendChild(option);
  });
  const names = webMIDIInputs.map(input => input.name);
  const preferred = names.includes(previous) ? previous :
    (names.find(n => !/through/i.test(n)) || names[0]);
  selectMIDIInterface(preferred);
}

/**
 * Select active MIDI input device
 * @param {string} name - dropdown value associated with input
 */
function selectMIDIInterface(name) {
  interfaceSelect.value = name;

  webMIDIInputs.forEach(input => input.removeListener());
  let curInterface = webMIDIInputs.find(input => input.name === name);
  if (!curInterface) {
    return;
  }

  // Listen on all channels: devices differ in which one they send on
  curInterface.addListener('noteon', e => {
    playNote(e.note.identifier, e.velocity);
  });
  curInterface.addListener('noteoff', e => {
    releaseNote(e.note.identifier);
  });
  curInterface.addListener('midimessage', e => {
    const data = e.message.data;
    if ((data[0] & 0xF0) === 0xB0) logPedal(data[1], data[2]);
    if ((data[0] & 0xF0) === 0xB0 && data[1] === AGENT_PEDAL_CC) onAgentPedal(data[2]);
  });
  if (MIDI_MONITOR) {
    curInterface.addListener('midimessage', e => logMidiMessage(e.message.data));
    midiMonitorLine(`listening on input "${name}"`);
  }
}

/**
 * MIDI monitor, on when the page is opened with ?midilog in the URL: shows
 * every incoming message (notes, pedals, other controllers) in a box at the
 * bottom of the page and in the console. For checking what an instrument
 * sends, e.g. which pedal controllers a Disklavier transmits and whether
 * the keys ReaLJam plays come back on its input.
 */
const MIDI_MONITOR = new URLSearchParams(window.location.search).has('midilog');
const MIDI_MONITOR_LINES = 20;
const MIDI_CC_NAMES = { 64: 'damper (right pedal)', 66: 'sostenuto (middle pedal)',
  67: 'soft (left pedal)' };
let midiMonitorBox = null;

/** @param {Uint8Array} data - raw MIDI bytes */
function logMidiMessage(data) {
  const status = data[0];
  if (status === 0xF8 || status === 0xFE) return;  // clock, active sensing
  const type = status & 0xF0, channel = (status & 0x0F) + 1;
  let text;
  if (type === 0x90 && data[2] > 0) {
    text = `note on   ${data[1]} vel ${data[2]}`;
  } else if (type === 0x80 || type === 0x90) {
    text = `note off  ${data[1]}`;
  } else if (type === 0xB0) {
    text = `CC ${data[1]} ${MIDI_CC_NAMES[data[1]] || ''} = ${data[2]}`;
  } else {
    text = `raw ${[...data].map(b => b.toString(16).padStart(2, '0')).join(' ')}`;
  }
  midiMonitorLine(`ch${channel}  ${text}`);
}

/** Append one timestamped line to the monitor box (created on first use) */
function midiMonitorLine(text) {
  if (!MIDI_MONITOR) return;
  const line = `${new Date().toISOString().slice(11, 23)}  ${text}`;
  console.log('[MIDI monitor]', line);
  if (!midiMonitorBox) {
    midiMonitorBox = document.createElement('pre');
    midiMonitorBox.style.cssText = 'position:fixed;bottom:0;right:0;margin:0;' +
      'padding:8px;max-width:100%;background:rgba(0,0,0,0.8);color:#0f0;' +
      'font-size:12px;z-index:1000;pointer-events:none;';
    document.body.appendChild(midiMonitorBox);
  }
  const lines = (midiMonitorBox.textContent ? midiMonitorBox.textContent.split('\n') : []);
  lines.push(line);
  midiMonitorBox.textContent = lines.slice(-MIDI_MONITOR_LINES).join('\n');
}

/**
 * Load metronome ToneJS sample instrument
 * @return {Promise!}
 */
function loadMetronome() {
  return new Promise((resolve) => {
    const metronome = SampleLibrary.load({
      instruments: 'metronome',
      baseUrl: 'https://lukewys.github.io/files/tonejs-samples/',
      onload: () => {
        resolve(metronome.toDestination());
      },
    });
  });
}

/**
 * Set ToneJS bpm
 * @param {number} bpm
 */
function setBPM(bpm) {
  Tone.Transport.bpm.value = bpm;
}

/**
 * Set ToneJS time signature
 * @param {number} timeSig
 */
function setTimeSig(timeSig) {
  Tone.Transport.timeSignature = timeSig;
  timeSigInput.value = Tone.Transport.timeSignature;
}

/** Schedule repeating events for metronome */
function startMetronome() {
  metronomeStatus = true;
  metronomeBtn.textContent = 'Disable Metronome';
  metronomeEvents.push(Tone.Transport.scheduleRepeat(time => {
    metronome.triggerAttackRelease('C1', '8n', time, metronomeVelocity);
  }, '1m', 0));
  const offBeatFreq = metronomeFreq === 'beat' ? 1 : 4;
  for (let i = 1; i < Tone.Transport.timeSignature * offBeatFreq; i++) {
    const offBeatOnset = metronomeFreq === 'beat' ? `0:${i}:0` : `0:0:${i}`;
    metronomeEvents.push(Tone.Transport.scheduleRepeat(time => {
      metronome.triggerAttackRelease('C0', '8n', time, metronomeVelocity);
    }, '1m', offBeatOnset));
  }
  if (Tone.Transport.state !== 'started') {
    Tone.Transport.start();
  }
}

/** Cancel scheduled metronome events */
function stopMetronome() {
  metronomeStatus = false;
  metronomeBtn.textContent = 'Enable Metronome';
  metronomeEvents.forEach(id => Tone.Transport.clear(id));
  metronomeEvents = [];
  if (!curSession) {
    Tone.Transport.stop();
  }
}

/** Start or stop playing metronome */
function toggleMetronome() {
  if (metronomeStatus) {
    stopMetronome();
  } else {
    startMetronome();
  }
}

/** Toggle metronome frequency */
function toggleMetronomeFreq() {
  if (metronomeFreq === 'beat') {
    metronomeFreq = 'frame';
    metronomeFreqBtn.textContent = '1/16';
  } else {
    metronomeFreq = 'beat';
    metronomeFreqBtn.textContent = '1/4';
  }
}

/**
 * Get frames since ToneJS start
 * @return {number}
 */
function getTransportFrame() {
  const [bars, beats, frames] = Tone.Transport.position.split(':');
  return parseInt(bars) * 16 + parseInt(beats) * 4 + parseFloat(frames);
}

/**
 * Convert frame to transport time format
 * @param {number} frame
 * @return {string}
 */
function frameToTransportTime(frame) {
  const bars = Math.floor(frame / 16);
  const beats = Math.floor((frame % 16) / 4);
  const frames = frame % 4;
  return `${bars}:${beats}:${frames}`;
}

/**
 * Initialize globals, load everything, setup input handlers
 * @param {NoteVisual!} visual_arg - initialized note visualizer class
 */
async function initializeMIDIReader(visual_arg) {
  visual = visual_arg;
  Tone.Transport.timeSignature = DEFAULTS.timeSignature;
  Tone.Transport.bpm.value = DEFAULTS.bpm;

  // Setup audio recorder
  const actx = Tone.context;
  const dest = actx.createMediaStreamDestination();
  recorder = new MediaRecorder(dest.stream);
  recorder.ondataavailable = (e) => {
    curAudioRecording.push(e.data);
  };
  recorder.onstop = saveSessionRecording;

  // Load instruments and metronome
  instrumentMap = await loadInstruments(dest);
  chordSynth = instrumentMap[DEFAULTS.chordInstrument];
  melodySynth = instrumentMap[DEFAULTS.melodyInstrument];
  metronome = await loadMetronome();

  // Set up button and input handlers
  playBtn = document.getElementById('play-btn');
  playBtn.addEventListener('click', showMainScreen);
  metronomeBtn = document.getElementById('metronome-button');
  metronomeBtn.addEventListener('click', toggleMetronome);
  bpmInput = document.getElementById('bpm-input');
  bpmInput.value = Tone.Transport.bpm.value;
  addNumericInputEventListener(bpmInput, setBPM);
  timeSigInput = document.getElementById('time-sig-input');
  timeSigInput.value = DEFAULTS.timeSignature;
  addNumericInputEventListener(timeSigInput, setTimeSig);
  metronomeFreqBtn = document.getElementById('metronome-freq-button');
  metronomeFreqBtn.addEventListener('click', toggleMetronomeFreq);
  if (window.location.hostname !== 'localhost') {
    metronomeFreqBtn.style = 'display: none';
  }
  interfaceSelect = document.getElementById('interface-select');
  interfaceSelect.addEventListener(
    'change', event => selectMIDIInterface(event.target.value));
  modelSelect = document.getElementById('model-select');
  modelSelect.addEventListener('change', onModelChanged);
  liveSessionBtn = document.getElementById('live-session-button');
  liveSessionBtn.addEventListener('click', toggleLiveSession);
  downloadSessionCheck = document.getElementById('download-session-check');
  metronomeCheck = document.getElementById('metronome-check');
  showChordsCheck = document.getElementById('show-chords-check');
  customVoicingsCheck = document.getElementById('custom-voicings-check');
  voicingTemperatureInput = document.getElementById('voicing-temperature-input');
  setupVoicingLookupSelect();
  const controlsToggle = document.getElementById('controls-toggle');
  controlsToggle.addEventListener('click', () => {
    const footer = document.querySelector('.footer');
    footer.classList.toggle('collapsed');
    controlsToggle.textContent =
      footer.classList.contains('collapsed') ? 'Controls ▾' : 'Controls ▴';
    controlsToggle.blur();  // so space keeps triggering chords, not this button
    window.dispatchEvent(new Event('resize'));  // let the piano roll re-fit
  });
  songSearchInput = document.getElementById('song-search-input');
  songSearchInput.addEventListener('input', onSongSearchInput);
  songSearchInput.addEventListener('keydown', onSongSearchKeydown);
  bassSongsOnlyCheck = document.getElementById('bass-songs-only-check');
  bassSongsOnlyCheck.addEventListener('change', () => {
    songSearchLastQuery = '';  // the narrowed pool no longer applies
    onSongSearchInput();
    bassSongsOnlyCheck.blur();
  });
  songSearchResults = document.getElementById('song-search-results');
  document.querySelectorAll('input[name="gt-mode"]').forEach(radio =>
    radio.addEventListener('change', () => {
      refreshModelOptions();
      radio.blur();
    }));
  chordTimingSelect = document.getElementById('chord-timing-select');
  chordTimingInfo = document.getElementById('chord-timing-info');
  agentGateCheck = document.getElementById('agent-gate-check');
  agentGateCheck.addEventListener('change', () => {
    // Switching the gate on or off mid-session lets the agent in or out now
    if (curSession && isCompleteMode() && !isInputMode()) agentGateChanged();
    updateChordTimingInfo();
  });
  chordTimingSelect.addEventListener('change', () => {
    updateChordTimingInfo();
    chordTimingSelect.blur();
  });
  voicingBrowserChordSelect = document.getElementById('voicing-browser-chord-select');
  voicingBrowserChordSelect.addEventListener('change', onVoicingBrowserChordSelected);
  voicingBrowserChordSelect.addEventListener('keydown', onVoicingBrowserKeydown);
  voicingBrowserInfo = document.getElementById('voicing-browser-info');
  voicingBrowserNotesSelect = document.getElementById('voicing-browser-notes-select');
  voicingBrowserLowInput = document.getElementById('voicing-browser-low-input');
  voicingBrowserHighInput = document.getElementById('voicing-browser-high-input');
  voicingBrowserLowValue = document.getElementById('voicing-browser-low-value');
  voicingBrowserHighValue = document.getElementById('voicing-browser-high-value');
  voicingBrowserNotesSelect.addEventListener('change', onVoicingBrowserFilterChanged);
  voicingBrowserLowInput.addEventListener('input', onVoicingBrowserFilterChanged);
  voicingBrowserHighInput.addEventListener('input', onVoicingBrowserFilterChanged);
  voicingBrowserNotesSelect.addEventListener('keydown', onVoicingBrowserKeydown);
  [voicingBrowserNotesSelect, voicingBrowserLowInput, voicingBrowserHighInput]
    .forEach(el => el.addEventListener('change', refocusVoicingBrowser));
  temperatureInput = document.getElementById('temperature-input');
  temperatureInput.value = DEFAULTS.temperature;
  addNumericInputEventListener(temperatureInput);
  silenceInput = document.getElementById('silence-input');
  silenceInput.value = DEFAULTS.silence;
  addNumericInputEventListener(silenceInput);
  commitaheadInput = document.getElementById('commitahead-input');
  commitaheadInput.value = DEFAULTS.commitahead;
  commitaheadInput.max = DEFAULTS.lookahead;
  addNumericInputEventListener(commitaheadInput);
  inputLagInput = document.getElementById('input-lag-input');
  inputLagInput.value = DEFAULTS.inputLagMs;
  addNumericInputEventListener(inputLagInput);
  lookaheadInput = document.getElementById('lookahead-input');
  lookaheadInput.value = DEFAULTS.lookahead;
  setLookahead(DEFAULTS.lookahead);
  addNumericInputEventListener(lookaheadInput, setLookahead);
  chordInstSelect = document.getElementById('chord-inst-select');
  melodyInstSelect = document.getElementById('melody-inst-select');
  setupInstrumentSelection();
  chordOutputSelect = document.getElementById('chord-output-select');
  chordOutputSelect.addEventListener('change', event => {
    selectChordOutput(event.target.value);
    chordOutputSelect.blur();
  });
  midiVelocityInput = bindSliderValueDisplay(
    'midi-velocity-input', 'midi-velocity-value', 0);
  midiLeadInput = document.getElementById('midi-lead-input');
  addNumericInputEventListener(midiLeadInput);
  midiMelodyVelocityInput = bindSliderValueDisplay(
    'midi-melody-velocity-input', 'midi-melody-velocity-value', 0);
  MIDI_RELEASE_ANCHORS.forEach(anchor => {
    anchor.input = document.getElementById(anchor.id);
    addNumericInputEventListener(anchor.input);
  });

  // Do initial setup with server
  establishServerConnection();
  updateChordTimingInfo();

  console.log('ready!');
  playBtn.textContent = 'Play';
  playBtn.removeAttribute('disabled');
  playBtn.classList.remove('loading');
}