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
  customVoicingsCheck, vlWeightInput, regWeightInput, songSearchInput,
  songSearchResults, voicingBrowserChordSelect, voicingBrowserInfo,
  chordTimingSelect, chordTimingInfo, referenceModeCheck;
let songCatalogue = [];   // [{dataset, split, id, title, artist}, ...]
let songSearchMatches = [];  // Current fuzzy hits, best first (capped)
let songSearchIndex = 0;     // Keyboard-highlighted row in songSearchMatches
let songSearchLastQuery = '';  // Query songSearchPool was narrowed down to
let songSearchPool = [];       // Every song matching it, not just the shown ones
const songSearchMaxResults = 30;
const songSearchMinChars = 2;
let voicingBrowserList = [];  // [{pitches, count, num_songs}, ...] for the selected chord
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
  WebMidi.enable().then(enableMIDIInputs).catch(err => alert(err));
}

/** Download audio recording and JSON of the most recent session */
function saveSessionRecording() {
  if (!downloadSessionCheck.checked) {
    return;
  }
  for (const [data, type] of [
    // disabled downloadding audio recording
    // [curAudioRecording, 'audio/ogg; codecs=opus'],
    [[JSON.stringify(lastSession)], 'text/plain']]) {
    const blob = new Blob(data, { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    document.body.appendChild(a);
    a.style = 'display: none';
    a.href = url;
    a.download =
      `GenJam Session - ${lastSession.startTime} ${lastSession.description}`;
    a.click();
    URL.revokeObjectURL(url);
  }
}

/** Start or stop a live session */
function toggleLiveSession() {
  if (curSession) {
    lastSession = curSession;
    lastSession.description = `sic${showChordsCheck.checked}_bpm${bpmInput.value}_met${metronomeStatus}_la${lookaheadInput.value}_com${commitaheadInput.value}_sil${silenceInput.value}_temp${temperatureInput.value}_timing${chordTimingSelect.value}_${modelSelect.value}`;
    curSession = undefined;
    liveSessionBtn.textContent = 'Start Live Session';
    showChordsCheck.disabled = false;
    bpmInput.disabled = false;
    timeSigInput.disabled = false;
    silenceInput.disabled = false;
    chordTimingSelect.disabled = false;
    Tone.Transport.stop();
    Tone.Transport.cancel();
    stopMetronome();
    pianoNotes.forEach(note => chordSynth.triggerRelease(note));
    visual.clearScheduledNotes(0);
    visual.stopAllNotes();
    recorder.stop();
    updateChordTimingInfo();
  } else {
    startSession();
  }
}

/**
 * Start a fresh session and the Transport clock. Shared by manual live
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
    manualChord: null,      // Manual mode: chord currently being sustained
    pendingChord: null,     // Manual mode: chord the next space press fires
    referenceOnly: !!options.referenceOnly,  // Playing ground truth, no model
  };
  liveSessionBtn.textContent = 'Stop Live Session';
  showChordsCheck.disabled = true;
  bpmInput.disabled = true;
  timeSigInput.disabled = true;
  silenceInput.disabled = true;
  chordTimingSelect.disabled = true;

  // Manual mode drives chords independently of the melody, so it doesn't wait
  // for a first note the way the auto loop does (see playNote).
  if (isManualChordMode() && !curSession.referenceOnly) {
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
 * but flagged referenceOnly so no /play or /advance_chord request is ever
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

  Tone.Transport.scheduleOnce(() => {
    if (curSession && curSession.referenceOnly) {
      scheduleReferenceLoop(melody, chords, nextFrame, loopFrames);
    }
  }, frameToTransportTime(nextFrame));
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
    if (on) {
      melodySynth.triggerAttack(note, '+0', melodyVelocity);
      visual.noteOn(pitch);
    } else {
      melodySynth.triggerRelease(note);
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

  // Set start frame if needed, subtract modulo to always start on the beat
  if (!curSession.startFrame) {
    curSession.startFrame = frame - (frame % fpb);
  }

  return frame - curSession.startFrame;
}

/**
 * If chords wait for the performer instead of playing on the model's own
 * predicted timing. See advanceChordNow.
 * @return {boolean}
 */
function isManualChordMode() {
  return chordTimingSelect.value === 'manual';
}

/**
 * Kick off whichever generation loop the current chord-timing mode uses.
 * Idempotent -- called from several places that can each be the first to
 * happen in a session (first melody note, robot playback, session start in
 * manual mode, or a space press before any of those).
 */
function startGenerationLoop() {
  if (!curSession || curSession.loopStarted || curSession.referenceOnly) {
    return;
  }
  curSession.loopStarted = true;
  getSessionCurrentFrame();  // Anchor session frame 0 before scheduling
  if (isManualChordMode()) {
    pendingChordLoop();
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
  const result = await fetch(`${window.location.origin}/play`, {
    'method': 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: modelSelect.value,
      notes: curSession.noteHistory,
      chordTokens: curSession.chordTokens,
      frame: curFrame + 1,  // Request chords to play at the next frame
      lookahead: getLookaheadFrames(),
      commitahead: getCommitaheadFrames(),
      silenceTill: getSilenceFrames(),
      temperature: temperatureInput.valueAsNumber,
      introSet: curSession.introSet,
      useCustomVoicings: customVoicingsCheck.checked,
      prevVoicing: curSession.lastVoicing,
      vlWeight: vlWeightInput.valueAsNumber,
      regWeight: regWeightInput.valueAsNumber,
    })
  });
  const json = await result.json();

  // Exit loop if current session ended (in case ended during fetch)
  if (!curSession) {
    return;
  }

  // Schedule chords based on agent response
  processAgentAction(json);

  // Send next request to server, wait until next frame if haven't advanced yet
  if (curFrame === getSessionCurrentFrame()) {
    const fps = fpb * Tone.Transport.bpm.value / 60;
    setTimeout(syncWithServer, 1000 / fps);
  } else {
    syncWithServer();
  }
}

/**
 * Manual chord timing: the model chooses which chord comes next, the
 * performer chooses when it lands. Nothing is scheduled onto the Transport
 * and nothing is committed -- a chord sustains until the next space press.
 *
 * The next chord is fetched ahead of time and cached in
 * curSession.pendingChord so a press fires instantly rather than waiting on
 * a round trip, and refreshed once per beat so it keeps responding to melody
 * played since the last press.
 */

/**
 * Bring curSession.chordTokens up to the current frame, so the model sees an
 * accurate account of what has been sounding. Frames with no chord are left
 * as -1, which the server fills with SILENCE.
 */
function fillManualChordTokens() {
  const curFrame = getSessionCurrentFrame();
  for (let i = 0; i <= curFrame; i++) {
    if (curSession.chordTokens[i] === undefined) {
      curSession.chordTokens[i] = -1;
    }
  }
  const held = curSession.manualChord;
  if (held) {
    // Everything after the onset frame is the same chord still ringing
    for (let i = held.startFrame + 1; i <= curFrame; i++) {
      curSession.chordTokens[i] = held.holdToken;
    }
  }
}

/** Ask the model which chord it would start right now, and cache it */
async function fetchPendingChord() {
  if (!curSession) {
    return;
  }
  fillManualChordTokens();
  const result = await fetch(`${window.location.origin}/advance_chord`, {
    'method': 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: modelSelect.value,
      notes: curSession.noteHistory,
      chordTokens: curSession.chordTokens,
      frame: getSessionCurrentFrame() + 1,
      temperature: temperatureInput.valueAsNumber,
      useCustomVoicings: customVoicingsCheck.checked,
      prevVoicing: curSession.lastVoicing,
      vlWeight: vlWeightInput.valueAsNumber,
      regWeight: regWeightInput.valueAsNumber,
    })
  });
  const chord = await result.json();
  if (!curSession) {
    return;
  }
  curSession.pendingChord = chord;
  updateChordTimingInfo();
}

/** Keep the queued chord fresh while waiting for the performer */
async function pendingChordLoop() {
  if (!curSession || !isManualChordMode()) {
    return;
  }
  await fetchPendingChord();
  if (!curSession || !isManualChordMode()) {
    return;
  }
  setTimeout(pendingChordLoop, 60000 / Tone.Transport.bpm.value);
}

/**
 * Play the queued chord immediately, releasing whatever is sustaining.
 * Bound to space in manual mode.
 */
async function advanceChordNow() {
  startGenerationLoop();
  if (!curSession.pendingChord) {
    // Nothing queued yet (session just started, or a fetch is still in
    // flight) -- fetch one now and take the round-trip hit for this press.
    await fetchPendingChord();
    if (!curSession || !curSession.pendingChord) {
      return;
    }
  }

  const chord = curSession.pendingChord;
  const frame = getSessionCurrentFrame();

  if (curSession.manualChord) {
    curSession.manualChord.pitches.forEach(pitch => {
      chordSynth.triggerRelease(pitchToNote[pitch]);
      visual.noteOff(pitch);
    });
  }
  chord.pitches.forEach(pitch => {
    chordSynth.triggerAttack(pitchToNote[pitch], '+0', chordVelocity);
    visual.noteOn(pitch, 'blue');
  });

  curSession.chordTokens[frame] = chord.onsetToken;
  curSession.manualChord = { ...chord, startFrame: frame };
  curSession.chordHistory.push(
    { scheduleFrame: frame, pitches: chord.pitches, symbol: chord.symbol, eventIDs: [] });
  if (chord.pitches.length) {
    curSession.lastVoicing = chord.pitches;
  }

  curSession.pendingChord = null;
  updateChordTimingInfo();
  fetchPendingChord();
}

/** Show what is sounding and what space will play next */
function updateChordTimingInfo() {
  if (!isManualChordMode()) {
    chordTimingInfo.textContent =
      'Chords play on the model\'s own predicted timing';
    return;
  }
  if (!curSession) {
    chordTimingInfo.textContent =
      'Start a session, then press space to play each chord';
    return;
  }
  const playing = curSession.manualChord
    ? curSession.manualChord.symbol : '--';
  const next = curSession.pendingChord
    ? curSession.pendingChord.symbol : 'thinking...';
  chordTimingInfo.textContent = `Playing: ${playing}   Space plays: ${next}`;
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
  pitches.forEach(pitch => {
    eventIDs.push(scheduleNote(pitchToNote[pitch], on, time));
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
function processAgentAction(
  { newChords, newChordTokens, introChordTokens, frame }) {
  const curFrame = getTransportFrame();
  const targetFrame = frame + curSession.startFrame;
  const silenceFrame = getSilenceFrames() + curSession.startFrame;

  // Fill in new chord tokens and fill possible gaps
  for (let i = Math.min(frame, curSession.chordTokens.length);
    i < frame + newChordTokens.length; i++) {
    if (i < frame) {
      curSession.chordTokens[i] = -1;
    } else {
      curSession.chordTokens[i] = newChordTokens[i - frame];
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

  // Track the most recent non-empty voicing (regardless of scheduling), so
  // it can be sent back as prevVoicing next poll for voice-leading
  // continuity across separate /play calls.
  for (const [, pitches] of newChords) {
    if (pitches && pitches.length) {
      curSession.lastVoicing = pitches;
    }
  }

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

    const isCommitted = frameOffset < getCommitaheadFrames();
    const prevFramePitches = getChordPitchesAtFrame(scheduleFrame - 1);

    // Onsets - release previous chord and play new chord (or rest)
    // Holds - release previous and play chord if different than previous
    //  (indicates model changed its mind about what chord to play)
    if (on || !arraysEqual(pitches, prevFramePitches)) {
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
    }
  });
}

/**
 * Get necessary info from server (model names) and then enable live sessions
 */
function establishServerConnection() {
  fetch(`${window.location.origin}/models`)
    .then(response => response.json())
    .then(json => {
      json.forEach(model => {
        const name = document.createTextNode(model);
        const option = document.createElement('option');
        option.append(name);
        option.value = model;
        modelSelect.appendChild(option);
      });
      console.log('Interactive agent is ready!');
      liveSessionBtn.disabled = false;
    });
  loadSongCatalogue();
  loadVoicingBrowserChords();
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
    ? songSearchPool : songCatalogue;

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
    row.addEventListener('click', () => playSongFromSearch(i));
    songSearchResults.appendChild(row);
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

  if (referenceModeCheck.checked) {
    // The song's own accompaniment, no model involved
    const custom = customVoicingsCheck.checked ? '?custom=1' : '';
    const result = await fetch(`${base}/reference${custom}`);
    const { melody, chords, num_beats } = await result.json();
    playSongReference(melody, chords, num_beats);
    return;
  }

  const result = await fetch(`${base}/melody`);
  const { melody, num_beats } = await result.json();
  playSongMelody(melody, num_beats);
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
    voicingBrowserList = [];
    voicingBrowserIndex = -1;
    return;
  }
  const result = await fetch(
    `${window.location.origin}/voicings?chord=${encodeURIComponent(chord)}`);
  voicingBrowserList = await result.json();
  voicingBrowserIndex = 0;
  playBrowsedVoicing();
}

/** Handle Left/Right arrow keys on the chord dropdown to step through voicings */
function onVoicingBrowserKeydown(event) {
  if (!voicingBrowserList.length) {
    return;
  }
  if (event.key === 'ArrowRight') {
    event.preventDefault();
    voicingBrowserIndex = Math.min(voicingBrowserIndex + 1, voicingBrowserList.length - 1);
    playBrowsedVoicing();
  } else if (event.key === 'ArrowLeft') {
    event.preventDefault();
    voicingBrowserIndex = Math.max(voicingBrowserIndex - 1, 0);
    playBrowsedVoicing();
  }
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
    chordSynth.triggerRelease(voicingBrowserActivePitches.map(pitch => pitchToNote[pitch]));
    voicingBrowserActivePitches.forEach(pitch => visual.noteOff(pitch));
    voicingBrowserActivePitches = [];
  }

  const beatsPerMeasure = Tone.Transport.timeSignature;
  const barSeconds = (60 / Tone.Transport.bpm.value) * beatsPerMeasure;
  const noteNames = entry.pitches.map(pitch => pitchToNote[pitch]);
  chordSynth.triggerAttackRelease(noteNames, barSeconds);

  // Highlight on the piano roll for the same duration as the audio -- plain
  // setTimeout (not Tone.Transport) since this tool runs outside any
  // session/Transport scheduling.
  entry.pitches.forEach(pitch => visual.noteOn(pitch));
  voicingBrowserActivePitches = entry.pitches;
  voicingBrowserOffTimeout = setTimeout(() => {
    entry.pitches.forEach(pitch => visual.noteOff(pitch));
    voicingBrowserActivePitches = [];
    voicingBrowserOffTimeout = null;
  }, barSeconds * 1000);

  const detail = entry.legacy
    ? '(legacy fixed-octave voicing -- not covered by the custom voicing lookup)'
    : `(seen ${entry.count} times across ${entry.num_songs} songs)`;
  voicingBrowserInfo.textContent =
    `Voicing ${voicingBrowserIndex + 1} / ${voicingBrowserList.length} -- ` +
    `${noteNames.join(', ')} ${detail}`;
}

/**
 * Play melody note, add to history and start live session loop if needed
 * @param {string} note
 * @param {number} velocity
 */
function playNote(note, velocity) {
  const keyIndex = noteToPitch[note];
  if (curSession) {
    curSession.noteHistory.push(
      { on: true, pitch: keyIndex, frame: getSessionCurrentFrame() });
    startGenerationLoop();
  }
  melodySynth.triggerAttack(note, '+0', velocity);
  visual.noteOn(keyIndex);
}

/**
 * Release melody note and add to history
 * @param {string} note
 */
function releaseNote(note) {
  const keyIndex = noteToPitch[note];
  if (curSession) {
    curSession.noteHistory.push(
      { on: false, pitch: keyIndex, frame: getSessionCurrentFrame() });
  }
  melodySynth.triggerRelease(note);
  visual.noteOff(keyIndex);
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
  if (on) {
    eventID = Tone.Transport.scheduleOnce(time => {
      chordSynth.triggerAttack(note, '+0', chordVelocity);
      Tone.Draw.schedule(() => {
        visual.noteOn(keyIndex, 'blue');
      }, time);
    }, playTime);
  } else {
    eventID = Tone.Transport.scheduleOnce(time => {
      chordSynth.triggerRelease(note);
      Tone.Draw.schedule(() => {
        visual.noteOff(keyIndex);
      }, time);
    }, playTime);
  }
  return eventID;
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
      playNote(note, melodyVelocity);
    };

    const releaseKey = () => {
      index = key.getAttribute('data-index');
      note = pianoNotes[index];
      releaseNote(note);
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
 * (in manual chord mode) every space would fire a chord.
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
    // Space advances the chord in manual timing mode. Swallow it so it
    // doesn't also activate whatever button happens to have focus.
    if (event.code === 'Space' && curSession && isManualChordMode()) {
      event.preventDefault();
      if (!event.repeat) {
        advanceChordNow();
      }
      return;
    }
    const note = keysToNotes[event.key];
    if (note && !heldNotes[note]) {
      heldNotes[note] = true;
      playNote(note, melodyVelocity);
    }
  });
  document.addEventListener('keyup', event => {
    if (isTypingTarget(event)) {
      return;
    }
    const note = keysToNotes[event.key];
    if (note) {
      heldNotes[note] = false;
      releaseNote(note);
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
function enableMIDIInputs() {
  if (WebMidi.inputs.length < 1) {
    console.log('No MIDI device detected.');
    webMIDIInputs = [];
    interfaceSelect.style = 'display: none';
  } else {
    console.log('MIDI devices detected.');
    webMIDIInputs = WebMidi.inputs;
    webMIDIInputs.forEach(input => {
      const name = document.createTextNode(input.name);
      const option = document.createElement('option');
      option.append(name);
      option.value = input.name;
      interfaceSelect.appendChild(option);
    });
    selectMIDIInterface(webMIDIInputs[0].name);
  }
}

/**
 * Select active MIDI input device
 * @param {string} name - dropdown value associated with input
 */
function selectMIDIInterface(name) {
  interfaceSelect.value = name;

  webMIDIInputs.forEach(input => input.removeListener());
  let curInterface = webMIDIInputs.find(input => input.name === name);

  // Display the note name and play the note
  curInterface.channels[1].addListener('noteon', e => {
    playNote(e.note.identifier, e.velocity);
  });
  curInterface.channels[1].addListener('noteoff', e => {
    releaseNote(e.note.identifier);
  });
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
  liveSessionBtn = document.getElementById('live-session-button');
  liveSessionBtn.addEventListener('click', toggleLiveSession);
  downloadSessionCheck = document.getElementById('download-session-check');
  metronomeCheck = document.getElementById('metronome-check');
  showChordsCheck = document.getElementById('show-chords-check');
  customVoicingsCheck = document.getElementById('custom-voicings-check');
  vlWeightInput = bindSliderValueDisplay('vl-weight-input', 'vl-weight-value', 2);
  regWeightInput = bindSliderValueDisplay('reg-weight-input', 'reg-weight-value', 2);
  songSearchInput = document.getElementById('song-search-input');
  songSearchInput.addEventListener('input', onSongSearchInput);
  songSearchInput.addEventListener('keydown', onSongSearchKeydown);
  songSearchResults = document.getElementById('song-search-results');
  referenceModeCheck = document.getElementById('reference-mode-check');
  chordTimingSelect = document.getElementById('chord-timing-select');
  chordTimingInfo = document.getElementById('chord-timing-info');
  chordTimingSelect.addEventListener('change', () => {
    updateChordTimingInfo();
    chordTimingSelect.blur();
  });
  voicingBrowserChordSelect = document.getElementById('voicing-browser-chord-select');
  voicingBrowserChordSelect.addEventListener('change', onVoicingBrowserChordSelected);
  voicingBrowserChordSelect.addEventListener('keydown', onVoicingBrowserKeydown);
  voicingBrowserInfo = document.getElementById('voicing-browser-info');
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
  lookaheadInput = document.getElementById('lookahead-input');
  lookaheadInput.value = DEFAULTS.lookahead;
  setLookahead(DEFAULTS.lookahead);
  addNumericInputEventListener(lookaheadInput, setLookahead);
  chordInstSelect = document.getElementById('chord-inst-select');
  melodyInstSelect = document.getElementById('melody-inst-select');
  setupInstrumentSelection();

  // Do initial setup with server
  establishServerConnection();
  updateChordTimingInfo();

  console.log('ready!');
  playBtn.textContent = 'Play';
  playBtn.removeAttribute('disabled');
  playBtn.classList.remove('loading');
}