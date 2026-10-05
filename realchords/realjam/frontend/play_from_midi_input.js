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
  customVoicingsCheck, voiceAroundMelodyCheck, voiceBassCheck, melodyGapInput, songSearchInput,
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
  enableMIDI();
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
    triggeredChord: null,   // Triggered mode: chord sustaining since the
                             // last space press
    voicedByFrame: new Map(),  // Auto mode, Voice Around Melody: frame ->
                                // {key, voiced}, see autoModeVoicing
    candidates: null,       // Complete mode: model's ranked chord onsets
    heldKeys: new Set(),    // Complete mode: keys the performer is holding
    gestureTimer: null,     // Complete mode: gathering a chord gesture
    completion: null,       // Complete mode: completed chord sounding
    completionHistory: [],  // Complete mode: {frame, held, symbol, pitches,
                             // prob, rank} per completed chord
    triggerHistory: [],     // Triggered mode: {frame, symbol, pitches} per
                             // press, frame relative to session start
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
  if ((isManualChordMode() || isCompleteMode()) && !curSession.referenceOnly) {
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
      if (!midiChordOut) melodySynth.triggerAttack(note, '+0', melodyVelocity);
      visual.noteOn(pitch);
    } else {
      if (!midiChordOut) melodySynth.triggerRelease(note);
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
 * If chords follow auto mode's schedule but only sound when the performer
 * presses space. The model runs exactly as in auto mode -- same polling,
 * lookahead, commit and chord history, never told about the presses -- so
 * the schedule is the one auto mode would play; a press plays whatever that
 * schedule has sounding at that moment. See triggerScheduledChord.
 * @return {boolean}
 */
function isTriggeredChordMode() {
  return chordTimingSelect.value === 'triggered' &&
    !(curSession && curSession.referenceOnly);
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
  return chordTimingSelect.value === 'complete' &&
    !(curSession && curSession.referenceOnly);
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
  } else if (isCompleteMode()) {
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
      vlWeight: CUSTOM_VOICING_VL_WEIGHT,
      regWeight: CUSTOM_VOICING_REG_WEIGHT,
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
      vlWeight: CUSTOM_VOICING_VL_WEIGHT,
      regWeight: CUSTOM_VOICING_REG_WEIGHT,
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

  const chord = { ...curSession.pendingChord };
  chord.pitches = voiceAroundMelody(chord.pitches,
    curSession.manualChord && curSession.manualChord.pitches);
  const frame = getSessionCurrentFrame();

  if (curSession.manualChord) {
    curSession.manualChord.pitches.forEach(pitch => {
      if (!midiChordOut) chordSynth.triggerRelease(pitchToNote[pitch]);
      visual.noteOff(pitch);
    });
  }
  chord.pitches.forEach(pitch => {
    if (!midiChordOut) chordSynth.triggerAttack(pitchToNote[pitch], '+0', chordVelocity);
    visual.noteOn(pitch, 'blue');
  });
  midiPlayNow(chord.pitches);

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

// Custom-voicing scoring weights; were sliders, fixed at their old defaults
const CUSTOM_VOICING_VL_WEIGHT = 0.5;
const CUSTOM_VOICING_REG_WEIGHT = 0.2;

/**
 * The melody note to voice around: the highest one sounding now, else the
 * last one played, else middle C.
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

/**
 * Voice a chord like a pianist's two hands, just below the melody (auto,
 * triggered and manual mode, Voice Around Melody on):
 * - Upper chord: three notes, each pitch class once, in close position, in
 *   whichever inversion fits best -- its top as close under the ceiling
 *   (melody - gap) as possible, and moving as little as possible from
 *   `prevPitches` -- so successive chords take varied positions. Chords with
 *   more notes keep the most characteristic ones (VOICE_AROUND_PRIORITY:
 *   3rds, 7ths, 6th, 9ths, ..., the fifth last); triads use root, 3rd, 5th.
 * - Bass (Bass Note on): the given voicing's lowest pitch class -- the root,
 *   or a slash chord's bass -- at least a fourth below the upper chord.
 * Returns the given voicing unchanged when the option is off.
 * @param {Array<number>} pitches - Voicing from the server
 * @param {Array<number>=} prevPitches - The previous chord's voicing
 * @return {Array<number>}
 */
function voiceAroundMelody(pitches, prevPitches) {
  if (!voiceAroundMelodyCheck.checked || !pitches || !pitches.length) return pitches;
  const gapValue = melodyGapInput.valueAsNumber;
  const gap = Number.isNaN(gapValue) ? 3 : Math.max(0, gapValue);
  const ceiling = currentMelodyPitch() - gap;
  const mod = (x, m) => ((x % m) + m) % m;
  const bassPc = Math.min(...pitches) % 12;

  // Upper pitch classes: the most characteristic intervals above the bass,
  // topped up with the bass pitch class itself for a triad
  const intervals = [...new Set(pitches.map(p => mod(p - bassPc, 12)))]
    .filter(i => i !== 0)
    .sort((x, y) => VOICE_AROUND_PRIORITY.indexOf(x) - VOICE_AROUND_PRIORITY.indexOf(y))
    .slice(0, VOICE_AROUND_UPPER_NOTES);
  const upperPcs = intervals.map(i => (bassPc + i) % 12);
  if (upperPcs.length < VOICE_AROUND_UPPER_NOTES) upperPcs.push(bassPc);

  // Every inversion in close position, each as high as fits under the ceiling
  const prevUpper = prevPitches && prevPitches.length > 1
    ? prevPitches.filter(p => p !== Math.min(...prevPitches)) : prevPitches || [];
  let best = null;
  upperPcs.forEach(lowestPc => {
    const others = upperPcs.filter(pc => pc !== lowestPc)
      .map(pc => mod(pc - lowestPc, 12)).sort((x, y) => x - y);
    const span = others.length ? others[others.length - 1] : 0;
    const low = ceiling - span - mod(ceiling - span - lowestPc, 12);
    const voicing = [low, ...others.map(i => low + i)];
    const fromCeiling = ceiling - voicing[voicing.length - 1];
    const movement = prevUpper.length
      ? voicing.reduce((sum, p) => sum + Math.min(...prevUpper.map(q => Math.abs(p - q))), 0)
      : 0;
    const score = fromCeiling + VOICE_AROUND_MOVEMENT_WEIGHT * movement;
    if (!best || score < best.score) best = { voicing, score };
  });
  let out = best.voicing;

  if (voiceBassCheck.checked) {
    const lowest = Math.min(...out);
    const bass = (lowest - 5) - mod(lowest - 5 - bassPc, 12);  // a 4th or more below
    if (bass >= LOWEST_PIANO_PITCH) out = [bass, ...out];
  }
  while (Math.min(...out) < LOWEST_PIANO_PITCH) out = out.map(p => p + 12);
  return out.sort((x, y) => x - y);
}
const VOICE_AROUND_UPPER_NOTES = 3;
// Intervals above the root, most characteristic first: thirds, sevenths,
// sixth, ninths, fourth / tritone, the fifth last (it adds the least colour)
const VOICE_AROUND_PRIORITY = [4, 3, 10, 11, 9, 2, 1, 5, 6, 8, 7];
// Semitones of top-note distance from the melody worth one semitone of
// voice movement from the previous chord
const VOICE_AROUND_MOVEMENT_WEIGHT = 0.5;

/**
 * Voice Around Melody for chords the model plans in auto (and triggered)
 * mode. A chord is voiced when it first appears in the plan, around the
 * melody note sounding then (or last played); when later polls re-plan the
 * same chord for that frame it keeps that voicing, so a chord whose MIDI
 * may already be on its way is never re-voiced. A hold of the same chord
 * keeps the voicing of the chord it holds. Only a changed chord is voiced
 * afresh, from the melody at that moment.
 * @return {Array<number>}
 */
function autoModeVoicing(frame, symbol, on, pitches) {
  if (!voiceAroundMelodyCheck.checked || !pitches || !pitches.length) return pitches;
  const key = `${on}|${symbol}|${pitches.join(',')}`;
  const prior = curSession.voicedByFrame.get(frame);
  if (prior && prior.key === key) return prior.voiced;
  let voiced;
  const held = !on && getScheduledChordAt(frame - 1);
  if (held && held.symbol === symbol && held.pitches.length) {
    voiced = held.pitches;
  } else {
    const prev = getScheduledChordAt(frame - 1);
    voiced = voiceAroundMelody(pitches, prev && prev.pitches);
  }
  curSession.voicedByFrame.set(frame, { key, voiced });
  return voiced;
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
function triggerScheduledChord() {
  const entry = getScheduledChordAt(Math.floor(getTransportFrame()));
  const pitches = voiceAroundMelody(entry ? entry.pitches || [] : [],
    curSession.triggeredChord || curSession.lastTriggered);

  if (curSession.triggeredChord) {
    curSession.triggeredChord.forEach(pitch => {
      if (!midiChordOut) chordSynth.triggerRelease(pitchToNote[pitch]);
      visual.noteOff(pitch);
    });
  }
  pitches.forEach(pitch => {
    if (!midiChordOut) chordSynth.triggerAttack(pitchToNote[pitch], '+0', chordVelocity);
    visual.noteOn(pitch, 'blue');
  });
  midiPlayNow(pitches);

  curSession.triggeredChord = pitches;
  if (pitches.length) curSession.lastTriggered = pitches;
  curSession.triggerHistory.push({
    frame: getSessionCurrentFrame(),
    symbol: entry ? entry.symbol : '',
    pitches,
  });
  updateChordTimingInfo();
}

const COMPLETION_WINDOW_MS = 50;  // notes this close together form one gesture

/** Fetch the model's ranked chord onsets for the next frame */
async function fetchCandidates() {
  if (!curSession) return;
  fillManualChordTokens();
  try {
    const result = await fetch(`${window.location.origin}/chord_candidates`, {
      'method': 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: modelSelect.value,
        notes: [],  // the performer's notes are constraints, not melody
        chordTokens: curSession.chordTokens,
        frame: getSessionCurrentFrame() + 1,
      })
    });
    const candidates = await result.json();
    if (curSession) curSession.candidates = candidates;
  } catch (e) {
    console.warn('Could not fetch chord candidates', e);
  }
}

/** Keep the candidates fresh, once per beat, while waiting for gestures */
async function candidateLoop() {
  if (!curSession || !isCompleteMode()) return;
  await fetchCandidates();
  if (!curSession || !isCompleteMode()) return;
  setTimeout(candidateLoop, 60000 / Tone.Transport.bpm.value);
}

/** Performer pressed a key in complete mode */
function onCompletionNoteDown(pitch) {
  curSession.heldKeys.add(pitch);
  // A chord is already sounding, or this key joins a gesture being gathered
  if (curSession.completion || curSession.gestureTimer) return;
  curSession.gestureTimer = setTimeout(completeHeldNotes, COMPLETION_WINDOW_MS);
}

/** Performer released a key; the chord ends with the last one */
function onCompletionNoteUp(pitch) {
  curSession.heldKeys.delete(pitch);
  if (!curSession.heldKeys.size && curSession.completion) {
    releaseCompletion();
  }
}

/**
 * Sample a chord among the candidates containing every pitch class held (or,
 * if none does, those sharing the most), weighted by the model's
 * probability sharpened or flattened by the Temperature field: p^(1/T).
 * Temperature 0 always takes the most probable.
 * @return {?{chord: Object, rank: number}}
 */
function pickCandidate(candidates, held) {
  const need = new Set(held.map(p => p % 12));
  const scored = candidates.map((chord, rank) => ({
    chord, rank, overlap: chord.pitchClasses.filter(pc => need.has(pc)).length,
  }));
  const bestOverlap = Math.max(...scored.map(c => c.overlap));
  const compatible = scored.filter(c => c.overlap === bestOverlap);
  if (!compatible.length) return null;
  const temperature = temperatureInput.valueAsNumber;
  if (!(temperature > 0)) return compatible[0];  // candidates are sorted
  const weights = compatible.map(c => Math.pow(c.chord.prob, 1 / temperature));
  let r = Math.random() * weights.reduce((a, b) => a + b, 0);
  for (let i = 0; i < compatible.length; i++) {
    r -= weights[i];
    if (r <= 0) return compatible[i];
  }
  return compatible[compatible.length - 1];
}

/**
 * The chord tones the performer isn't playing, placed around their hand,
 * plus the root in the bass unless they're already playing it lowest.
 * Never a key the performer is holding.
 * @param {Object} chord - Candidate with pitchClasses and root
 * @param {Array<number>} held
 * @return {Array<number>}
 */
function completeAround(chord, held) {
  const heldSet = new Set(held);
  const heldPcs = new Set(held.map(p => p % 12));
  const low = Math.min(...held), high = Math.max(...held);
  const center = (low + high) / 2;
  const out = [];
  chord.pitchClasses.forEach(pc => {
    if (heldPcs.has(pc)) return;
    let best = null;
    for (let p = low - 12; p <= high; p++) {
      if (p % 12 === pc && !heldSet.has(p) &&
          (best === null || Math.abs(p - center) < Math.abs(best - center))) {
        best = p;
      }
    }
    if (best !== null) out.push(best);
  });
  if (low % 12 !== chord.root) {
    let bass = low - 1;
    while (bass % 12 !== chord.root) bass--;
    if (low - bass < 12) bass -= 12;
    if (bass < LOWEST_PIANO_PITCH) bass += 12;
    if (!heldSet.has(bass) && !out.includes(bass)) out.push(bass);
  }
  return out.sort((a, b) => a - b);
}
const LOWEST_PIANO_PITCH = 21;

/** Complete the notes gathered in the gesture window */
async function completeHeldNotes() {
  if (!curSession) return;
  curSession.gestureTimer = null;
  if (!curSession.candidates) {
    await fetchCandidates();  // session just started: take the round trip once
  }
  if (!curSession || !curSession.candidates || !curSession.heldKeys.size) {
    return;  // stopped, no candidates, or released before they arrived
  }
  const held = [...curSession.heldKeys];
  const picked = pickCandidate(curSession.candidates, held);
  if (!picked) return;
  const { chord, rank } = picked;
  const pitches = completeAround(chord, held);

  pitches.forEach(pitch => {
    if (!midiChordOut) chordSynth.triggerAttack(pitchToNote[pitch], '+0', chordVelocity);
    visual.noteOn(pitch, 'blue');
  });
  midiStrikeNow(pitches);

  const frame = getSessionCurrentFrame();
  fillManualChordTokens();
  curSession.chordTokens[frame] = chord.onsetToken;
  curSession.manualChord = { ...chord, pitches, startFrame: frame };
  curSession.completion = { pitches };
  curSession.chordHistory.push(
    { scheduleFrame: frame, pitches, symbol: chord.symbol, eventIDs: [] });
  curSession.completionHistory.push(
    { frame, held, symbol: chord.symbol, pitches, prob: chord.prob, rank });
  updateChordTimingInfo();
}

/** All keys released: the completed chord ends */
function releaseCompletion() {
  fillManualChordTokens();     // the chord held up to now
  curSession.manualChord = null;
  curSession.completion.pitches.forEach(pitch => {
    if (!midiChordOut) chordSynth.triggerRelease(pitchToNote[pitch]);
    visual.noteOff(pitch);
  });
  midiStrikeToken++;           // cancel a re-press midiStrikeNow may have pending
  midiApplyChord([], Tone.context.currentTime);
  curSession.completion = null;
  fetchCandidates();           // history changed
}

/** Space let go in triggered mode: the chord stops */
function releaseTriggeredChord() {
  if (!curSession.triggeredChord) return;
  curSession.triggeredChord.forEach(pitch => {
    if (!midiChordOut) chordSynth.triggerRelease(pitchToNote[pitch]);
    visual.noteOff(pitch);
  });
  midiStrikeToken++;  // cancel a re-press midiStrikeNow may still have pending
  midiApplyChord([], Tone.context.currentTime);
  curSession.triggeredChord = null;
  curSession.triggerHistory.push(
    { frame: getSessionCurrentFrame(), symbol: '', pitches: [], release: true });
  updateChordTimingInfo();
}

/** Show what is sounding and what space will play next */
function updateChordTimingInfo() {
  if (chordTimingSelect.value === 'complete') {
    if (!curSession) {
      chordTimingInfo.textContent =
        'Start a session, then play a few notes: the model completes the chord';
      return;
    }
    const last = curSession.completionHistory[curSession.completionHistory.length - 1];
    chordTimingInfo.textContent = last
      ? `Completed: ${last.symbol} (model's #${last.rank + 1} choice)` +
        `${curSession.completion ? '' : ' -- released'}`
      : (curSession.candidates ? 'Play a few notes' : 'thinking...');
    return;
  }
  if (chordTimingSelect.value === 'triggered') {
    if (!curSession) {
      chordTimingInfo.textContent =
        'Start a session; hold space to play the chord scheduled at that moment';
      return;
    }
    const last = curSession.triggerHistory[curSession.triggerHistory.length - 1];
    const now = getScheduledChordAt(Math.floor(getTransportFrame()));
    chordTimingInfo.textContent =
      `Playing: ${last && last.symbol || '--'}   Space plays: ${now && now.symbol || '--'}`;
    return;
  }
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

  if (!curSession.voicedByFrame) curSession.voicedByFrame = new Map();
  curSession.voicedByFrame.forEach((_, frame) => {
    if (frame < curFrame - 64) curSession.voicedByFrame.delete(frame);
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
    pitches = autoModeVoicing(scheduleFrame, symbol, on, pitches);

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
function playNote(note, velocity, fromLaptop = false) {
  const keyIndex = noteToPitch[note];
  const completing = curSession && isCompleteMode();
  if (curSession && !completing) {
    curSession.noteHistory.push(
      { on: true, pitch: keyIndex, frame: getSessionCurrentFrame() });
    startGenerationLoop();
  }
  // With a MIDI output the piano sounds the melody. Notes played on the
  // laptop (keys or mouse) are pressed on the piano; notes coming from a MIDI
  // input are not sent back, since on a Disklavier that input *is* the piano
  // and the key is already down.
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
      { on: false, pitch: keyIndex, frame: getSessionCurrentFrame() });
  }
  if (!midiChordOut) {
    melodySynth.triggerRelease(note);
  } else if (fromLaptop) {
    midiKeyUp(keyIndex, 'laptop', Tone.context.currentTime);
  }
  visual.noteOff(keyIndex);
  if (completing) onCompletionNoteUp(keyIndex);
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
      if (!midiChordOut) chordSynth.triggerAttack(note, '+0', chordVelocity);
      Tone.Draw.schedule(() => {
        visual.noteOn(keyIndex, 'blue');
      }, time);
    }, playTime);
  } else {
    eventID = Tone.Transport.scheduleOnce(time => {
      if (!midiChordOut) chordSynth.triggerRelease(note);
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
  }
  holders.add(holder);
  midiKeyHolders.set(pitch, holders);
}

function midiKeyUp(pitch, holder, audioTime) {
  const holders = midiKeyHolders.get(pitch);
  if (!holders || !holders.delete(holder)) return;
  if (holders.size === 0) {
    midiChordOut.send([0x80, pitch, 0], { time: midiTimestamp(audioTime) });
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
 * @return {?Array<{pitch: number, offFrame: number}>} each note's release
 *   with the current settings, for drawing (null without a MIDI output)
 */
function scheduleMidiMelody(notes, baseFrame, loopFrames, redraw = false) {
  if (!midiChordOut) return null;
  const lead = (midiLeadInput.valueAsNumber || 0) / 1000;
  const now = Tone.Transport.seconds;
  const spf = Tone.Time('16n').toSeconds();
  const toSec = beats => Tone.Time(frameToTransportTime(
    baseFrame + Math.round(beats * fpb))).toSeconds();

  const timed = notes
    .map(({ pitch, onset, offset }) => ({ pitch, on: toSec(onset), off: toSec(offset) }))
    .sort((a, b) => a.on - b.on);
  const upcoming = loopFrames
    ? [...timed, ...timed.map(n => ({ ...n, on: n.on + loopFrames * spf }))]
    : timed;

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
 * Sound a chord the performer just triggered (triggered and manual mode)
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
    if (event.code === 'Space' && curSession && isTriggeredChordMode()) {
      event.preventDefault();
      if (!event.repeat) {
        triggerScheduledChord();
      }
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
    return;
  }
  // Chrome asks for MIDI permission via an icon in the address bar; until
  // it's answered, enable() stays pending
  showMIDIStatus('Waiting for MIDI permission…');
  WebMidi.enable()
    .then(() => {
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
  voiceAroundMelodyCheck = document.getElementById('voice-around-melody-check');
  voiceBassCheck = document.getElementById('voice-bass-check');
  melodyGapInput = document.getElementById('melody-gap-input');
  addNumericInputEventListener(melodyGapInput);
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