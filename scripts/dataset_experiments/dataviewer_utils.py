import pretty_midi
from scipy.interpolate import interp1d
from typing import Any


"""Convert a Hooktheory song to midi and write to file.
        Args:
            example: The dictionary containing one Hooktheory song.
             
"""
def hooktheory_to_midi(example: dict[str, Any], chord_octave: int = 4, melody_octave: int = 5) -> None: 

    # map a given beat to the absolute time
    beat_to_time_fn = interp1d(
    example['alignment']['refined']['beats'],   # how many beats in the song
    example['alignment']['refined']['times'],   # at what absolute time [s] the beats occur
    kind='linear',
    fill_value='extrapolate') 
    start_time = beat_to_time_fn(0)
    num_beats = example['annotations']['num_beats'] 
    end_time = beat_to_time_fn(num_beats)

    CHORD_OCTAVE = chord_octave
    MELODY_OCTAVE = melody_octave

    midi = pretty_midi.PrettyMIDI()

    # Create click track
    # click = pretty_midi.Instrument(program=0, is_drum=True)
    # midi.instruments.append(click)
    # beats_per_bar = example['annotations']['meters'][0]['beats_per_bar']
    # for b in range(num_beats + 1):
    #     downbeat = b % beats_per_bar == 0
    #     click.notes.append(pretty_midi.Note(
    #         100 if downbeat else 75, 
    #         37 if downbeat else 31,
    #         beat_to_time_fn(b),
    #         beat_to_time_fn(b + 1)))

    # Create harmony track
    harmony = pretty_midi.Instrument(program=0)
    midi.instruments.append(harmony)
    for c in example['annotations']['harmony']:
        root_position_pitches = [c['root_pitch_class']]
        for interval in c['root_position_intervals']:
            root_position_pitches.append(root_position_pitches[-1] + interval) # Put all notes of the chord into the list
        for p in root_position_pitches: # Write the notes of the chord to the midi track
            harmony.notes.append(pretty_midi.Note(  # Note(velocity, pitch, onset, offset)
                67,
                p + CHORD_OCTAVE * 12,
                beat_to_time_fn(c['onset']),
                beat_to_time_fn(c['offset'])
            ))

    # Create melody track
    melody = pretty_midi.Instrument(program=0)
    midi.instruments.append(melody)
    for n in example['annotations']['melody']:
        melody.notes.append(pretty_midi.Note(     # Note(velocity, pitch, onset, offset)
            100,
            n['pitch_class'] + (MELODY_OCTAVE + n['octave']) * 12,
            beat_to_time_fn(n['onset']),
            beat_to_time_fn(n['offset'])
        ))

    midi.write(f"midis/annotations_{example['hooktheory']['song']}_{example['hooktheory']['id']}.midi")


    # Synthesize aligned preview
    # annotations_audio = midi.fluidsynth(fs=sr)
    # annotations_audio = annotations_audio[round(start_time * sr):]
    # annotations_audio = annotations_audio[:audio.shape[0]]
    # if annotations_audio.shape[0] < audio.shape[0]:
    #     annotations_audio = np.pad(annotations_audio, [(0, audio.shape[0] - annotations_audio.shape[0])])
    # display(Audio([audio, annotations_audio], rate=sr))