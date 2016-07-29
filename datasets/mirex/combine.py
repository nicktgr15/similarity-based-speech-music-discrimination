import glob
import os
import random
import shutil
import subprocess

from sac.model.audacity_label import AudacityLabel
from sac.util import Util
from sac.cli.wav_editor import WavEditor


def mp3_to_wav():
    mp3s = glob.glob("./muspeak*/*.mp3")
    for mp3 in mp3s:
        subprocess.check_output(["sox", mp3, "-r", "22050", "-c", "1", mp3+".wav"])


def all_same(items):
    return all(x[2] == items[0][2] for x in items)


def find_min_start(items):
    min = float('Inf')
    for i in items:
        if i[0] <= min:
            min = i[0]
    return min


def find_max_end(items):
    max = -float('Inf')
    for i in items:
        if i[1] >= max:
            max = i[1]
    return max


def chunks(l, n):
    c = []
    for i in range(0, len(l), n):
        c.append(l[i:i+n] + ["m"])
    return c


def not_in_overlapping(item, overlapping_items):
    for i in overlapping_items:
        for j in i:
            if item == j:
                return False
    return True

def main():

    subprocess.check_output(["unzip", "muspeak-mirex2015-detection-examples.zip", "-d",
                             "muspeak-mirex2015-detection-examples"])

    mp3_to_wav()

    wav_files = glob.glob("./muspeak-mirex2015-detection-examples/*.wav")

    for wav_file in wav_files:
        print wav_file
        label_file = wav_file.replace(".mp3.wav", ".csv")
        if not os.path.isfile(label_file):
            label_file = label_file.replace(".csv", "_v2.csv")
        WavEditor.create_audio_segments(label_file, wav_file, "segments", True, ",", "f2", remove_overlapping=True)

    speech_wavs = glob.glob("./segments/*_s.wav")
    music_wavs = glob.glob("./segments/*_m.wav")

    all_files_dict = {}

    for f in speech_wavs:
        all_files_dict[f] = "s"

    for f in music_wavs:
        all_files_dict[f] = "m"

    random.seed(2222)
    all_files_random_keys = random.sample(all_files_dict.keys(), len(all_files_dict.keys()))

    last_seconds = 0
    files_to_concatenate = []

    labels = []
    for v in all_files_random_keys:
        duration = float(subprocess.check_output(["soxi", "-D", v]).strip())
        segment_start_time = last_seconds
        segment_end_time = last_seconds + duration
        last_seconds += duration
        labels.append(AudacityLabel(segment_start_time, segment_end_time, all_files_dict[v]))
        files_to_concatenate.append(v)

    audacity_labels = Util.combine_adjacent_labels_of_the_same_class(labels)
    Util.write_audacity_labels(audacity_labels, "mirex_combined.txt")

    command = []
    command.append("sox")
    command.extend(files_to_concatenate)
    command.append("mirex_combined.wav")
    subprocess.check_output(command)

    shutil.rmtree("./segments")
    shutil.rmtree("./muspeak-mirex2015-detection-examples")

if __name__ == '__main__':
    main()
