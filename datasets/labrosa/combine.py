import glob
import shutil
import subprocess
import random

from sac.model.audacity_label import AudacityLabel
from sac.util import Util

subprocess.check_output(["tar", "-zxf", "music-speech-20100223.tgz"])

music_no_vocals_test = glob.glob("./music-speech/wavfile/test/music/novocals/*.wav")
music_vocals_test = glob.glob("./music-speech/wavfile/test/music/vocals/*.wav")
speech_test = glob.glob("./music-speech/wavfile/test/speech/*.wav")

music_train = glob.glob("./music-speech/wavfile/train/music/*.wav")
speech_train = glob.glob("./music-speech/wavfile/train/speech/*.wav")

speech_wavs = speech_test + speech_train
music_wavs = music_train + music_no_vocals_test + music_vocals_test

# TODO: music_and_speech

all_files_dict = {}

for f in speech_wavs:
    all_files_dict[f] = "s"

for f in music_wavs:
    all_files_dict[f] = "m"

random.seed(1111)
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
Util.write_audacity_labels(audacity_labels, "labrosa_combined.txt")

command = []
command.append("sox")
command.extend(files_to_concatenate)
command.append("labrosa_combined.wav")
subprocess.check_output(command)

subprocess.call(['chmod', '-R', '777', './music-speech'])
shutil.rmtree("./music-speech")
