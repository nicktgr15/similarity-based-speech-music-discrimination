import glob
import shutil
import subprocess
import random

from sac.model.audacity_label import AudacityLabel
from sac.util import Util


subprocess.check_output(["tar", "-zxf", "music_speech.tar.gz"])

speech_wavs = glob.glob("./music_speech/speech_wav/*.wav")
music_wavs = glob.glob("./music_speech/music_wav/*.wav")

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
Util.write_audacity_labels(audacity_labels, "gtzan_combined.txt")

command = []
command.append("sox")
command.extend(files_to_concatenate)
command.append("gtzan_combined.wav")
subprocess.check_output(command)

subprocess.call(['chmod', '-R', '777', './music_speech'])
shutil.rmtree("./music_speech")
