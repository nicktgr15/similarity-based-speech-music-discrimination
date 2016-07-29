import pickle

import feat
import os
from sac.methods.sm_analysis import kernel
from sac.util import Util
import subprocess
import argparse
import glob

FEATURE_PLAN = "/opt/speech-music-discrimination/featureplan"


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-file', dest='input_file', required=True)
    args = parser.parse_args()

    input_dir = os.path.split(args.input_file)[0]
    temp_file = input_dir + "/temp.wav"

    cmd = ["/usr/bin/ffmpeg", "-i", args.input_file, "-ar", "22050", "-ac", "1", "-acodec", "pcm_s16le",
           temp_file, "-y"]
    subprocess.check_call(cmd)

    cmd = ["yaafe", "-c", FEATURE_PLAN, "-r", "22050", temp_file]

    subprocess.check_output(cmd)

    features1 = ["zcr", "flux", "spectral_rollof", "energy_stats"]
    features2 = ["mfcc_stats"]
    features3 = ["spectral_flatness_per_band"]
    features4 = features1 + features2 + features3

    FEATURE_GROUPS = [features1, features2, features3, features4]

    peaks, convolution_values, timestamps = feat.get_combined_peaks(temp_file, FEATURE_GROUPS,
                                                                    kernel_type="gaussian")
    detected_segments = kernel.calculate_segment_start_end_times_from_peak_positions(peaks, timestamps)

    timestamps, feature_vectors = feat.read_features(features4, temp_file, scale=True)

    with open("/opt/speech-music-discrimination/pickled/model.pickle", 'r') as f:
        trained_model = pickle.load(f)

    frame_level_predictions = trained_model.predict(feature_vectors)

    annotated_segments = Util.get_annotated_labels_from_predictions_and_sm_segments(frame_level_predictions,
                                                                            detected_segments,
                                                                            timestamps)

    annotated_segments = Util.combine_adjacent_labels_of_the_same_class(annotated_segments)
    annotated_segments = feat.filter_noisy_labels(annotated_segments)
    annotated_segments = Util.combine_adjacent_labels_of_the_same_class(annotated_segments)

    Util.write_audacity_labels(annotated_segments, input_dir + "/annotated-segments.txt")

    for f in glob.glob(input_dir + "/*.csv"):
        os.remove(f)

    os.remove(temp_file)


if __name__ == '__main__':
    main()
