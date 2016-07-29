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

    cmd = ["yaafe", "-c", FEATURE_PLAN, "-r", "22050", args.input_file]

    subprocess.check_output(cmd)

    features1 = ["zcr", "flux", "spectral_rollof", "energy_stats"]
    features2 = ["mfcc_stats"]
    features3 = ["spectral_flatness_per_band"]
    features4 = features1 + features2 + features3

    FEATURE_GROUPS = [features1, features2, features3, features4]

    peaks, convolution_values, timestamps = feat.get_combined_peaks(args.input_file, FEATURE_GROUPS,
                                                                    kernel_type="gaussian")
    detected_segments = kernel.calculate_segment_start_end_times_from_peak_positions(peaks, timestamps)

    timestamps, feature_vectors = feat.read_features(features4, args.input_file, scale=True)

    with open("/opt/speech-music-discrimination/pickled/model.pickle", 'r') as f:
        trained_model = pickle.load(f)

    frame_level_predictions = trained_model.predict(feature_vectors)

    annotated_segments = Util.get_annotated_labels_from_predictions_and_sm_segments(frame_level_predictions,
                                                                            detected_segments,
                                                                            timestamps)

    annotated_segments = Util.combine_adjacent_labels_of_the_same_class(annotated_segments)
    annotated_segments = feat.filter_noisy_labels(annotated_segments)
    annotated_segments = Util.combine_adjacent_labels_of_the_same_class(annotated_segments)

    Util.write_audacity_labels(annotated_segments, os.path.split(args.input_file)[0] + "/annotated-segments.txt")

    for f in glob.glob(os.path.split(args.input_file)[0] + "/*.csv"):
        os.remove(f)


if __name__ == '__main__':
    main()
