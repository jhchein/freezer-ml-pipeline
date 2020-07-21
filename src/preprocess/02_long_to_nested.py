import argparse
import os

import pandas as pd

from sktime.utils.load_data import from_long_to_nested

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="output data")
parser.add_argument("--output", type=str, help="output data")
parser.add_argument("--threshold", type=float, help="threshold cutoff")
args = parser.parse_args()

# Get input data
pickle_path = os.path.join(args.input, "df_long.pkl")
df_long = pd.read_pickle(pickle_path)

# Convert to Sktime "nested" Format
df_nested = from_long_to_nested(df_long)

# Fake some labels
# We simply explore the data, set an arbitrary threshold and define all series above that threshold as "True".
df_nested["label"] = df_nested["dim_0"].apply(lambda x: x.max()) > args.threshold

# Define output
if not (args.output is None):
    os.makedirs(args.output, exist_ok=True)
    print("%s created" % args.output)
df_nested.to_pickle(os.path.join(args.output, "df_nested.pkl"))
