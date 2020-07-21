
import argparse
import os
from azureml.core import Run, Dataset

import json
import pandas as pd

time_series_length = 10

# Load arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, help="name of the input dataset")
parser.add_argument("--output", type=str, help="output data")
parser.add_argument("--time_series_length", type=int, help="number of samles per time series")
args = parser.parse_args()

ws = Run.get_context().experiment.workspace

rawdata = Dataset.get_by_name(ws, name=args.dataset_name)
# input_named = input_data.as_named_input('rawdata')

# Get dataframe
# rawdata = Run.get_context().input_datasets["rawdata"]
rawdata_df = rawdata.to_pandas_dataframe()

# Convert to JSON
rawdata_df["allevents"] = rawdata_df["allevents"].apply(lambda x: json.loads(x))

# Reset PartitionDate Index to a simple range. We'll use it to index our "cases" ("samples")
rawdata_df.reset_index(drop=True, inplace=True)

# sktime expects a specific format. For now the easiest way is to convert our DataFrame to a long format
# and then use the sktime parser. 
def dataframe_to_long(df, size):
    case_id = 0
    for _, case in df.iterrows():
        events = case["allevents"]

        # We ignore cases with insufficient readings
        if len(events) < size:
            continue

        # We also slice samples with too many readings ([-size:])
        for reading_id, values in enumerate(events[-size:]):
            yield case_id, 0, reading_id, values["temperature"]
            # We can add more dimensions later on.
            # yield case_id, 1, reading_id, values["ambienttemperature"]

        case_id += 1  # can't use the row index because we skip rows.

df_long = pd.DataFrame(
    dataframe_to_long(rawdata_df, size=args.time_series_length),
    columns=["case_id", "dim_id", "reading_id", "value"],
)

# Store output
if not (args.output is None):
    os.makedirs(args.output, exist_ok=True)
    print("%s created" % args.output)
df_long.to_pickle(os.path.join(args.output, "df_long.pkl"))
