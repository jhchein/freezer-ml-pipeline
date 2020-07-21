import json

import pandas as pd

from sktime.utils.load_data import from_long_to_nested

def prepare_dataframe(
    processed_json_df: pd.DataFrame, time_series_length: int, threshold: float
):
    # Convert to JSON
    processed_json_df["allevents"] = processed_json_df["allevents"].apply(lambda x: json.loads(x))

    # Reset PartitionDate Index to a simple range. We'll use it to index our "cases" ("samples")
    processed_json_df.reset_index(drop=True, inplace=True)

    # sktime expects a specific format. For now the easiest way is to convert our DataFrame to a long format
    # and then use the sktime parser.
    def dataframe_to_long(df, size=time_series_length):
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
        dataframe_to_long(processed_json_df, size=time_series_length),
        columns=["case_id", "dim_id", "reading_id", "value"],
    )

    # Convert to Sktime "nested" Format
    df_nested = from_long_to_nested(df_long)

    # Fake some labels
    # We simply explore the data, set an arbitrary threshold and define all series above that threshold as "True".
    df_nested["label"] = df_nested["dim_0"].apply(lambda x: x.max()) > threshold

    return df_nested
