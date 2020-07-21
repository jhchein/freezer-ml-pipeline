
import os

import argparse

from azureml.core import Run, Workspace, Model
from azureml.core.resource_configuration import ResourceConfiguration

from sktime.highlevel.strategies import TSCStrategy
from sktime.highlevel.tasks import TSCTask
from sktime.classifiers.compose import TimeSeriesForestClassifier

from sklearn.metrics import accuracy_score

from joblib import dump

import pandas as pd

run = Run.get_context()

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default=None, help='input dataset')
parser.add_argument('--n_estimators', type=int, default=10, help='Number of tree estimators used in the model')
parser.add_argument('--train_data_split', type=float, default=0.8, help='Fraction of samples for training')
args = parser.parse_args()

# Load data    
pickle_path= os.path.join(args.input, "df_nested.pkl")
processed_data_df = pd.read_pickle(pickle_path)

# Split data
train = processed_data_df.sample(frac=args.train_data_split, random_state=42)
test = processed_data_df.drop(train.index)

# Example logging
run.log("data_split_fraction", args.train_data_split, "Fraction of samples used for training")
run.log("train_samples", train.shape[0], "Number of samples used for training")
run.log("test_samples", test.shape[0], "Number of samples used for testing")

# Train
task = TSCTask(target="label", metadata=train)
clf = TimeSeriesForestClassifier(n_estimators=args.n_estimators)
strategy = TSCStrategy(clf)
strategy.fit(task, train)
run.log("n_estimators", args.n_estimators, "Number of tree estimators used in the model")

# Metrics
y_pred = strategy.predict(test)
y_test = test[task.target]
accuracy = accuracy_score(y_test, y_pred)
run.log("Accuracy", f"{accuracy:1.3f}", "Accuracy of model")

# Add to outputs
os.makedirs("outputs", exist_ok=True)
local_model_path = os.path.join("outputs", "model.pkl")
dump(strategy, local_model_path)
run.upload_file("pickled_model", local_model_path)

model = Model.register(
    workspace=run.experiment.workspace,
    model_name="sktime_freezer_classifier",
    model_path=local_model_path,  # Local file to upload and register as a model.
    tags={
        "area": "freezerchain",
        "type": "classification",
        "purpose": "demonstration",
        "source": "pipeline"
    },
    description="Sktime classifier to predict if freezer chain was interrupted.",
    resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=0.5))
