from azureml.core import Dataset, Run, Workspace
from azureml.data import TabularDataset
from utils import prepare_dataframe

from sktime.highlevel.strategies import TSCStrategy
from sktime.highlevel.tasks import TSCTask
from sktime.classifiers.compose import TimeSeriesForestClassifier

from sklearn.metrics import accuracy_score

from joblib import dump

run = Run.get_context()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeserieslength', type=int, default=10, help='Timesteps per timeseries')
    parser.add_argument('--threshold', type=float, default=180.0, help='Threshold temperature')
    parser.add_argument('--n_estimators', type=int, default=10, help='Number of tree estimators used in the model')
    parser.add_argument('--train_data_split', type=float, default=0.8, help='Fraction of samples for training')
    parser.add_argument('--model_folder', type=str, default="outputs", help='Folder where model will be stored')
    parser.add_argument('--model_filename', type=str, default="model.pkl", help='Model filename')
    # parser.add_argument('--dataset', type=TabularDataset, help='Model filename')
    args = parser.parse_args()
    return args

def main(args):
    # Load and wrangle data
    dataset = run.input_datasets['rawdata']
    raw_data_df = dataset.to_pandas_dataframe()

    processed_data_df = prepare_dataframe(
        raw_data_df, time_series_length=args.timeserieslength, threshold=args.threshold
    )

    # Split data
    train = processed_data_df.sample(frac=args.train_data_split, random_state=42)
    test = processed_data_df.drop(train.index)

    # Example for logging
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

    # Persist model
    os.makedirs(args.model_folder, exist_ok=True)
    model_path = os.path.join(args.model_folder, args.model_filename)
    dump(strategy, model_path)

if __name__ == '__main__':
    args = parse_args()
    main(args=args)