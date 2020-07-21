
from azureml.core import Run, Datastore, Dataset
from azureml.data import DataType
from azureml.data.datapath import DataPath

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, help="name of the input dataset")
args = parser.parse_args()

run = Run.get_context()
ws = run.experiment.workspace

datastore = Datastore.get(ws, "sensordata")
datapath = DataPath(datastore=datastore, path_on_datastore="/processed/json/**")
dataset = Dataset.Tabular.from_json_lines_files(path=datapath, validate=True, include_path=False, set_column_types={"allevents": DataType.to_string(), "ConnectionDeviceID": DataType.to_string()}, partition_format='/{PartitionDate:yyyy/MM/dd}/')
dataset.register(workspace=ws, name="processed_json", description="Output from Stream Analytics", create_new_version=True)
print("dataset registered")
