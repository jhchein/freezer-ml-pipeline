import json
import os

from azureml.core import Dataset, Datastore, Experiment, Model, Webservice, Workspace
from azureml.core.compute import ComputeTarget
from azureml.core.model import InferenceConfig
from azureml.core.resource_configuration import ResourceConfiguration
from azureml.core.webservice import AciWebservice
from azureml.data import DataType
from azureml.data.datapath import DataPath
from azureml.exceptions import WebserviceException
from azureml.train.estimator import Estimator

# PARAMS

TIMESERIESLENGTH = 10
THRESHOLD = 180.0
TRAIN_DATA_SPLIT = 0.8
NUMBER_ESTIMATORS = 10
TRAIN_FOLDER_NAME = "train"
TRAIN_FILE_NAME = "train.py"
MODELNAME = "script-classifier"
SERVICENAME = "script-deployment"
MODELFILENAME = "model.pkl"

ws = Workspace.from_config()
exp = Experiment(ws, "MaxFreezerTemperatureExceeded", _create_in_cloud=True)

# ACCESS DATA

datastore = Datastore.get(ws, "sensordata")
datapath = DataPath(datastore=datastore, path_on_datastore="/processed/json/**")
dataset = Dataset.Tabular.from_json_lines_files(
    path=datapath,
    validate=True,
    include_path=False,
    set_column_types={
        "allevents": DataType.to_string(),
        "ConnectionDeviceID": DataType.to_string(),
    },
    partition_format="/{PartitionDate:yyyy/MM/dd}/",
)
dataset.register(
    workspace=ws,
    name="processed_json",
    description="Output from Stream Analytics",
    create_new_version=True,
)
print("dataset registered")

# PREPARE TRAINING

compute_target = ComputeTarget(ws, "freezertrain")

script_params = {
    "--timeserieslength": TIMESERIESLENGTH,
    "--n_estimators": NUMBER_ESTIMATORS,
    "--threshold": THRESHOLD,
    "--model_filename": MODELFILENAME,
    "--train_data_split": TRAIN_DATA_SPLIT,
}

est = Estimator(
    source_directory=TRAIN_FOLDER_NAME,
    entry_script=TRAIN_FILE_NAME,
    script_params=script_params,
    inputs=[Dataset.get_by_name(ws, name="processed_json").as_named_input("rawdata")],
    compute_target=compute_target,
    conda_packages=["scikit-learn", "pandas", "cython"],
    pip_packages=[
        "azureml-dataprep[pandas,fuse]==1.4.0",
        "azureml-defaults==1.1.5",
        "sktime",
    ],
)

# TRAIN

run = exp.submit(est)
run.wait_for_completion(show_output=False, wait_post_processing=True)

model = run.register_model(
    model_name=MODELNAME,
    model_path=os.path.join("outputs", MODELFILENAME),
    tags={
        "area": "freezerchain",
        "type": "classification",
        "purpose": "demonstration",
    },
    properties=None,
    description="Sktime classifier to predict if freezer chain was interrupted.",
    datasets=[("training_data", dataset)],
    resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=0.5),
)

# DEPLOY

freezer_environment = ws.environments["sktime_freezer_environment"]

try:
    Webservice(ws, SERVICENAME).delete()
    print("deleted existing Webservice.")
except WebserviceException:
    pass

service = Model.deploy(
    workspace=ws,
    name=SERVICENAME,
    models=[model],
    inference_config=InferenceConfig(
        entry_script="score.py",
        source_directory="deployment",
        environment=freezer_environment,
    ),
    deployment_config=AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1,),
)

service.wait_for_deployment(show_output=False)

print(f"Scoring Uri: '{service.scoring_uri}'")

keyvault = ws.get_default_keyvault()
keyvault.set_secret(name="webservice-name", value=SERVICENAME)
keyvault.set_secret(name="MLENDPOINT", value=service.scoring_uri)

# TEST DEPLOYMENT

keyvault = ws.get_default_keyvault()
service = Webservice(ws, keyvault.get_secret("webservice-name"))

with open("deployment/sample_data.json", "r") as fh:
    sample_data = json.loads(fh.read())

input_payload = json.dumps(sample_data)
output = service.run(input_payload)

print(output)
print(service.get_logs())
