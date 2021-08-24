import logging

from azureml.core import Experiment, RunConfiguration, Workspace
from azureml.core.compute import ComputeTarget
from azureml.pipeline.core import Pipeline, PipelineData, PipelineParameter
from azureml.pipeline.steps import EstimatorStep, PythonScriptStep
from azureml.train.estimator import Estimator

# PREPARE LOGGING

logger = logging.getLogger()
logger.setLevel("INFO")
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# GET WS, EXP, ENV and COMPUTE TARGET

ws = Workspace.from_config()
experiment = Experiment(ws, "FreezerTemperatureExceededPipeline", _create_in_cloud=True)
compute_target = ComputeTarget(ws, "freezertrain")
run_config = RunConfiguration()
freezer_environment = ws.environments["sktime_freezer_environment"]
run_config.environment = freezer_environment
logger.info("Environment complete")

# PIPELINE PARAMS

output_df_long = PipelineData("output_df_long", datastore=ws.get_default_datastore())
output_df_nested = PipelineData(
    "output_df_nested", datastore=ws.get_default_datastore()
)
time_series_length_param = PipelineParameter(
    name="time_series_length", default_value=10
)
threshold_param = PipelineParameter(name="threshold", default_value=180.0)
dataset_name_param = PipelineParameter(
    name="dataset_name", default_value="processed_json"
)
n_estimators_param = PipelineParameter(name="n_estimators", default_value=10)
train_data_split_param = PipelineParameter(name="train_data_split", default_value=0.8)
redeploy_webservice_param = PipelineParameter(name="redeploy", default_value=True)
webservicename_param = PipelineParameter(
    name="webservice", default_value="freezerchain-prediction-v0-2"
)
logger.info("Prepared Pipeline paramaters")

# DEFINE PIPELINE STEPS

update_dataset_step = PythonScriptStep(
    name="Update Dataset",
    script_name="00_update_dataset.py",
    arguments=["--dataset_name", dataset_name_param],
    compute_target=compute_target,
    source_directory="src/preprocess",
    runconfig=run_config,
    allow_reuse=False,
)

first_prepro_step = PythonScriptStep(
    name="Parse dataset",
    script_name="01_raw_to_long.py",
    arguments=[
        "--dataset_name",
        dataset_name_param,
        "--output",
        output_df_long,
        "--time_series_length",
        time_series_length_param,
    ],
    outputs=[output_df_long],
    compute_target=compute_target,
    source_directory="src/preprocess",
    runconfig=run_config,
    allow_reuse=True,
)

second_prepro_step = PythonScriptStep(
    name="Convert to SKTime format",
    script_name="02_long_to_nested.py",
    arguments=[
        "--input",
        output_df_long,
        "--output",
        output_df_nested,
        "--threshold",
        threshold_param,
    ],
    inputs=[output_df_long],
    outputs=[output_df_nested],
    compute_target=compute_target,
    source_directory="src/preprocess",
    runconfig=run_config,
    allow_reuse=True,
)

estimator_step = EstimatorStep(
    name="Train Model",
    estimator=Estimator(
        source_directory="src/train",
        entry_script="train_pipeline.py",
        compute_target=compute_target,
        environment_definition=freezer_environment,
    ),
    estimator_entry_script_arguments=[
        "--input",
        output_df_nested,
        "--n_estimators",
        n_estimators_param,
        "--train_data_split",
        train_data_split_param,
    ],
    runconfig_pipeline_params=None,
    inputs=[output_df_nested],
    compute_target=compute_target,
    allow_reuse=True,
)

deploy_step = PythonScriptStep(
    name="Deploy Model",
    script_name="deploy.py",
    arguments=[
        "--redeploy",
        redeploy_webservice_param,
        "--webservicename",
        webservicename_param,
    ],
    compute_target=compute_target,
    source_directory="src/deployment",
    runconfig=run_config,
    allow_reuse=True,
)

validate_deployment_step = PythonScriptStep(
    name="Validate Deployment",
    script_name="validate.py",
    arguments=["--webservicename", webservicename_param],
    compute_target=compute_target,
    source_directory="src/deployment",
    runconfig=run_config,
    allow_reuse=False,
)

logger.info("Pipeline steps prepared")

# DEFINE STEP DEPENDENCIES

first_prepro_step.run_after(update_dataset_step)
deploy_step.run_after(estimator_step)
validate_deployment_step.run_after(deploy_step)

# DEFINE PIPELINE

steps = [
    update_dataset_step,
    first_prepro_step,
    second_prepro_step,
    estimator_step,
    deploy_step,
    validate_deployment_step,
]

pipeline = Pipeline(workspace=ws, steps=[steps])
logger.info("Pipeline preparation done.")

# RUN PIPELINE
logger.info("Starting Pipeline.")
pipeline_run = experiment.submit(pipeline, pipeline_parameters={"threshold": 180.0})
pipeline_run.wait_for_completion()
logger.info("Pipeline complete")

# PUBLISH PIPELINE

published_pipeline = pipeline_run.publish_pipeline(
    name="Freezer_SKTIME_pipeline",
    description="Freezer SKTIME train and deploy pipeline",
)
logger.info("Pipeline published")
