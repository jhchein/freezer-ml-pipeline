import logging

from azureml.core import Environment, Workspace
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DEFAULT_CPU_IMAGE

logger = logging.getLogger()
logger.setLevel("INFO")
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

ws = Workspace.from_config()

freezer_environment = Environment("sktime_freezer_environment")
cd = CondaDependencies.create(
    conda_packages=["numpy", "cython", "pandas", "scikit-learn"],
    pip_packages=[
        "azureml-defaults",
        "inference-schema[numpy-support]",
        "joblib==0.13.*",
        "azureml-dataprep[pandas, fuse]",
        "sktime",
    ],
)
freezer_environment.docker.enabled = True
freezer_environment.docker.base_image = DEFAULT_CPU_IMAGE
freezer_environment.python.conda_dependencies = cd
freezer_environment.register(workspace=ws)
logger.info("Environment registered")

try:
    cpu_cluster = ComputeTarget(workspace=ws, name="freezertrain")
    logger.info("Found existing compute target")
except ComputeTargetException:
    logger.info("Creating a new compute target...")
    cpu_cluster = ComputeTarget.create(
        ws,
        "freezertrain",
        AmlCompute.provisioning_configuration(vm_size="STANDARD_DS3_V2", max_nodes=8),
    )

    cpu_cluster.wait_for_completion(
        show_output=True, min_node_count=None, timeout_in_minutes=20
    )

logger.info(cpu_cluster.get_status().serialize())
