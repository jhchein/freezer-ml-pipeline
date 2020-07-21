from azureml.core import Environment, Workspace
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DEFAULT_CPU_IMAGE

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
