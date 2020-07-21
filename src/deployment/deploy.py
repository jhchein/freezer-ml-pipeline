
from azureml.core import Model, Environment, Webservice, Keyvault, Run
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.exceptions import WebserviceException

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--redeploy', type=bool, default=True, help='do you want to re-deploy an existing service?')
parser.add_argument('--webservicename', type=str, default="freezerchain-prediction-v0-2", help='Name of the deployed Webservice')
args = parser.parse_args()

run = Run.get_context()
ws = run.experiment.workspace

freezer_environment = ws.environments["sktime_freezer_environment"]

try:
    service = Webservice(ws, args.webservicename)
except WebserviceException:
    service = None

if args.redeploy:
    if service is not None:
        service.delete()
        print("deleted existing Webservice.")
    
    model = Model(ws, "sktime_freezer_classifier")

    inference_config = InferenceConfig(
        entry_script="score.py", 
        source_directory="./", 
        environment=freezer_environment)

    aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1,)

    service = Model.deploy(
        workspace=ws,
        name=args.webservicename,
        models=[model],
        inference_config=inference_config,
        deployment_config=aci_config,
    )

    # service.wait_for_deployment(show_output=False)
