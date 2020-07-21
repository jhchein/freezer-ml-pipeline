
from azureml.core import Model, Environment, Webservice, Keyvault, Run
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.exceptions import WebserviceException

import argparse
import time

run = Run.get_context()
ws = run.experiment.workspace

parser = argparse.ArgumentParser()
parser.add_argument('--webservicename', type=str, default="freezerchain-prediction-v0-2", help='Name of the deployed Webservice')
args = parser.parse_args()

service = Webservice(ws, args.webservicename)

service.update_deployment_state()

retries = 0
max_wait_time_minutes=5
polling_frequency_seconds=15
max_retries = int(max_wait_time_minutes*60/polling_frequency_seconds)

assert max_retries > 0

while service.state == "Transitioning" and retries <= max_retries:
    retries += 1
    print("Service still transitioning")
    
    time.sleep(polling_frequency_seconds)
    service.update_deployment_state()

service.update_deployment_state()

assert service.state=="Healthy", f"Service state not healthy after {retries} retries."

print(f"Scoring Uri: '{service.scoring_uri}'")
run.log("Scoring Uri", service.scoring_uri)

# Test WEBSERVICE
with open("sample_data.json", "r") as fh:
    sample_data = json.loads(fh.read())
input_payload = json.dumps(sample_data)
output = service.run(input_payload)
print(f"Test output : '{output}'")

# Store Endpoint in Keyvault
keyvault = ws.get_default_keyvault()
keyvault.set_secret(name="webservice-name", value = args.webservicename)
keyvault.set_secret(name="MLENDPOINT", value = service.scoring_uri)
print("Keyvault secrets updated.")
