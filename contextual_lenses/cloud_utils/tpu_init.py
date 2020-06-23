"""Function to connect VM instance to Cloud TPU."""


import requests
import yaml
import subprocess as sp
from jax.config import config


def connect_tpu(tpu_name=None):
    """Runs necessary commands to connect VM to Cloud TPU."""
    
    if tpu_name is not None:
        command = 'gcloud compute tpus describe ' + tpu_name

        output = sp.getoutput(command)
        output = yaml.load(stream=output, Loader=yaml.SafeLoader)

        ip_address = output['ipAddress']
        port = output['port']

        url = 'http://' + ip_address + ':8475/requestversion/tpu_driver_nightly'
        requests.post(url)

        config.FLAGS.jax_xla_backend = 'tpu_driver'
        config.FLAGS.jax_backend_target = "grpc://" + ip_address + ':' + port
        
        print('Successfully connected to TPU named \"' + tpu_name + '\"!')
        print()