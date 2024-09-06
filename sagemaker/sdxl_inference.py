import boto3
import io
import json
from PIL import Image

region='us-east-1'

# Initialize the SageMaker runtime client
sagemaker_runtime = boto3.client('sagemaker-runtime',
                                 region_name='',
                                 aws_access_key_id='',
                                 aws_secret_access_key='')

# Define the endpoint name
endpoint_name = 'huggingface-pytorch-inference-2024-02-21-05-58-10-645'  # Replace with your endpoint name

# Define the payload for inference
payload = {"inputs": "A cinematic shot of a tiger in space"}

# Make a prediction request to the SageMaker endpoint
response = sagemaker_runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    Body=json.dumps(payload),
    ContentType='application/json'
)

# Parse the JSON response
result = json.loads(response['Body'].read().decode())

# Get the image bytes from the result
image_bytes = result['image']

# Open the image using PIL
image = Image.open(io.BytesIO(image_bytes))

# Display the image
image.show()



####################################################################################################################

# from sagemaker.predictor import Predictor

# endpoint = 'huggingface-pytorch-inference-2024-02-21-05-58-10-645'

# # Read image into memory
# payload = {"inputs": "A cinematic shot of a tiger in space."}

# predictor = Predictor(endpoint)
# inference_response = predictor.predict(data=payload)
# print (inference_response)