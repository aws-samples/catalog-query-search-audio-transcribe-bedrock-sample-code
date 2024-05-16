###########################################################################
#
# catalog_query_transcribe_transcripts_sample.py
#
# This Python script:
# -Uses Amazon Bedrock Knowledge Base to send a query to Claude Anthropic v2.1 LLM
# -The Knowledge Base is synced with an S3 bucket containing transcriptions of podcast episodes in JSON format
# -The episodes were transcribed by Amazon Transcribe
# -Upon receiving the query, this script will call Knowledge Base API which will orchestrate RAG workflow
# -The script will then lookup the S3 URI location for the first releveant chunk returned by KB.
# -It will use the S3 URI location to lookup the associated podcast episode title.
# -It will then use the trasncribe JSON object to return the start time of the relevant chunk.
#
################################################################



import boto3
import json
import numpy as np

from urllib.parse import urlparse

from botocore.client import Config

bedrock_config = Config(connect_timeout=120, read_timeout=120, retries={'max_attempts': 0})
bedrock_client = boto3.client('bedrock-runtime')
bedrock_agent_client = boto3.client("bedrock-agent-runtime",
                              config=bedrock_config)

#Submit Query to Amazon Bedrock Knowlege Base
def retrieveAndGenerate(input, kbId, sessionId=None, model_id = "anthropic.claude-instant-v1", region_id = "us-east-1"):

    model_arn = f'arn:aws:bedrock:{region_id}::foundation-model/{model_id}'
    if sessionId:
        return bedrock_agent_client.retrieve_and_generate(
            input={
                'text': input
            },
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': kbId,
                    'modelArn': model_arn
                }
            },
            sessionId=sessionId
        )
    else:
        return bedrock_agent_client.retrieve_and_generate(
            input={
                'text': input
            },
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': kbId,
                    'modelArn': model_arn
                }
            }
        )



#query = "How long has Todd Pond, AWS Director of Strategic Sales, been working in the tech industry?"
#query = "What is the title of the AWS reThink episode where Todd Pond tells us how long he has been working in tech?"
#query = 'What advice does Todd Pond of AWS offer to young people for being successful in tech today?'
#query = "Please recommend an AWS reThink podcast episode that can help me get ready for Gen AI"
    
query = "Why did AWS acquire Annupura Labs?"

model_id = 'anthropic.claude-v2'
region_id = "us-west-2" # replace it with the region you're running sagemaker notebook
kb_id = "XCCECUI94P"   #OpenSearch Serverless

print ("Query: "+ query + "\n")

response = retrieveAndGenerate(query, kb_id,model_id=model_id,region_id=region_id)

generated_text = response['output']['text']
f = open("response.json", "w")
f.write(str(response))
f.close()

print ("Query Response:")
print(generated_text)
print ("\n")


#Use the first retrieved reference chunk to look up episode title
#Extract the text from the first chunk returned
retrievedReferences = response['citations'][0]['retrievedReferences']
chunk_text = retrievedReferences[0]['content']['text']

f = open("chunk_text.txt", "w")
for chunk in retrievedReferences:
    f.write(chunk['content']['text'])
    f.write('\n')
f.close()


#Load contents of Transcribe transcription S3 object into a Python dictionary
s3_uri = response['citations'][0]['retrievedReferences'][0]['location']['s3Location']['uri']
print("s3_uri: "+ s3_uri)
print("\n")


s3 = boto3.client('s3')
o = urlparse(s3_uri)
bucket = o.netloc
obj_name = o.path
key = obj_name[1:] #strip first character '/'
obj = s3.get_object(Bucket=bucket, Key=key)

# Load the JSON data into a Python dictionary
dict = json.loads(obj['Body'].read().decode('utf-8'))

f = open("dict.json", 'w')
json.dump(dict,f)
f.close()


######################
# get episode title
#######################

# Get the object tags
response = s3.get_object_tagging(
    Bucket=bucket,
    Key=key
)

# Print the tags
TagSet = response['TagSet']
title = "Podcast Title"


for Tag in TagSet:
    if (Tag["Key"]=="title"):
        title = Tag["Value"]

#Load the transcription into a string
transcript_text = dict['results']['transcripts'][0]['transcript']
f = open("transcript_text.txt", "w")
f.write(transcript_text)
f.close()

#Initialize Array of Start Time for Each Character Position at "0.00"
transcript_length = len(transcript_text)
start_times = []
start_time = "0.00"
for i in range(transcript_length):
    start_times.append(start_time)

#Create Numpy array initialized with start time "0.00" with size of transpcrition string
start_times_np = np.array(start_times)

#Enter start time of each token based on position of first character in the token
# items[] is an array of dicionaries.  Each dictionary containes start_time for each token
# Loop through each token and get the start time of the token
# Set the start time for the first character of each token
# This will create an array called 'start_times_np[]' which gives the start time of each character position
character_count = 0
items = dict["results"]["items"]
for key in items:

    token = key['alternatives'][0]['content']
    type = key['type']

    token_length = len(token)

    if (type == 'pronunciation'):       
        start_time = key['start_time']
        #print("character_count: " + str(character_count))
        start_times_np[character_count] = start_time
        #start_times = np.assign(start_times, character_count, start_time)
        #print("setting start time! "+ str(character_count)+ " " + start_times[character_count])

    if ((character_count + token_length + 1) <= transcript_length):
        character_count = character_count + token_length + 1

    if (type=='punctuation'):
        character_count = character_count - 1   


#Find the starting character position of the chunk inside the entire transcription
print("find start time for chunk:")
print(chunk_text)
pos = transcript_text.find(chunk_text)

#Return start time for the chunk
#Default to start time of "0.00" if chunk is not found in the transcript
if pos > 0:
    start_time = start_times_np[pos]
    print("\n")
    print("character start position:" + str(pos))
else:    
    start_time = "0.00"


f = open("start_times.json", 'w')
output = items
json.dump(output,f)
f.close()



###########################################
# Print final output of Start Time and Episode Title
###########################################
from datetime import timedelta

sec = float(start_time)
print('Start Time in Seconds:', sec)

td = timedelta(seconds=sec)
print('Start Time in hh:mm:ss:', td)

# Use the below code if you want it in a string
start_time_hh_mm_ss = str(timedelta(seconds=sec))
print ("From episode: "+ title)


