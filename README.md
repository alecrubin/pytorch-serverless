# PyTorch Serverless
PyTorch Serverless API (w/ AWS Lambda)

## Setup
 - Install [Serverless Framework](https://serverless.com/) via npm
    ```
    npm i -g serverless@v1.27.3
    ```
    
 - Install python requirements plugin
    ```
    sls plugin install -n serverless-python-requirements
    ```
    
    
## Configuration
 - Setup your model in `lib/models.py` so that it can be imported by your handler in `api/predict.py` as a method.
 
 - Define your class labels in `lib/labels.json`, for example:
    ```
    {
        "0": "cat",
        "1": "dog"
    } 
    ```
    
 - Setup an [AWS CLI profile](https://docs.aws.amazon.com/cli/latest/userguide/cli-multiple-profiles.html) if you 
 don't have one already
 
 - Create an [S3 Bucket](https://docs.aws.amazon.com/AmazonS3/latest/dev/UsingBucket.html#create-bucket-intro) that your 
 profile can access and upload your state dictionary
 
 - Configure the `serverless.yml`
    ```
    ## Change service name to whatever you please
    service: pytorch-serverless
    
    provider:
        ...
        # set this to your deployment stage
        stage: dev
        
        # set this to your aws region
        region: us-west-2
        
        # set this to your aws profile
        profile: slsadmin
        ...
        
        environment:
            # set this to your S3 bucket name
            BUCKET_NAME: pytorch-serverless
            
            # set this to your state dict filename
            STATE_DICT_NAME: dogscats-resnext50.h5
            
            # set path to labels file
            LABELS_PATH: lib/labels.json
            
            # set this to your input image size
            IMAGE_SIZE: 224
         
        variables:
            # set this to your api version
            api_version: v0.0.1
    ```
    

## Invoke Local
Run function locally with params defined in `tests/predict_event.json`
```
AWS_PROFILE=yourProfile sls invoke local -f predict -p tests/predict_event.json
```

## Deployment
Deploy to AWS Lambda
```
sls -v deploy
```

