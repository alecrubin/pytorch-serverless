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

## Invoke Local
Run function locally with params defined in `tests/predict_event.json`
```
AWS_PROFILE=YOUR_PROFILE sls invoke local -f predict -p tests/predict_event.json
```

## Deployment
Deploy to AWS Lambda
```
sls -v deploy
```
