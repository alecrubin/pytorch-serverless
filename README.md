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

## Deployment
Deploy to AWS Lambda
```
sls -v deploy
```
