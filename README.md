# REMLA 2022 Group 8

This project was conducted as part of the course [*Release Engineering for Machine Learning Applications* (REMLA)] taught at the Delft University of Technology by [Prof. Luís Cruz] and [Prof. Sebastian Proksch].

## Installation

Run: `pip install -e .[extra]`

## Linting

Find the commands per tool below:

- `mllint`: `mllint .`
- `flake8`: `flake8 .`
- `pylint`: `pylint src && pylint tests`
- `mypy`: `mypy .`

## Experiment tracking with Weights and Biases

Loging to Weights and Biases using the CLI tool:

`wandb login`

Run the code and the results will be logged to wandb. Note that you need to be part of the `remla-2022-group-8` entity to have access to already pushed artifacts. Contact one of the authors to be added to the organization. 

## Reproducibility pipeline

To add a new model to the DVC reproducibility pipeline, add a class that extends `BaseModel` in the `models` package.

Add the name of the module, the name of the class and a config name in the `params.yaml` file. Note: the `config` field is not used right now.

## Grafana Dashboard

Build the service located in the root of the repository:
`docker-compose build `

Run the service:
`docker-compose up`

## Kubernetes
### Setup

Create the namespace where we will run the deployment 
```
kubectl create ns remla
```

Install the nginx ingress constroller and set prometheus config
```
helm upgrade -i ingress-nginx ingress-nginx/ingress-nginx --namespace ingress-nginx --set controller.metrics.enabled=true --set controller.podAnnotations."prometheus\.io/scrape"=true --set controller.podAnnotations."prometheus\.io/port"=5000
```

Install flagger and prometheus
```
helm upgrade -i flagger flagger/flagger --namespace ingress-nginx --set prometheus.install=true --set meshProvider=nginx
```

Install the flagger load tester
```
helm upgrade -i flagger-loadtester flagger/loadtester --namespace=remla
```

### Deploy

Create deployment out of the resources in k8s folder
```
kubectl apply -f .\k8s\  
```

Change the image of the remla-deployment, which trigger the canary update. Currently the deployment runs the farsene/remla-v1 image. The following command updates to a new version.
```
kubectl set image deployment/remla-deployment remla=farsene/remla-v2 -n remla
```

###  Command to open the canary gate to let them fly
Enter the load tester and approve the canary rollout by opening the gate
```
kubectl -n remla exec -it flagger-loadtester-7c47f949d-j8wt7 sh
curl -d '{"name": "remla","namespace":"remla"}' http://localhost:8080/gate/open
```

[*Release Engineering for Machine Learning Applications* (REMLA)]: https://se.ewi.tudelft.nl/remla/ 
[Prof. Luís Cruz]: https://luiscruz.github.io/
[Prof. Sebastian Proksch]: https://proks.ch/
