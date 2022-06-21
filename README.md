# StackOverflow post tag prediction using ML

This project was conducted as part of the course [*Release Engineering for Machine Learning Applications* (REMLA)] taught at the Delft University of Technology by [Prof. Luís Cruz] and [Prof. Sebastian Proksch].

## Installation

Run: `pip install -e .[extra]`

## Implementation

### Multilabel classification on Stack Overflow tags
Predict tags for posts from StackOverflow with multilabel classification approach.

#### Dataset
- Dataset of post titles from StackOverflow

#### Transforming text to a vector
- Transformed text data to numeric vectors using bag-of-words and TF-IDF.

#### MultiLabel classifier
[MultiLabelBinarizer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html) to transform labels in a binary form and the prediction will be a mask of 0s and 1s.

[Logistic Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) for Multilabel classification
- Coefficient = 10
- L2-regularization technique

#### Evaluation
Results evaluated using several classification metrics:
- [Accuracy](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
- [F1-score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

Note: this sample project was originally created by @partoftheorigin

## Linting

Find the commands per tool below:

- `mllint`: `mllint .`
- `flake8`: `flake8 .`
- `pylint`: `pylint src && pylint tests`
- `mypy`: `mypy .`

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
