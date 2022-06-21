# Setup

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

# Deploy

Create deployment out of the resources in k8s folder
```
kubectl apply -f .\k8s\  
```

Change the image of the remla-deployment, which trigger the canary update. Currently the deployment runs the farsene/remla-v1 image. The following command updates to a new version.
```
kubectl set image deployment/remla-deployment remla=farsene/remla-v2 -n remla
```

#  Command to open the canary gate to let them fly
Enter the load tester and approve the canary rollout by opening the gate
```
kubectl -n remla exec -it flagger-loadtester-7c47f949d-j8wt7 sh
curl -d '{"name": "remla","namespace":"remla"}' http://localhost:8080/gate/open
```