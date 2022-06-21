# Setup

```
kubectl create ns remla
```

```
helm upgrade -i ingress-nginx ingress-nginx/ingress-nginx --namespace ingress-nginx --set controller.metrics.enabled=true --set controller.podAnnotations."prometheus\.io/scrape"=true --set controller.podAnnotations."prometheus\.io/port"=5000
```

```
helm upgrade -i flagger flagger/flagger --namespace ingress-nginx --set prometheus.install=true --set meshProvider=nginx
```

```
helm upgrade -i flagger-loadtester flagger/loadtester --namespace=remla
```

# Deploy

```
kubectl apply -f .\k8s\  
```

```
kubectl set image deployment/remla-deployment remla=farsene/remla-v2 -n remla
```

#  Command to open the canary gate to let them fly
```
kubectl -n remla exec -it flagger-loadtester-7c47f949d-j8wt7 sh
curl -d '{"name": "remla","namespace":"remla"}' http://localhost:8080/gate/open
```