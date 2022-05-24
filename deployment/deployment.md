### Start the docker container
1. minkube start
### Enable the ingress addon 
2. minikube addons enable ingress
### Make ingress available at localhost
3. minikube tunnel
### Monitor deployment in the browser
3. minikube dashboard 
### Create/Remove deployment
4. kubectl apply -f .\deployment\deployment.yml
5. kubectl delete -f .\deployment\deployment.yml
