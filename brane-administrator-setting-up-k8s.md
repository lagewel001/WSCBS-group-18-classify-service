# Brane administrator K8S Setup guide
This guide will help set up the pipeline on the Brane framework deployed on Kubernetes.
This guide will assume you have at least three VM's: one for the Brane control plane and two
for the Kubernetes cluster itself (one control and one worker node). The Brane control plane
VM must have at least 8GB of memory.

## K8S control node VM:
1. Install Docker (https://docs.docker.com/engine/install/ubuntu/)
2. Run `sudo groupadd docker`
3. Run `sudo usermod -aG docker $USER`
4. logout of the VM and log back in
5. Run `curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add`
6. Run `sudo apt-add-repository "deb http://apt.kubernetes.io/ kubernetes-xenial main"`
7. Install kubeadm `sudo apt install kubeadm -y`
8. Init k8s `sudo kubeadm init`
9. Configure k8s config
    ```
    mkdir $HOME/.kube
    sudo cp /etc/kubernetes/admin.conf $HOME/.kube/config
    sudo chown $(id -u):$(id -g) $HOME/.kube/config
    ```
10. Join worker nodes with the `kubeadm join` command gained from step 8
11. Label the worker nodes `kubectl label node <node> node-role.kubernetes.io/worker=worker`
12. Install Calico
    ```
    kubectl create -f https://docs.projectcalico.org/manifests/tigera-operator.yaml
    kubectl create -f https://docs.projectcalico.org/manifests/custom-resources.yaml
    ```
13. Follow the cluster setup guide steps from Brane guide (https://wiki.enablingpersonalizedinterventions.nl/admins/installation/deployment.html#distributed-deployment)
    <br /><br />
    *note: to restart daemon when opening the registry connection run
    ```
    sudo systemctl daemon-reload
    sudo systemctl restart docker
    ```
15. Create and copy a token for the default service account (note: this will not be secure)
    ```
    kubectl apply -f - <<EOF
    apiVersion: v1
    kind: Secret
    metadata:
      name: default-token
      annotations:
        kubernetes.io/service-account.name: default
    type: kubernetes.io/service-account-token
    EOF
    
    kubectl describe secret default-token
    ```
16. Configure access rights for the service account (note: the example below is most definitely not secure)
    ```
    cat <<EOF | kubectl apply -f -
    apiVersion: rbac.authorization.k8s.io/v1
    kind: ClusterRole
    
    metadata:
      name: default-cluster-role
    
    rules:
      - apiGroups: ["", "apps", "batch"]
        resources: ["nodes", "services", "pods", "pods/log", "events", endpoints", "configmaps", "deployments", "replicasets", "statefulsets", "jobs", "cronjobs", "persistentvolumeclaims"]
        verbs: ["get", "list", "watch", "create", "delete", "deletecollection", "patch", "update"]
    EOF
    ```
    ```
    cat <<EOF | kubectl apply -f -
    apiVersion: rbac.authorization.k8s.io/v1
    kind: ClusterRoleBinding
    metadata:
      name: default-role-binding
      namespace: default
    roleRef:
      apiGroup: rbac.authorization.k8s.io
      kind: ClusterRole
      name: default-cluster-role
    subjects:
    - kind: ServiceAccount
      name: default
      namespace: default
    
    ---
    apiVersion: rbac.authorization.k8s.io/v1
    kind: ClusterRoleBinding
    metadata:
      name: default-role-binding-brane-control
      namespace: brane-control
    roleRef:
      apiGroup: rbac.authorization.k8s.io
      kind: ClusterRole
      name: default-cluster-role
    subjects:
    - kind: ServiceAccount
      name: default
      namespace: brane-control
    
    ---
    apiVersion: rbac.authorization.k8s.io/v1
    kind: ClusterRoleBinding
    metadata:
      name: default-role-binding-brane
      namespace: brane
    roleRef:
      apiGroup: rbac.authorization.k8s.io
      kind: ClusterRole
      name: default-cluster-role
    subjects:
    - kind: ServiceAccount
      name: default
      namespace: brane
    EOF
    ```


## Brane control plane VM:
1. Install and compile brane from source following the guide (https://wiki.enablingpersonalizedinterventions.nl/admins/installation/get-binaries.html)
2. Open registry connection by following the brane guide
   * `<cluster control node IP>`: the IP of the K8S cluster control node
   * `<cluster name>`: come up with a fancy name
   * `<service account access token>` token gained from step 15 in the cluster steps above
    ```
    mkdir -p ~/.kube
   
    cat >> ~/.kube/config <<EOF
    apiVersion: v1
    clusters:
    - cluster:
        insecure-skip-tls-verify: true
        server: https://<cluster control node IP>:6443
      name: <cluster name>
    contexts:
    - context:
        cluster: lucas-wscbs-vm-cluster
        user: <cluster name>-admin
      name: <cluster name>
    current-context: <cluster name>
    kind: Config
    preferences: {}
    users:
    - name: <cluster name>-admin
      user:
        token: <service account access token>
    EOF
   
    chown $(id -u):$(id -g) ~/.kube/config
    chmod 644 ~/.kube/config
    ```
   
Congratulations! After having run these steps your cluster should be up and running for using Brane.
