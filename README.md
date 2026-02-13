# GPU-Powered LLM Chat Interface on Akamai Cloud (Linode) LKE

Deploy a GPU-accelerated chat interface using vLLM and Open WebUI on Akamai Cloud's Linode Kubernetes Engine (LKE) with RTX 4000 Ada GPUs.

## Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    Akamai Cloud LKE                     │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │            Namespace: ai-chat                    │  │
│  │                                                  │  │
│  │  ┌────────────────┐      ┌──────────────────┐  │  │
│  │  │  Open WebUI    │      │   vLLM Server    │  │  │
│  │  │                │─────▶│                  │  │  │
│  │  │  Port: 8080    │ HTTP │ Port: 8000       │  │  │
│  │  │                │      │                  │  │  │
│  │  └────────┬───────┘      │ RTX 4000 Ada GPU │  │  │
│  │           │              │                  │  │  │
│  │           │              │ Llama 3.2 3B     │  │  │
│  │  ┌────────▼───────┐      │ Instruct Model   │  │  │
│  │  │      PVC       │      └──────────────────┘  │  │
│  │  │   (10Gi)       │                            │  │
│  │  └────────────────┘                            │  │
│  │                                                  │  │
│  └──────────────────────────────────────────────────┘  │
│                          │                              │
│                ┌─────────▼─────────┐                    │
│                │   LoadBalancer    │                    │
│                │   External IP     │                    │
│                └─────────┬─────────┘                    │
└──────────────────────────┼──────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │   Browser   │
                    │  (Port 80)  │
                    └─────────────┘
```

## Features

- 🚀 High-performance LLM inference with vLLM
- 💬 Modern chat interface with Open WebUI
- 🎮 GPU acceleration (RTX 4000 Ada)
- 🔓 Open-source model (Llama 3.2 3B Instruct)
- ☁️ Cloud-native deployment on Kubernetes

## Prerequisites

- Akamai Cloud (Linode) account with LKE cluster
- NVIDIA GPU Operator installed via Helm
- `kubectl` configured to access your cluster
- `helm` CLI installed
- Hugging Face account and token ([Get one here](https://huggingface.co/settings/tokens))

## Quick Start

### 1. Create namespace
```bash
kubectl create namespace ai-chat
```

### 2. Configure Hugging Face token

Edit `vllm-deployment.yaml` and replace `<YOUR_HUGGINGFACE_TOKEN>` with your actual token:
```yaml
env:
- name: HF_TOKEN
  value: "hf_xxxxxxxxxxxxxxxxxxxxx"  # Your token here
```

### 3. Deploy all resources
```bash
# Deploy vLLM server
kubectl apply -f vllm-deployment.yaml
kubectl apply -f vllm-service.yaml

# Wait for vLLM to be ready (model download may take 5-10 minutes)
kubectl wait --for=condition=ready pod -l app=vllm-server -n ai-chat --timeout=600s

# Deploy Open WebUI
kubectl apply -f openwebui-deployment.yaml
kubectl apply -f openwebui-service.yaml
```

### 4. Get external IP and access
```bash
kubectl get svc open-webui-service -n ai-chat
```

Wait for `EXTERNAL-IP` to be assigned, then open in your browser:
```
http://<EXTERNAL-IP>
```

## Configuration Files

### vllm-deployment.yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
  namespace: ai-chat
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-server
  template:
    metadata:
      labels:
        app: vllm-server
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:v0.6.3.post1
        args:
          - --model
          - meta-llama/Llama-3.2-3B-Instruct
          - --dtype
          - auto
          - --max-model-len
          - "8192"
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 16Gi
          requests:
            nvidia.com/gpu: 1
            memory: 8Gi
        env:
        - name: HF_TOKEN
          value: "<YOUR_HUGGINGFACE_TOKEN>"
        - name: NCCL_DEBUG
          value: "INFO"
```

### vllm-service.yaml
```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
  namespace: ai-chat
spec:
  selector:
    app: vllm-server
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
```

### openwebui-deployment.yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: open-webui
  namespace: ai-chat
spec:
  replicas: 1
  selector:
    matchLabels:
      app: open-webui
  template:
    metadata:
      labels:
        app: open-webui
    spec:
      containers:
      - name: open-webui
        image: ghcr.io/open-webui/open-webui:main
        ports:
        - containerPort: 8080
        env:
        - name: OPENAI_API_BASE_URL
          value: "http://vllm-service:8000/v1"
        - name: OPENAI_API_KEY
          value: "not-needed"
        volumeMounts:
        - name: webui-data
          mountPath: /app/backend/data
      volumes:
      - name: webui-data
        persistentVolumeClaim:
          claimName: webui-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: webui-pvc
  namespace: ai-chat
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

### openwebui-service.yaml
```yaml
apiVersion: v1
kind: Service
metadata:
  name: open-webui-service
  namespace: ai-chat
spec:
  selector:
    app: open-webui
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

## Monitoring

### Check deployment status
```bash
kubectl get pods -n ai-chat
```

Expected output:
```
NAME                           READY   STATUS    RESTARTS   AGE
open-webui-xxxxxxxxxx-xxxxx    1/1     Running   0          5m
vllm-server-xxxxxxxxxx-xxxxx   1/1     Running   0          10m
ollama-xxxxxxxxxx-xxxxx        1/1     Running   0          15m  # If Ollama installed
```

### View vLLM logs
```bash
kubectl logs -f deployment/vllm-server -n ai-chat
```

### View Open WebUI logs
```bash
kubectl logs -f deployment/open-webui -n ai-chat
```

### View Ollama logs
```bash
kubectl logs -f deployment/ollama -n ai-chat
```

### Check GPU allocation
```bash
kubectl describe node <node-name> | grep -A 10 "Allocated resources"
```

## Clean Up

Remove all deployed resources:
```bash
# Uninstall Ollama (if installed)
helm uninstall ollama -n ai-chat

# Delete namespace and all resources
kubectl delete namespace ai-chat
```

## Security Considerations

For production deployments:

1. **Enable authentication**: Configure Open WebUI user authentication
2. **Use HTTPS**: Set up Ingress with TLS certificates
3. **Network policies**: Restrict traffic between pods
4. **Secrets management**: Use Kubernetes Secrets for tokens
5. **Resource quotas**: Limit resource usage per namespace

## Hardware Specifications

- **GPU**: NVIDIA RTX 4000 Ada
- **VRAM**: 20GB
- **CUDA**: 12.4 (via vLLM v0.6.3.post1)

## Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) - High-performance LLM inference
- [Open WebUI](https://github.com/open-webui/open-webui) - Chat interface
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [Meta Llama](https://llama.meta.com/) - Open-source language models
- [Akamai Cloud](https://www.akamai.com/cloud) - Cloud infrastructure

## Support

For issues and questions:
- vLLM: [GitHub Issues](https://github.com/vllm-project/vllm/issues)
- Open WebUI: [GitHub Issues](https://github.com/open-webui/open-webui/issues)
- Ollama: [GitHub Issues](https://github.com/ollama/ollama/issues)
---

**Note**: This setup is designed for development and testing.
