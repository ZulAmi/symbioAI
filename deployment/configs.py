"""
Production deployment configuration and infrastructure as code.

Provides Docker Compose, Kubernetes manifests, Helm charts,
and cloud deployment configurations for Symbio AI.
"""

# Docker Compose Production Configuration
docker_compose_production = '''version: '3.8'

services:
  # API Gateway
  api-gateway:
    image: symbio-ai/api-gateway:${VERSION:-latest}
    ports:
      - "8080:8080"
      - "8443:8443"
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
      - JWT_SECRET=${JWT_SECRET}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:${DB_PASSWORD}@postgres:5432/symbio_ai
    depends_on:
      - redis
      - postgres
    volumes:
      - ./config:/app/config:ro
      - ./ssl:/app/ssl:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - symbio-network

  # Training Service
  training-service:
    image: symbio-ai/training:${VERSION:-latest}
    environment:
      - ENVIRONMENT=production
      - GPU_ENABLED=true
      - CUDA_VISIBLE_DEVICES=0,1
      - DATABASE_URL=postgresql://postgres:${DB_PASSWORD}@postgres:5432/symbio_ai
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
    networks:
      - symbio-network

  # Inference Service
  inference-service:
    image: symbio-ai/inference:${VERSION:-latest}
    environment:
      - ENVIRONMENT=production
      - MODEL_CACHE_SIZE=2GB
      - BATCH_SIZE=32
      - DATABASE_URL=postgresql://postgres:${DB_PASSWORD}@postgres:5432/symbio_ai
    depends_on:
      - postgres
    volumes:
      - ./models:/app/models:ro
    restart: unless-stopped
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
    networks:
      - symbio-network

  # Data Processing Service
  data-service:
    image: symbio-ai/data-processor:${VERSION:-latest}
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://postgres:${DB_PASSWORD}@postgres:5432/symbio_ai
      - S3_BUCKET=${S3_BUCKET}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    depends_on:
      - postgres
    volumes:
      - ./data:/app/data
    restart: unless-stopped
    networks:
      - symbio-network

  # Monitoring Service
  monitoring-service:
    image: symbio-ai/monitoring:${VERSION:-latest}
    ports:
      - "9090:9090"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://postgres:${DB_PASSWORD}@postgres:5432/symbio_ai
      - PROMETHEUS_URL=http://prometheus:9090
    depends_on:
      - postgres
      - prometheus
    restart: unless-stopped
    networks:
      - symbio-network

  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=symbio_ai
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_INITDB_ARGS=--auth-host=scram-sha-256
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - symbio-network

  # Redis Cache
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
    networks:
      - symbio-network

  # Prometheus Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
      - '--web.enable-lifecycle'
    restart: unless-stopped
    networks:
      - symbio-network

  # Grafana Dashboard
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/datasources:/etc/grafana/provisioning/datasources:ro
    restart: unless-stopped
    networks:
      - symbio-network

  # Nginx Load Balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - api-gateway
    restart: unless-stopped
    networks:
      - symbio-network

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  symbio-network:
    driver: bridge
'''

# Kubernetes Production Manifests
kubernetes_namespace = '''apiVersion: v1
kind: Namespace
metadata:
  name: symbio-ai
  labels:
    name: symbio-ai
    environment: production
'''

kubernetes_configmap = '''apiVersion: v1
kind: ConfigMap
metadata:
  name: symbio-config
  namespace: symbio-ai
data:
  config.yaml: |
    environment: production
    logging:
      level: INFO
      format: json
    database:
      type: postgresql
      host: postgres-service
      port: 5432
      database: symbio_ai
    redis:
      host: redis-service
      port: 6379
    api:
      port: 8080
      cors_enabled: true
      rate_limit: 1000
    training:
      gpu_enabled: true
      max_concurrent_jobs: 10
    inference:
      batch_size: 32
      timeout: 30
    monitoring:
      metrics_enabled: true
      prometheus_port: 9090
'''

kubernetes_secrets = '''apiVersion: v1
kind: Secret
metadata:
  name: symbio-secrets
  namespace: symbio-ai
type: Opaque
data:
  jwt-secret: <base64-encoded-jwt-secret>
  db-password: <base64-encoded-db-password>
  redis-password: <base64-encoded-redis-password>
  aws-access-key: <base64-encoded-aws-access-key>
  aws-secret-key: <base64-encoded-aws-secret-key>
'''

kubernetes_postgres = '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: symbio-ai
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: symbio_ai
        - name: POSTGRES_USER
          value: postgres
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: symbio-secrets
              key: db-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: symbio-ai
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: symbio-ai
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
'''

kubernetes_api_gateway = '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
  namespace: symbio-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
    spec:
      containers:
      - name: api-gateway
        image: symbio-ai/api-gateway:latest
        ports:
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: symbio-secrets
              key: jwt-secret
        - name: DATABASE_URL
          value: "postgresql://postgres:$(DB_PASSWORD)@postgres-service:5432/symbio_ai"
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: symbio-secrets
              key: db-password
        volumeMounts:
        - name: config
          mountPath: /app/config
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: symbio-config
---
apiVersion: v1
kind: Service
metadata:
  name: api-gateway-service
  namespace: symbio-ai
spec:
  selector:
    app: api-gateway
  ports:
  - port: 8080
    targetPort: 8080
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-gateway-hpa
  namespace: symbio-ai
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-gateway
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
'''

# Helm Chart Configuration
helm_chart_yaml = '''apiVersion: v2
name: symbio-ai
description: A Helm chart for Symbio AI Production Deployment
type: application
version: 1.0.0
appVersion: "1.0.0"
keywords:
  - ai
  - machine-learning
  - modular-ai
home: https://github.com/symbio-ai/symbio-ai
sources:
  - https://github.com/symbio-ai/symbio-ai
maintainers:
  - name: Symbio AI Team
    email: team@symbio-ai.com
'''

helm_values_yaml = '''# Default values for symbio-ai
replicaCount: 3

image:
  repository: symbio-ai
  pullPolicy: IfNotPresent
  tag: "latest"

nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  annotations: {}
  name: ""

podAnnotations: {}

podSecurityContext:
  fsGroup: 2000

securityContext:
  capabilities:
    drop:
    - ALL
  readOnlyRootFilesystem: true
  runAsNonRoot: true
  runAsUser: 1000

service:
  type: LoadBalancer
  port: 8080

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "1000"
  hosts:
    - host: api.symbio-ai.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: symbio-ai-tls
      hosts:
        - api.symbio-ai.com

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

nodeSelector: {}

tolerations: []

affinity: {}

# Database configuration
postgresql:
  enabled: true
  auth:
    database: symbio_ai
    username: postgres
  primary:
    persistence:
      size: 10Gi
    resources:
      requests:
        memory: 1Gi
        cpu: 500m

# Redis configuration
redis:
  enabled: true
  auth:
    enabled: true
  master:
    persistence:
      size: 1Gi

# Monitoring configuration
prometheus:
  enabled: true
  server:
    persistentVolume:
      size: 5Gi

grafana:
  enabled: true
  persistence:
    enabled: true
    size: 1Gi

# AI-specific configuration
ai:
  training:
    gpu:
      enabled: true
      count: 2
    resources:
      requests:
        memory: 4Gi
        cpu: 2
      limits:
        memory: 8Gi
        cpu: 4
  
  inference:
    replicas: 5
    resources:
      requests:
        memory: 2Gi
        cpu: 1
      limits:
        memory: 4Gi
        cpu: 2

  dataProcessor:
    replicas: 2
    storage:
      size: 50Gi
'''

# Terraform AWS Infrastructure
terraform_aws_main = '''terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"

  name = "${var.project_name}-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway = true
  enable_vpn_gateway = false
  enable_dns_hostnames = true
  enable_dns_support = true

  tags = {
    Project = var.project_name
    Environment = var.environment
  }
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"

  cluster_name    = "${var.project_name}-cluster"
  cluster_version = "1.28"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # EKS Managed Node Groups
  eks_managed_node_groups = {
    general = {
      desired_size = 3
      max_size     = 10
      min_size     = 3

      instance_types = ["t3.large"]
      capacity_type  = "ON_DEMAND"

      k8s_labels = {
        role = "general"
      }
    }

    gpu = {
      desired_size = 2
      max_size     = 5
      min_size     = 0

      instance_types = ["g4dn.xlarge"]
      capacity_type  = "ON_DEMAND"

      k8s_labels = {
        role = "gpu"
        "nvidia.com/gpu" = "true"
      }

      taints = [
        {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }

  tags = {
    Project = var.project_name
    Environment = var.environment
  }
}

# RDS Database
resource "aws_db_instance" "postgres" {
  identifier = "${var.project_name}-postgres"

  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.medium"

  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type          = "gp2"
  storage_encrypted     = true

  db_name  = "symbio_ai"
  username = "postgres"
  password = var.db_password

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.postgres.name

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  skip_final_snapshot = false
  final_snapshot_identifier = "${var.project_name}-postgres-final-snapshot"

  tags = {
    Name = "${var.project_name}-postgres"
    Project = var.project_name
    Environment = var.environment
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "redis" {
  name       = "${var.project_name}-redis-subnet-group"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = "${var.project_name}-redis"
  description                = "Redis cluster for Symbio AI"

  node_type            = "cache.t3.medium"
  port                 = 6379
  parameter_group_name = "default.redis7"

  num_cache_clusters = 3

  subnet_group_name  = aws_elasticache_subnet_group.redis.name
  security_group_ids = [aws_security_group.redis.id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = var.redis_password

  tags = {
    Name = "${var.project_name}-redis"
    Project = var.project_name
    Environment = var.environment
  }
}

# S3 Buckets
resource "aws_s3_bucket" "data" {
  bucket = "${var.project_name}-data-${random_id.bucket_suffix.hex}"

  tags = {
    Name = "${var.project_name}-data"
    Project = var.project_name
    Environment = var.environment
  }
}

resource "aws_s3_bucket" "models" {
  bucket = "${var.project_name}-models-${random_id.bucket_suffix.hex}"

  tags = {
    Name = "${var.project_name}-models"
    Project = var.project_name
    Environment = var.environment
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets           = module.vpc.public_subnets

  enable_deletion_protection = false

  tags = {
    Name = "${var.project_name}-alb"
    Project = var.project_name
    Environment = var.environment
  }
}
'''

# CI/CD Pipeline Configuration
github_actions_workflow = '''name: Symbio AI CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  release:
    types: [published]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: symbio-ai

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=. --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Bandit Security Scan
      run: |
        pip install bandit
        bandit -r . -f json -o bandit-report.json
    
    - name: Run Safety Check
      run: |
        pip install safety
        safety check --json --output safety-report.json
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: "*-report.json"

  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha,prefix=sha-
    
    - name: Build and push Docker images
      uses: docker/build-push-action@v4
      with:
        context: .
        file: ./Dockerfile
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2
    
    - name: Deploy to EKS Staging
      run: |
        aws eks update-kubeconfig --region us-west-2 --name symbio-ai-staging
        helm upgrade --install symbio-ai ./helm/symbio-ai \\
          --namespace symbio-ai-staging \\
          --set image.tag=sha-${{ github.sha }} \\
          --set environment=staging \\
          --values ./helm/values-staging.yaml

  deploy-production:
    needs: build
    runs-on: ubuntu-latest
    if: github.event_name == 'release'
    environment: production
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2
    
    - name: Deploy to EKS Production
      run: |
        aws eks update-kubeconfig --region us-west-2 --name symbio-ai-production
        helm upgrade --install symbio-ai ./helm/symbio-ai \\
          --namespace symbio-ai \\
          --set image.tag=${{ github.event.release.tag_name }} \\
          --set environment=production \\
          --values ./helm/values-production.yaml
    
    - name: Run post-deployment tests
      run: |
        ./scripts/health-check.sh production
        ./scripts/integration-tests.sh production
'''

def create_deployment_files():
    """Create all deployment configuration files."""
    
    deployment_configs = {
        "docker-compose.production.yml": docker_compose_production,
        "k8s/namespace.yaml": kubernetes_namespace,
        "k8s/configmap.yaml": kubernetes_configmap,
        "k8s/secrets.yaml": kubernetes_secrets,
        "k8s/postgres.yaml": kubernetes_postgres,
        "k8s/api-gateway.yaml": kubernetes_api_gateway,
        "helm/Chart.yaml": helm_chart_yaml,
        "helm/values.yaml": helm_values_yaml,
        "terraform/main.tf": terraform_aws_main,
        ".github/workflows/cicd.yml": github_actions_workflow
    }
    
    return deployment_configs

# Environment-specific configurations
production_env = '''# Production Environment Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO

# Database
DATABASE_TYPE=postgresql
DATABASE_HOST=postgres-service
DATABASE_PORT=5432
DATABASE_NAME=symbio_ai
DATABASE_SSL_MODE=require

# Redis
REDIS_HOST=redis-service
REDIS_PORT=6379
REDIS_SSL=true

# API Configuration
API_PORT=8080
API_WORKERS=4
API_TIMEOUT=120
CORS_ENABLED=true
RATE_LIMIT=1000

# Authentication
JWT_ALGORITHM=HS256
JWT_EXPIRATION=3600
SESSION_TIMEOUT=1800

# Training Configuration
TRAINING_GPU_ENABLED=true
TRAINING_MAX_CONCURRENT_JOBS=10
TRAINING_TIMEOUT=86400

# Inference Configuration
INFERENCE_BATCH_SIZE=32
INFERENCE_TIMEOUT=30
MODEL_CACHE_SIZE=2GB

# Monitoring
METRICS_ENABLED=true
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
LOG_FORMAT=json

# AWS Configuration
AWS_REGION=us-west-2
S3_DATA_BUCKET=symbio-ai-data
S3_MODELS_BUCKET=symbio-ai-models

# Security
SSL_ENABLED=true
ENCRYPTION_KEY_SIZE=256
BACKUP_ENABLED=true
BACKUP_RETENTION_DAYS=30
'''

staging_env = '''# Staging Environment Configuration
ENVIRONMENT=staging
LOG_LEVEL=DEBUG

# Database
DATABASE_TYPE=postgresql
DATABASE_HOST=staging-postgres
DATABASE_PORT=5432
DATABASE_NAME=symbio_ai_staging

# Reduced resource limits for staging
API_WORKERS=2
TRAINING_MAX_CONCURRENT_JOBS=2
INFERENCE_BATCH_SIZE=8

# Development features enabled
DEBUG_MODE=true
PROFILING_ENABLED=true
'''

development_env = '''# Development Environment Configuration
ENVIRONMENT=development
LOG_LEVEL=DEBUG

# Local database
DATABASE_TYPE=sqlite
DATABASE_NAME=symbio_ai_dev

# Local Redis (optional)
REDIS_HOST=localhost
REDIS_PORT=6379

# Development settings
DEBUG_MODE=true
API_WORKERS=1
CORS_ENABLED=true
RATE_LIMIT=10000

# Minimal resources for local development
TRAINING_MAX_CONCURRENT_JOBS=1
INFERENCE_BATCH_SIZE=4
'''

# Monitoring and alerting configuration
prometheus_config = '''global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

rule_files:
  - "symbio_ai_rules.yml"

scrape_configs:
  - job_name: 'symbio-api-gateway'
    static_configs:
      - targets: ['api-gateway:8080']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'symbio-training'
    static_configs:
      - targets: ['training-service:8080']
    metrics_path: /metrics
    scrape_interval: 60s

  - job_name: 'symbio-inference'
    static_configs:
      - targets: ['inference-service:8080']
    metrics_path: /metrics
    scrape_interval: 15s

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
'''

if __name__ == "__main__":
    configs = create_deployment_files()
    print("Production deployment configurations created:")
    for filename in configs.keys():
        print(f"  - {filename}")