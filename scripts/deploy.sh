#!/bin/bash

# Symbio AI Production Deployment Script
# Complete deployment automation for production environments

set -euo pipefail

# Configuration
PROJECT_NAME="symbio-ai"
VERSION=${1:-"latest"}
ENVIRONMENT=${2:-"production"}
NAMESPACE="symbio-ai"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking deployment prerequisites..."
    
    # Check required tools
    local required_tools=("docker" "kubectl" "helm" "aws")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is not installed or not in PATH"
            exit 1
        fi
    done
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check Helm
    if ! helm version &> /dev/null; then
        log_error "Helm is not properly configured"
        exit 1
    fi
    
    log_success "All prerequisites met"
}

# Build and push Docker images
build_and_push_images() {
    log_info "Building and pushing Docker images..."
    
    local components=("api-gateway" "training" "inference" "data-processor" "monitoring")
    local registry="ghcr.io/symbio-ai"
    
    # Build base image first
    log_info "Building base image..."
    docker build -f deployment/Dockerfile.base -t "$registry/base:$VERSION" .
    docker push "$registry/base:$VERSION"
    
    # Build component images
    for component in "${components[@]}"; do
        log_info "Building $component image..."
        docker build \
            -f "deployment/Dockerfile.$component" \
            -t "$registry/$component:$VERSION" \
            --build-arg BASE_IMAGE="$registry/base:$VERSION" \
            .
        docker push "$registry/$component:$VERSION"
    done
    
    log_success "All images built and pushed"
}

# Setup Kubernetes namespace and secrets
setup_kubernetes() {
    log_info "Setting up Kubernetes resources..."
    
    # Create namespace
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply secrets (if they don't exist)
    if ! kubectl get secret symbio-secrets -n "$NAMESPACE" &> /dev/null; then
        log_warning "Creating placeholder secrets - REPLACE WITH REAL VALUES"
        kubectl create secret generic symbio-secrets \
            --from-literal=jwt-secret="$(openssl rand -base64 32)" \
            --from-literal=db-password="$(openssl rand -base64 16)" \
            --from-literal=redis-password="$(openssl rand -base64 16)" \
            --namespace="$NAMESPACE"
    fi
    
    # Apply ConfigMap
    kubectl apply -f deployment/k8s/configmap.yaml
    
    log_success "Kubernetes resources configured"
}

# Deploy database infrastructure
deploy_infrastructure() {
    log_info "Deploying infrastructure components..."
    
    # Deploy PostgreSQL
    kubectl apply -f deployment/k8s/postgres.yaml
    
    # Deploy Redis
    kubectl apply -f deployment/k8s/redis.yaml
    
    # Wait for databases to be ready
    log_info "Waiting for databases to be ready..."
    kubectl wait --for=condition=ready pod -l app=postgres -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=redis -n "$NAMESPACE" --timeout=300s
    
    log_success "Infrastructure deployed successfully"
}

# Deploy Symbio AI application
deploy_application() {
    log_info "Deploying Symbio AI application..."
    
    # Update Helm values with current version
    local values_file="deployment/helm/values-$ENVIRONMENT.yaml"
    if [[ ! -f "$values_file" ]]; then
        values_file="deployment/helm/values.yaml"
    fi
    
    # Deploy with Helm
    helm upgrade --install "$PROJECT_NAME" deployment/helm/symbio-ai \
        --namespace "$NAMESPACE" \
        --set image.tag="$VERSION" \
        --set environment="$ENVIRONMENT" \
        --values "$values_file" \
        --wait \
        --timeout 10m
    
    log_success "Application deployed successfully"
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Wait for all deployments to be ready
    local deployments=("api-gateway" "training-service" "inference-service" "data-service" "monitoring-service")
    
    for deployment in "${deployments[@]}"; do
        log_info "Checking $deployment..."
        kubectl wait --for=condition=available deployment/"$deployment" -n "$NAMESPACE" --timeout=300s
    done
    
    # Check service endpoints
    log_info "Checking service endpoints..."
    local api_gateway_ip
    api_gateway_ip=$(kubectl get service api-gateway-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [[ -n "$api_gateway_ip" ]]; then
        log_info "Testing API Gateway at $api_gateway_ip..."
        if curl -f "http://$api_gateway_ip:8080/health" &> /dev/null; then
            log_success "API Gateway is responding"
        else
            log_warning "API Gateway health check failed"
        fi
    else
        log_warning "API Gateway external IP not yet assigned"
    fi
    
    log_success "Health checks completed"
}

# Setup monitoring and alerting
setup_monitoring() {
    log_info "Setting up monitoring and alerting..."
    
    # Deploy Prometheus
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --values deployment/monitoring/prometheus-values.yaml \
        --wait
    
    # Deploy Grafana dashboards
    kubectl apply -f deployment/monitoring/dashboards/ -n monitoring
    
    log_success "Monitoring setup completed"
}

# Backup and rollback functionality
create_backup() {
    log_info "Creating deployment backup..."
    
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup Kubernetes resources
    kubectl get all -n "$NAMESPACE" -o yaml > "$backup_dir/k8s-resources.yaml"
    
    # Backup Helm values
    helm get values "$PROJECT_NAME" -n "$NAMESPACE" > "$backup_dir/helm-values.yaml"
    
    # Backup database (if accessible)
    if kubectl get pod -n "$NAMESPACE" -l app=postgres -o name | head -n1 | xargs -I {} kubectl exec {} -n "$NAMESPACE" -- pg_dump -U postgres symbio_ai > "$backup_dir/database.sql" 2>/dev/null; then
        log_success "Database backup created"
    else
        log_warning "Database backup failed - manual backup recommended"
    fi
    
    log_success "Backup created in $backup_dir"
}

rollback_deployment() {
    local revision=${1:-""}
    
    log_info "Rolling back deployment..."
    
    if [[ -n "$revision" ]]; then
        helm rollback "$PROJECT_NAME" "$revision" -n "$NAMESPACE"
    else
        helm rollback "$PROJECT_NAME" -n "$NAMESPACE"
    fi
    
    log_success "Rollback completed"
}

# Cleanup function
cleanup_deployment() {
    log_info "Cleaning up deployment..."
    
    # Remove Helm deployment
    helm uninstall "$PROJECT_NAME" -n "$NAMESPACE" || true
    
    # Remove monitoring
    helm uninstall prometheus -n monitoring || true
    
    # Remove namespaces
    kubectl delete namespace "$NAMESPACE" --ignore-not-found
    kubectl delete namespace monitoring --ignore-not-found
    
    log_success "Cleanup completed"
}

# Performance testing
run_performance_tests() {
    log_info "Running performance tests..."
    
    # Install k6 if not available
    if ! command -v k6 &> /dev/null; then
        log_warning "k6 not found, installing..."
        curl -s https://github.com/grafana/k6/releases/download/v0.47.0/k6-v0.47.0-linux-amd64.tar.gz | tar -xz -C /tmp/
        sudo mv /tmp/k6-v0.47.0-linux-amd64/k6 /usr/local/bin/
    fi
    
    # Run load tests
    local api_gateway_ip
    api_gateway_ip=$(kubectl get service api-gateway-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [[ -n "$api_gateway_ip" ]]; then
        export API_GATEWAY_URL="http://$api_gateway_ip:8080"
        k6 run tests/performance/load-test.js
        log_success "Performance tests completed"
    else
        log_warning "Cannot run performance tests - API Gateway IP not available"
    fi
}

# Security scanning
run_security_scan() {
    log_info "Running security scans..."
    
    # Scan images for vulnerabilities
    local components=("api-gateway" "training" "inference" "data-processor" "monitoring")
    local registry="ghcr.io/symbio-ai"
    
    for component in "${components[@]}"; do
        log_info "Scanning $component image..."
        if command -v trivy &> /dev/null; then
            trivy image "$registry/$component:$VERSION" --exit-code 1 --severity HIGH,CRITICAL || log_warning "$component has security vulnerabilities"
        else
            log_warning "Trivy not installed - skipping vulnerability scan"
        fi
    done
    
    # Scan Kubernetes configuration
    if command -v kubesec &> /dev/null; then
        log_info "Scanning Kubernetes configurations..."
        find deployment/k8s -name "*.yaml" -exec kubesec scan {} \;
    else
        log_warning "Kubesec not installed - skipping Kubernetes security scan"
    fi
    
    log_success "Security scans completed"
}

# Display deployment information
show_deployment_info() {
    log_info "Deployment Information"
    echo "========================="
    echo "Project: $PROJECT_NAME"
    echo "Version: $VERSION"
    echo "Environment: $ENVIRONMENT"
    echo "Namespace: $NAMESPACE"
    echo ""
    
    log_info "Service URLs:"
    kubectl get services -n "$NAMESPACE" -o custom-columns="NAME:.metadata.name,TYPE:.spec.type,EXTERNAL-IP:.status.loadBalancer.ingress[0].ip,PORT:.spec.ports[0].port"
    
    echo ""
    log_info "Pod Status:"
    kubectl get pods -n "$NAMESPACE"
    
    echo ""
    log_info "Deployment Status:"
    kubectl get deployments -n "$NAMESPACE"
}

# Main deployment function
main() {
    log_info "Starting Symbio AI production deployment..."
    log_info "Version: $VERSION, Environment: $ENVIRONMENT"
    
    case "${3:-deploy}" in
        "deploy")
            create_backup
            check_prerequisites
            build_and_push_images
            setup_kubernetes
            deploy_infrastructure
            deploy_application
            setup_monitoring
            run_health_checks
            run_security_scan
            show_deployment_info
            log_success "Deployment completed successfully!"
            ;;
        "rollback")
            rollback_deployment "${4:-}"
            ;;
        "cleanup")
            cleanup_deployment
            ;;
        "test")
            run_performance_tests
            ;;
        "monitor")
            setup_monitoring
            ;;
        "backup")
            create_backup
            ;;
        "info")
            show_deployment_info
            ;;
        *)
            echo "Usage: $0 [version] [environment] [action]"
            echo "Actions: deploy, rollback [revision], cleanup, test, monitor, backup, info"
            exit 1
            ;;
    esac
}

# Trap errors and cleanup
trap 'log_error "Deployment failed at line $LINENO"' ERR

# Run main function
main "$@"