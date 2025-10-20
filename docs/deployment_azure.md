# Cloud Deployment Guide - Microsoft Azure

This guide covers deploying the Market Data ETL & Backtesting Engine on Microsoft Azure.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Infrastructure Setup](#infrastructure-setup)
- [Application Deployment](#application-deployment)
- [Database Configuration](#database-configuration)
- [Monitoring and Logging](#monitoring-and-logging)
- [Cost Optimization](#cost-optimization)

## Architecture Overview

The Azure deployment uses the following services:

- **Virtual Machines**: Compute instances for running the backtesting engine
- **Azure Database for PostgreSQL**: Managed PostgreSQL with TimescaleDB
- **Blob Storage**: Object storage for historical data
- **Azure Functions**: Serverless compute for ETL
- **Container Instances / AKS**: Containerized deployments
- **Azure Monitor**: Monitoring and logging
- **Virtual Network**: Network isolation

## Prerequisites

1. Azure subscription with appropriate permissions
2. Azure CLI installed
3. Python 3.9+ installed locally
4. Docker installed

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login

# Set subscription
az account set --subscription <subscription-id>
```

## Infrastructure Setup

### 1. Create Resource Group and Virtual Network

```bash
# Create resource group
az group create \
    --name trading-rg \
    --location eastus

# Create virtual network
az network vnet create \
    --resource-group trading-rg \
    --name trading-vnet \
    --address-prefix 10.0.0.0/16 \
    --subnet-name trading-subnet \
    --subnet-prefix 10.0.1.0/24

# Create network security group
az network nsg create \
    --resource-group trading-rg \
    --name trading-nsg

# Add security rules
az network nsg rule create \
    --resource-group trading-rg \
    --nsg-name trading-nsg \
    --name allow-ssh \
    --priority 1000 \
    --source-address-prefixes '*' \
    --destination-port-ranges 22 \
    --access Allow \
    --protocol Tcp
```

### 2. Create Virtual Machine

```bash
# Create VM
az vm create \
    --resource-group trading-rg \
    --name trading-vm \
    --image UbuntuLTS \
    --size Standard_D4s_v3 \
    --vnet-name trading-vnet \
    --subnet trading-subnet \
    --nsg trading-nsg \
    --admin-username azureuser \
    --generate-ssh-keys \
    --public-ip-sku Standard

# Open port for web dashboard (optional)
az vm open-port \
    --resource-group trading-rg \
    --name trading-vm \
    --port 5000
```

**Recommended VM Sizes:**
- **Development**: Standard_D2s_v3 (2 vCPU, 8 GB RAM)
- **Production**: Standard_D8s_v3 (8 vCPU, 32 GB RAM)
- **Heavy Backtesting**: Standard_F16s_v2 (16 vCPU, 32 GB RAM)

### 3. Setup Azure Database for PostgreSQL

```bash
# Create PostgreSQL server
az postgres flexible-server create \
    --resource-group trading-rg \
    --name trading-db-server \
    --location eastus \
    --admin-user dbadmin \
    --admin-password '<secure-password>' \
    --sku-name Standard_D4s_v3 \
    --tier GeneralPurpose \
    --storage-size 128 \
    --version 14 \
    --vnet trading-vnet \
    --subnet trading-subnet \
    --yes

# Create database
az postgres flexible-server db create \
    --resource-group trading-rg \
    --server-name trading-db-server \
    --database-name trading_db

# Configure firewall (for development)
az postgres flexible-server firewall-rule create \
    --resource-group trading-rg \
    --name trading-db-server \
    --rule-name allow-azure-services \
    --start-ip-address 0.0.0.0 \
    --end-ip-address 0.0.0.0
```

### 4. Create Storage Account and Container

```bash
# Create storage account
az storage account create \
    --name tradingdatastorage \
    --resource-group trading-rg \
    --location eastus \
    --sku Standard_LRS \
    --kind StorageV2

# Get storage account key
STORAGE_KEY=$(az storage account keys list \
    --resource-group trading-rg \
    --account-name tradingdatastorage \
    --query '[0].value' \
    --output tsv)

# Create blob container
az storage container create \
    --name trading-data \
    --account-name tradingdatastorage \
    --account-key $STORAGE_KEY

# Set lifecycle management
cat > lifecycle.json << EOF
{
  "rules": [
    {
      "enabled": true,
      "name": "move-to-cool",
      "type": "Lifecycle",
      "definition": {
        "actions": {
          "baseBlob": {
            "tierToCool": {
              "daysAfterModificationGreaterThan": 30
            },
            "tierToArchive": {
              "daysAfterModificationGreaterThan": 90
            }
          }
        },
        "filters": {
          "blobTypes": ["blockBlob"]
        }
      }
    }
  ]
}
EOF

az storage account management-policy create \
    --account-name tradingdatastorage \
    --policy @lifecycle.json \
    --resource-group trading-rg
```

## Application Deployment

### Method 1: Direct VM Deployment

1. **SSH into VM:**

```bash
# Get public IP
VM_IP=$(az vm show -d -g trading-rg -n trading-vm --query publicIps -o tsv)

# SSH
ssh azureuser@$VM_IP
```

2. **Install dependencies:**

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python and dependencies
sudo apt-get install -y python3.9 python3-pip git

# Clone repository
git clone https://github.com/ambicuity/Build-a-Market-Data-ETL-Strategy-Backtesting-Engine.git
cd Build-a-Market-Data-ETL-Strategy-Backtesting-Engine

# Install Python packages
pip3 install -r requirements.txt
```

3. **Configure environment:**

```bash
# Create .env file
cat > .env << EOF
DATABASE_URL=postgresql://dbadmin:<password>@trading-db-server.postgres.database.azure.com:5432/trading_db
AZURE_STORAGE_ACCOUNT=tradingdatastorage
AZURE_STORAGE_KEY=<storage-key>
AZURE_STORAGE_CONTAINER=trading-data
EOF
```

4. **Create systemd service:**

```bash
sudo tee /etc/systemd/system/trading-etl.service << EOF
[Unit]
Description=Trading ETL Pipeline
After=network.target

[Service]
Type=simple
User=azureuser
WorkingDirectory=/home/azureuser/Build-a-Market-Data-ETL-Strategy-Backtesting-Engine
Environment="PATH=/home/azureuser/.local/bin:/usr/bin"
ExecStart=/usr/bin/python3 -m etl.pipeline
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable trading-etl
sudo systemctl start trading-etl
```

### Method 2: Container Instances

1. **Create container registry:**

```bash
# Create ACR
az acr create \
    --resource-group trading-rg \
    --name tradingregistry \
    --sku Basic

# Login to ACR
az acr login --name tradingregistry
```

2. **Build and push image:**

```bash
# Build image
docker build -t tradingregistry.azurecr.io/trading-engine:latest .

# Push to ACR
docker push tradingregistry.azurecr.io/trading-engine:latest
```

3. **Deploy to Container Instances:**

```bash
# Create container instance
az container create \
    --resource-group trading-rg \
    --name trading-container \
    --image tradingregistry.azurecr.io/trading-engine:latest \
    --cpu 2 \
    --memory 4 \
    --registry-login-server tradingregistry.azurecr.io \
    --registry-username tradingregistry \
    --registry-password $(az acr credential show --name tradingregistry --query passwords[0].value -o tsv) \
    --environment-variables \
        DATABASE_URL='<connection-string>' \
        AZURE_STORAGE_ACCOUNT='tradingdatastorage'
```

### Method 3: Azure Kubernetes Service (AKS)

For production-scale deployments:

```bash
# Create AKS cluster
az aks create \
    --resource-group trading-rg \
    --name trading-aks \
    --node-count 3 \
    --node-vm-size Standard_D4s_v3 \
    --enable-addons monitoring \
    --generate-ssh-keys \
    --attach-acr tradingregistry

# Get credentials
az aks get-credentials \
    --resource-group trading-rg \
    --name trading-aks

# Deploy application
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Example Kubernetes deployment:

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-engine
spec:
  replicas: 3
  selector:
    matchLabels:
      app: trading-engine
  template:
    metadata:
      labels:
        app: trading-engine
    spec:
      containers:
      - name: trading-engine
        image: tradingregistry.azurecr.io/trading-engine:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: trading-secrets
              key: database-url
```

### Method 4: Azure Functions for ETL

```bash
# Create function app
az functionapp create \
    --resource-group trading-rg \
    --consumption-plan-location eastus \
    --runtime python \
    --runtime-version 3.9 \
    --functions-version 4 \
    --name trading-etl-func \
    --storage-account tradingdatastorage

# Deploy function code
func azure functionapp publish trading-etl-func
```

## Database Configuration

1. **Connect and enable TimescaleDB:**

```bash
# Connect via psql
psql "host=trading-db-server.postgres.database.azure.com port=5432 dbname=trading_db user=dbadmin sslmode=require"

# Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;
```

2. **Create tables:**

```sql
-- Create tick data table
CREATE TABLE ticks (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    price DOUBLE PRECISION,
    volume BIGINT
);

-- Convert to hypertable
SELECT create_hypertable('ticks', 'time');

-- Create indexes
CREATE INDEX idx_ticks_symbol_time ON ticks (symbol, time DESC);

-- Enable compression
ALTER TABLE ticks SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);

-- Add compression policy
SELECT add_compression_policy('ticks', INTERVAL '7 days');
```

3. **Configure automated backups:**

```bash
# Backups are automatic in Azure Database for PostgreSQL
# Configure retention period
az postgres flexible-server parameter set \
    --resource-group trading-rg \
    --server-name trading-db-server \
    --name backup_retention_days \
    --value 30
```

## Monitoring and Logging

### Azure Monitor Setup

1. **Enable Application Insights:**

```bash
# Create Application Insights
az monitor app-insights component create \
    --app trading-insights \
    --location eastus \
    --resource-group trading-rg \
    --application-type web

# Get instrumentation key
INSIGHTS_KEY=$(az monitor app-insights component show \
    --app trading-insights \
    --resource-group trading-rg \
    --query instrumentationKey \
    --output tsv)
```

2. **Configure monitoring in application:**

```python
# In your application
from applicationinsights import TelemetryClient

tc = TelemetryClient('<instrumentation-key>')

# Track metrics
tc.track_metric('backtest_duration', duration)
tc.track_event('strategy_executed', {'strategy': 'mean_reversion'})
tc.flush()
```

3. **Create alerts:**

```bash
# Create action group
az monitor action-group create \
    --name trading-alerts \
    --resource-group trading-rg \
    --short-name trading

# Create metric alert
az monitor metrics alert create \
    --name high-cpu-alert \
    --resource-group trading-rg \
    --scopes "/subscriptions/<subscription-id>/resourceGroups/trading-rg/providers/Microsoft.Compute/virtualMachines/trading-vm" \
    --condition "avg Percentage CPU > 80" \
    --description "Alert when CPU exceeds 80%" \
    --evaluation-frequency 5m \
    --window-size 15m \
    --action trading-alerts
```

### Log Analytics

```bash
# Create Log Analytics workspace
az monitor log-analytics workspace create \
    --resource-group trading-rg \
    --workspace-name trading-logs \
    --location eastus

# Enable VM insights
az vm extension set \
    --resource-group trading-rg \
    --vm-name trading-vm \
    --name OmsAgentForLinux \
    --publisher Microsoft.EnterpriseCloud.Monitoring \
    --settings "{'workspaceId':'<workspace-id>'}" \
    --protected-settings "{'workspaceKey':'<workspace-key>'}"
```

## Cost Optimization

### 1. Use Spot VMs for Backtesting

```bash
# Create spot VM (up to 90% discount)
az vm create \
    --resource-group trading-rg \
    --name trading-spot-vm \
    --image UbuntuLTS \
    --size Standard_D8s_v3 \
    --priority Spot \
    --max-price -1 \
    --eviction-policy Deallocate
```

### 2. Azure Reservations

Purchase reserved instances for 1-3 years to save 40-72%.

```bash
# List available reservations
az reservations reservation list
```

### 3. Auto-shutdown for Development VMs

```bash
# Enable auto-shutdown
az vm auto-shutdown \
    --resource-group trading-rg \
    --name trading-vm \
    --time 1900 \
    --timezone "Eastern Standard Time"
```

### 4. Use Azure Advisor

```bash
# Get cost recommendations
az advisor recommendation list --category Cost
```

### 5. Implement Auto-scaling

```bash
# Create VM scale set
az vmss create \
    --resource-group trading-rg \
    --name trading-vmss \
    --image UbuntuLTS \
    --upgrade-policy-mode automatic \
    --instance-count 2 \
    --vm-sku Standard_D4s_v3

# Configure autoscaling
az monitor autoscale create \
    --resource-group trading-rg \
    --resource trading-vmss \
    --resource-type Microsoft.Compute/virtualMachineScaleSets \
    --name autoscale-trading \
    --min-count 2 \
    --max-count 10 \
    --count 2

# Add scale-out rule
az monitor autoscale rule create \
    --resource-group trading-rg \
    --autoscale-name autoscale-trading \
    --condition "Percentage CPU > 70 avg 5m" \
    --scale out 2
```

## Security Best Practices

1. **Use Managed Identities**
2. **Enable Azure Key Vault for secrets**
3. **Implement Network Security Groups**
4. **Enable Azure Security Center**
5. **Use Azure Active Directory authentication**
6. **Enable encryption at rest and in transit**
7. **Implement Azure Policy for compliance**

### Key Vault Example

```bash
# Create Key Vault
az keyvault create \
    --name trading-keyvault \
    --resource-group trading-rg \
    --location eastus

# Store secrets
az keyvault secret set \
    --vault-name trading-keyvault \
    --name db-password \
    --value '<secure-password>'

# Grant VM access
az vm identity assign \
    --resource-group trading-rg \
    --name trading-vm

VM_PRINCIPAL=$(az vm show \
    --resource-group trading-rg \
    --name trading-vm \
    --query identity.principalId \
    --output tsv)

az keyvault set-policy \
    --name trading-keyvault \
    --object-id $VM_PRINCIPAL \
    --secret-permissions get list
```

## Estimated Costs (USD per month)

**Development Environment:**
- VM Standard_D2s_v3: ~$70
- Azure Database for PostgreSQL: ~$150
- Blob Storage (100GB): ~$2
- Bandwidth: ~$10
- **Total: ~$232/month**

**Production Environment:**
- VM Standard_D8s_v3: ~$280
- Azure Database for PostgreSQL (high availability): ~$600
- Blob Storage (1TB): ~$20
- Application Insights: ~$25
- Bandwidth: ~$50
- **Total: ~$975/month**

**With Reservations (3-year):**
- Savings: ~40-50% on compute
- Production Total: ~$600/month

## Troubleshooting

### Common Issues

1. **Cannot connect to PostgreSQL:**
   - Check firewall rules
   - Verify SSL/TLS requirements
   - Check network security group

2. **Slow performance:**
   - Check VM metrics in Azure Monitor
   - Verify database performance tier
   - Optimize queries

3. **Storage access denied:**
   - Verify storage account keys
   - Check SAS token expiration
   - Verify network access

## CI/CD with Azure DevOps

```yaml
# azure-pipelines.yml
trigger:
  - main

pool:
  vmImage: 'ubuntu-latest'

steps:
- task: UsePythonVersion@0
  inputs:
    versionSpec: '3.9'

- script: |
    pip install -r requirements.txt
    pytest tests/
  displayName: 'Run tests'

- task: Docker@2
  inputs:
    containerRegistry: 'tradingregistry'
    repository: 'trading-engine'
    command: 'buildAndPush'
    Dockerfile: '**/Dockerfile'
    tags: |
      $(Build.BuildId)
      latest

- task: AzureWebAppContainer@1
  inputs:
    azureSubscription: '<subscription>'
    appName: 'trading-app'
    containers: 'tradingregistry.azurecr.io/trading-engine:latest'
```

## Next Steps

- Set up Azure Front Door for global distribution
- Implement Azure Traffic Manager for failover
- Configure Azure Backup for disaster recovery
- Enable Azure DDoS Protection
- Implement Azure Sentinel for security monitoring
