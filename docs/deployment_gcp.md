# Cloud Deployment Guide - Google Cloud Platform (GCP)

This guide covers deploying the Market Data ETL & Backtesting Engine on Google Cloud Platform.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Infrastructure Setup](#infrastructure-setup)
- [Application Deployment](#application-deployment)
- [Database Configuration](#database-configuration)
- [Monitoring and Logging](#monitoring-and-logging)
- [Cost Optimization](#cost-optimization)

## Architecture Overview

The GCP deployment uses the following services:

- **Compute Engine**: VM instances for running the backtesting engine
- **Cloud SQL**: PostgreSQL with TimescaleDB extension
- **Cloud Storage**: Object storage for historical data
- **Cloud Functions**: Serverless functions for ETL
- **Cloud Run**: Containerized application deployment
- **Cloud Monitoring**: Monitoring and logging
- **VPC**: Network isolation

## Prerequisites

1. GCP Account with billing enabled
2. Google Cloud SDK (gcloud) installed
3. Python 3.9+ installed locally
4. Docker installed

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize gcloud
gcloud init

# Set project
gcloud config set project <project-id>
```

## Infrastructure Setup

### 1. Create VPC Network

```bash
# Create VPC
gcloud compute networks create trading-vpc --subnet-mode=custom

# Create subnet
gcloud compute networks subnets create trading-subnet \
    --network=trading-vpc \
    --region=us-central1 \
    --range=10.0.0.0/24

# Create firewall rules
gcloud compute firewall-rules create allow-ssh \
    --network=trading-vpc \
    --allow=tcp:22 \
    --source-ranges=0.0.0.0/0

gcloud compute firewall-rules create allow-internal \
    --network=trading-vpc \
    --allow=tcp,udp,icmp \
    --source-ranges=10.0.0.0/24
```

### 2. Launch Compute Engine Instance

```bash
# Create instance
gcloud compute instances create trading-engine \
    --zone=us-central1-a \
    --machine-type=n2-standard-4 \
    --subnet=trading-subnet \
    --network-tier=PREMIUM \
    --maintenance-policy=MIGRATE \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --metadata=startup-script='#!/bin/bash
        apt-get update
        apt-get install -y python3-pip git
        '
```

**Recommended Machine Types:**
- **Development**: n2-standard-2 (2 vCPU, 8 GB RAM)
- **Production**: n2-standard-8 (8 vCPU, 32 GB RAM)
- **Heavy Backtesting**: c2-standard-16 (16 vCPU, 64 GB RAM)

### 3. Setup Cloud SQL for PostgreSQL

```bash
# Create Cloud SQL instance
gcloud sql instances create trading-db \
    --database-version=POSTGRES_14 \
    --tier=db-custom-4-16384 \
    --region=us-central1 \
    --network=projects/<project-id>/global/networks/trading-vpc \
    --no-assign-ip \
    --storage-size=100GB \
    --storage-type=SSD \
    --storage-auto-increase

# Set root password
gcloud sql users set-password postgres \
    --instance=trading-db \
    --password=<secure-password>

# Create database
gcloud sql databases create trading_db --instance=trading-db
```

### 4. Create Cloud Storage Bucket

```bash
# Create bucket
gsutil mb -l us-central1 gs://trading-data-bucket/

# Set lifecycle policy
cat > lifecycle.json << EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {
          "type": "SetStorageClass",
          "storageClass": "NEARLINE"
        },
        "condition": {
          "age": 30
        }
      },
      {
        "action": {
          "type": "SetStorageClass",
          "storageClass": "COLDLINE"
        },
        "condition": {
          "age": 90
        }
      }
    ]
  }
}
EOF

gsutil lifecycle set lifecycle.json gs://trading-data-bucket/

# Enable versioning
gsutil versioning set on gs://trading-data-bucket/
```

## Application Deployment

### Method 1: Direct Compute Engine Deployment

1. **SSH into instance:**

```bash
gcloud compute ssh trading-engine --zone=us-central1-a
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
DATABASE_URL=postgresql://postgres:<password>@<cloud-sql-private-ip>:5432/trading_db
GCS_BUCKET=trading-data-bucket
GCP_PROJECT=<project-id>
EOF
```

4. **Run the application:**

```bash
# Start with systemd service
sudo tee /etc/systemd/system/trading-etl.service << EOF
[Unit]
Description=Trading ETL Pipeline
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/Build-a-Market-Data-ETL-Strategy-Backtesting-Engine
ExecStart=/usr/bin/python3 -m etl.pipeline
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable trading-etl
sudo systemctl start trading-etl
```

### Method 2: Cloud Run Deployment

1. **Create Dockerfile:**

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT 8080
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 backtesting.dashboard:app
```

2. **Build and deploy:**

```bash
# Build image using Cloud Build
gcloud builds submit --tag gcr.io/<project-id>/trading-engine

# Deploy to Cloud Run
gcloud run deploy trading-engine \
    --image gcr.io/<project-id>/trading-engine \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --set-env-vars DATABASE_URL=<connection-string>
```

### Method 3: Cloud Functions for ETL

Create a Cloud Function for periodic ETL tasks:

```python
# main.py
import json
from etl import ETLPipeline

def process_etl(request):
    """HTTP Cloud Function for ETL processing."""
    request_json = request.get_json()
    
    pipeline = ETLPipeline()
    result = pipeline.process_batch(request_json.get('batch_id'))
    
    return json.dumps(result)
```

Deploy:

```bash
# Deploy function
gcloud functions deploy trading-etl \
    --runtime python39 \
    --trigger-http \
    --allow-unauthenticated \
    --memory 1024MB \
    --timeout 300s \
    --entry-point process_etl
```

### Method 4: Kubernetes Engine (GKE)

For production-scale deployments:

```bash
# Create GKE cluster
gcloud container clusters create trading-cluster \
    --zone us-central1-a \
    --num-nodes 3 \
    --machine-type n2-standard-4 \
    --enable-autoscaling \
    --min-nodes 1 \
    --max-nodes 10

# Get credentials
gcloud container clusters get-credentials trading-cluster --zone us-central1-a

# Deploy application
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## Database Configuration

1. **Enable TimescaleDB extension:**

```bash
# Connect via Cloud SQL Proxy
cloud_sql_proxy -instances=<project-id>:us-central1:trading-db=tcp:5432 &

# Connect with psql
psql -h localhost -U postgres -d trading_db

# Enable extension
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
# Backups are automatic in Cloud SQL
# Configure backup time
gcloud sql instances patch trading-db \
    --backup-start-time=03:00 \
    --retained-backups-count=7
```

## Monitoring and Logging

### Cloud Monitoring Setup

1. **Install monitoring agent:**

```bash
# On Compute Engine instance
curl -sSO https://dl.google.com/cloudagents/add-monitoring-agent-repo.sh
sudo bash add-monitoring-agent-repo.sh
sudo apt-get update
sudo apt-get install -y stackdriver-agent

sudo service stackdriver-agent start
```

2. **Create custom metrics:**

```python
# In your application
from google.cloud import monitoring_v3

client = monitoring_v3.MetricServiceClient()
project_name = f"projects/{project_id}"

# Create custom metric
series = monitoring_v3.TimeSeries()
series.metric.type = "custom.googleapis.com/trading/backtest_duration"
series.resource.type = "gce_instance"

# Write data point
client.create_time_series(name=project_name, time_series=[series])
```

3. **Create alerts:**

```bash
# Create alert policy via gcloud
gcloud alpha monitoring policies create \
    --notification-channels=<channel-id> \
    --display-name="High CPU Usage" \
    --condition-display-name="CPU > 80%" \
    --condition-threshold-value=0.8 \
    --condition-threshold-duration=300s
```

### Cloud Logging

```python
# Configure logging in application
import google.cloud.logging

client = google.cloud.logging.Client()
client.setup_logging()

# Use standard Python logging
import logging
logging.info("ETL pipeline started")
```

## Cost Optimization

### 1. Use Preemptible VMs for Backtesting

```bash
# Create preemptible instance
gcloud compute instances create trading-batch \
    --zone=us-central1-a \
    --machine-type=n2-standard-8 \
    --preemptible \
    --subnet=trading-subnet
```

Preemptible VMs cost 60-91% less than regular instances.

### 2. Committed Use Discounts

Purchase committed use contracts for 1-3 years to save 37-55%.

```bash
# Create commitment
gcloud compute commitments create trading-commitment \
    --resources=vcpu=8,memory=32GB \
    --plan=12-month \
    --region=us-central1
```

### 3. Storage Lifecycle Management

Already configured - automatically moves data to cheaper storage classes.

### 4. Auto Scaling with Managed Instance Groups

```bash
# Create instance template
gcloud compute instance-templates create trading-template \
    --machine-type=n2-standard-4 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud

# Create managed instance group
gcloud compute instance-groups managed create trading-group \
    --base-instance-name=trading \
    --size=2 \
    --template=trading-template \
    --zone=us-central1-a

# Set autoscaling
gcloud compute instance-groups managed set-autoscaling trading-group \
    --max-num-replicas=10 \
    --min-num-replicas=2 \
    --target-cpu-utilization=0.7 \
    --zone=us-central1-a
```

## Security Best Practices

1. **Use Service Accounts instead of user credentials**
2. **Enable VPC Service Controls**
3. **Use Cloud IAM for access management**
4. **Enable Cloud SQL encryption**
5. **Use Secret Manager for credentials**
6. **Enable Cloud Audit Logs**
7. **Implement least privilege principle**

### Secret Manager Example

```bash
# Store database password
echo -n "<password>" | gcloud secrets create db-password --data-file=-

# Grant access to service account
gcloud secrets add-iam-policy-binding db-password \
    --member="serviceAccount:<service-account>@<project>.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

## Estimated Costs (USD per month)

**Development Environment:**
- Compute Engine n2-standard-2: ~$50
- Cloud SQL db-custom-2-8192: ~$140
- Cloud Storage (100GB): ~$2
- Networking: ~$10
- **Total: ~$202/month**

**Production Environment:**
- Compute Engine n2-standard-8: ~$200
- Cloud SQL db-custom-8-32768: ~$560
- Cloud Storage (1TB): ~$20
- Cloud Load Balancing: ~$20
- Networking: ~$50
- **Total: ~$850/month**

## Troubleshooting

### Common Issues

1. **Cannot connect to Cloud SQL:**
   - Verify private IP configuration
   - Check VPC peering
   - Ensure authorized networks are set

2. **Out of disk space:**
   - Enable auto-increase for Cloud SQL
   - Clean up old Cloud Storage data
   - Increase instance disk size

3. **High latency:**
   - Use Cloud CDN for static content
   - Deploy in multiple regions
   - Optimize database queries

## CI/CD with Cloud Build

```yaml
# cloudbuild.yaml
steps:
  # Build Docker image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/trading-engine', '.']
  
  # Push to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/trading-engine']
  
  # Deploy to Cloud Run
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'trading-engine'
      - '--image=gcr.io/$PROJECT_ID/trading-engine'
      - '--region=us-central1'
      - '--platform=managed'

images:
  - 'gcr.io/$PROJECT_ID/trading-engine'
```

## Next Steps

- Set up multi-region deployment for high availability
- Implement Cloud Armor for DDoS protection
- Configure Cloud CDN for static assets
- Set up disaster recovery plan
- Implement data encryption at rest and in transit
