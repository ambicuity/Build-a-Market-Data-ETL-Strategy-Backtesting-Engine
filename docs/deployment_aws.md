# Cloud Deployment Guide - AWS

This guide covers deploying the Market Data ETL & Backtesting Engine on Amazon Web Services (AWS).

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Infrastructure Setup](#infrastructure-setup)
- [Application Deployment](#application-deployment)
- [Database Configuration](#database-configuration)
- [Monitoring and Logging](#monitoring-and-logging)
- [Cost Optimization](#cost-optimization)

## Architecture Overview

The AWS deployment uses the following services:

- **EC2**: Compute instances for running the backtesting engine
- **RDS**: PostgreSQL with TimescaleDB for time-series data storage
- **S3**: Object storage for historical data and backups
- **Lambda**: Serverless functions for ETL processing
- **CloudWatch**: Monitoring and logging
- **VPC**: Network isolation and security

## Prerequisites

1. AWS Account with appropriate permissions
2. AWS CLI installed and configured
3. Python 3.9+ installed locally
4. Docker installed (optional, for containerized deployment)

```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure
```

## Infrastructure Setup

### 1. Create VPC and Networking

```bash
# Create VPC
aws ec2 create-vpc --cidr-block 10.0.0.0/16 --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=trading-vpc}]'

# Create subnets
aws ec2 create-subnet --vpc-id <vpc-id> --cidr-block 10.0.1.0/24 --availability-zone us-east-1a
aws ec2 create-subnet --vpc-id <vpc-id> --cidr-block 10.0.2.0/24 --availability-zone us-east-1b

# Create Internet Gateway
aws ec2 create-internet-gateway --tag-specifications 'ResourceType=internet-gateway,Tags=[{Key=Name,Value=trading-igw}]'

# Attach to VPC
aws ec2 attach-internet-gateway --vpc-id <vpc-id> --internet-gateway-id <igw-id>
```

### 2. Launch EC2 Instance

```bash
# Launch EC2 instance
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \  # Ubuntu 22.04 LTS
    --count 1 \
    --instance-type t3.xlarge \
    --key-name my-key-pair \
    --security-group-ids <sg-id> \
    --subnet-id <subnet-id> \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=trading-engine}]'
```

**Recommended Instance Types:**
- **Development**: t3.medium (2 vCPU, 4 GB RAM)
- **Production**: c5.2xlarge (8 vCPU, 16 GB RAM)
- **Heavy Backtesting**: c5.4xlarge (16 vCPU, 32 GB RAM)

### 3. Setup RDS PostgreSQL with TimescaleDB

```bash
# Create RDS instance
aws rds create-db-instance \
    --db-instance-identifier trading-db \
    --db-instance-class db.t3.medium \
    --engine postgres \
    --engine-version 14.7 \
    --master-username admin \
    --master-user-password <password> \
    --allocated-storage 100 \
    --vpc-security-group-ids <sg-id> \
    --db-subnet-group-name <subnet-group>
```

**Note**: TimescaleDB extension needs to be installed manually after instance creation.

### 4. Create S3 Bucket for Data Storage

```bash
# Create S3 bucket
aws s3 mb s3://trading-data-bucket --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
    --bucket trading-data-bucket \
    --versioning-configuration Status=Enabled

# Set lifecycle policy for data archival
cat > lifecycle.json << EOF
{
    "Rules": [
        {
            "Id": "Archive old data",
            "Status": "Enabled",
            "Transitions": [
                {
                    "Days": 90,
                    "StorageClass": "GLACIER"
                }
            ]
        }
    ]
}
EOF

aws s3api put-bucket-lifecycle-configuration \
    --bucket trading-data-bucket \
    --lifecycle-configuration file://lifecycle.json
```

## Application Deployment

### Method 1: Direct EC2 Deployment

1. **SSH into EC2 instance:**

```bash
ssh -i my-key-pair.pem ubuntu@<ec2-public-ip>
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
DATABASE_URL=postgresql://admin:<password>@<rds-endpoint>:5432/trading_db
AWS_S3_BUCKET=trading-data-bucket
AWS_REGION=us-east-1
EOF
```

4. **Run the application:**

```bash
# Start ETL pipeline
python3 -m etl.pipeline &

# Start backtesting engine
python3 -m backtesting.engine &
```

### Method 2: Docker Deployment

1. **Create Dockerfile:**

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "etl.pipeline"]
```

2. **Build and push to ECR:**

```bash
# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Create ECR repository
aws ecr create-repository --repository-name trading-engine

# Build and push
docker build -t trading-engine .
docker tag trading-engine:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/trading-engine:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/trading-engine:latest
```

3. **Deploy using ECS:**

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name trading-cluster

# Create task definition and service (via AWS Console or CLI)
```

### Method 3: Lambda for ETL Processing

Create a Lambda function for periodic ETL tasks:

```python
# lambda_function.py
import json
from etl import ETLPipeline

def lambda_handler(event, context):
    pipeline = ETLPipeline()
    result = pipeline.process_batch(event['batch_id'])
    
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

Deploy:

```bash
# Package dependencies
pip install -r requirements.txt -t package/
cd package && zip -r ../lambda.zip .
cd .. && zip -g lambda.zip lambda_function.py

# Create Lambda function
aws lambda create-function \
    --function-name trading-etl \
    --runtime python3.9 \
    --role <lambda-role-arn> \
    --handler lambda_function.lambda_handler \
    --zip-file fileb://lambda.zip \
    --timeout 300 \
    --memory-size 1024
```

## Database Configuration

1. **Connect to RDS and enable TimescaleDB:**

```bash
# Connect via psql
psql -h <rds-endpoint> -U admin -d postgres

# Create database and enable extension
CREATE DATABASE trading_db;
\c trading_db
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
```

3. **Configure automated backups:**

```bash
# Enable automated backups in RDS
aws rds modify-db-instance \
    --db-instance-identifier trading-db \
    --backup-retention-period 7 \
    --preferred-backup-window "03:00-04:00"
```

## Monitoring and Logging

### CloudWatch Setup

1. **Install CloudWatch agent on EC2:**

```bash
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i amazon-cloudwatch-agent.deb
```

2. **Configure CloudWatch agent:**

```json
{
    "metrics": {
        "namespace": "TradingEngine",
        "metrics_collected": {
            "cpu": {
                "measurement": [
                    {"name": "cpu_usage_idle", "rename": "CPU_IDLE", "unit": "Percent"}
                ]
            },
            "mem": {
                "measurement": [
                    {"name": "mem_used_percent", "rename": "MEM_USED", "unit": "Percent"}
                ]
            }
        }
    },
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "/var/log/trading/*.log",
                        "log_group_name": "/aws/trading/engine",
                        "log_stream_name": "{instance_id}"
                    }
                ]
            }
        }
    }
}
```

3. **Create CloudWatch alarms:**

```bash
# CPU alarm
aws cloudwatch put-metric-alarm \
    --alarm-name trading-high-cpu \
    --alarm-description "Alert when CPU exceeds 80%" \
    --metric-name CPUUtilization \
    --namespace AWS/EC2 \
    --statistic Average \
    --period 300 \
    --threshold 80 \
    --comparison-operator GreaterThanThreshold

# Database connections alarm
aws cloudwatch put-metric-alarm \
    --alarm-name trading-db-connections \
    --alarm-description "Alert when DB connections exceed 80" \
    --metric-name DatabaseConnections \
    --namespace AWS/RDS \
    --statistic Average \
    --period 300 \
    --threshold 80 \
    --comparison-operator GreaterThanThreshold
```

## Cost Optimization

### 1. Use Spot Instances for Backtesting

```bash
# Request spot instance
aws ec2 request-spot-instances \
    --spot-price "0.10" \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification file://spot-spec.json
```

### 2. Auto Scaling

```bash
# Create Auto Scaling group
aws autoscaling create-auto-scaling-group \
    --auto-scaling-group-name trading-asg \
    --launch-configuration-name trading-lc \
    --min-size 1 \
    --max-size 5 \
    --desired-capacity 2 \
    --health-check-type ELB \
    --health-check-grace-period 300
```

### 3. S3 Lifecycle Policies

Already configured above - moves old data to Glacier after 90 days.

### 4. Reserved Instances

For production workloads, purchase Reserved Instances for 1-3 year terms to save 30-70%.

## Security Best Practices

1. **Use IAM roles instead of access keys**
2. **Enable VPC Flow Logs**
3. **Use Security Groups to restrict access**
4. **Enable RDS encryption at rest**
5. **Use AWS Secrets Manager for credentials**
6. **Enable CloudTrail for audit logging**

## Estimated Costs

**Development Environment (per month):**
- EC2 t3.medium: ~$30
- RDS db.t3.medium: ~$60
- S3 storage (100GB): ~$2.30
- Data transfer: ~$10
- **Total: ~$102/month**

**Production Environment (per month):**
- EC2 c5.2xlarge: ~$250
- RDS db.r5.xlarge: ~$400
- S3 storage (1TB): ~$23
- CloudWatch: ~$20
- Data transfer: ~$50
- **Total: ~$743/month**

## Troubleshooting

### Common Issues

1. **Connection timeout to RDS:**
   - Check security group rules
   - Verify subnet routing
   - Ensure RDS is publicly accessible (if needed)

2. **High latency:**
   - Check network performance metrics
   - Consider using Placement Groups
   - Optimize database queries

3. **Out of memory:**
   - Increase EC2 instance size
   - Optimize application memory usage
   - Use swap space

## Next Steps

- Set up CI/CD pipeline using AWS CodePipeline
- Implement blue-green deployment
- Configure AWS WAF for API protection
- Set up disaster recovery in another region
