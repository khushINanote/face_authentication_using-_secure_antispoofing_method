#!/bin/bash

echo "🚀 Starting SpoofGuard Deployment..."

# Check if docker is installed
if ! [ -x "$(command -v docker)" ]; then
  echo '❌ Error: docker is not installed.' >&2
  exit 1
fi

# Check if docker-compose is installed
if ! [ -x "$(command -v docker-compose)" ]; then
  echo '❌ Error: docker-compose is not installed.' >&2
  exit 1
fi

# Create static folders if they don't exist
mkdir -p static/uploads

# Build and start the containers
echo "📦 Building and starting containers..."
docker-compose up -d --build

echo "✅ Deployment complete! App is running at http://localhost"
echo "📝 Use 'docker-compose logs -f' to see real-time logs."
