#!/bin/bash
set -e  # exit immediately if any command fails

# Create geoip_db directory (if it doesn't exist already)
mkdir -p /app/geoip_db

# Download GeoLite2-Country.mmdb (if it doesn't exist already)
if [ ! -f /app/geoip_db/GeoLite2-Country.mmdb ]; then
  echo "Downloading GeoLite2-Country.mmdb..."
  
  # Download the database using your MaxMind license key 
  # Create an account at https://www.maxmind.com/, create a license key and add it as a secret in your Hugging Face Space
  curl -L -o /app/geoip_db/GeoLite2-Country.tar.gz \
    "https://download.maxmind.com/app/geoip_download?edition_id=GeoLite2-Country&license_key=${MAXMIND_LICENSE_KEY}&suffix=tar.gz"

  # Extract the .mmdb file and remove archive
  tar -xzf /app/geoip_db/GeoLite2-Country.tar.gz -C /app/geoip_db --strip-components=1
  rm /app/geoip_db/GeoLite2-Country.tar.gz
fi

# Start the FastAPI backend with Uvicorn 
uvicorn backend.app:app --host 0.0.0.0 --port 8000 &

# Start the Gradio frontend 
python -m frontend.app
