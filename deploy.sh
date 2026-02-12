#!/bin/bash
set -euo pipefail

# Optional: set CLOUDFLARE_ACCOUNT_ID in the environment to force a specific account.
# If unset, wrangler will use the default account associated with your login/token.

DOMAIN="https://frankensqlite.com"
SQLITE_FILE="spec_evolution_v1.sqlite3"

# Ensure dist exists and is populated
mkdir -p dist
cp visualization_of_the_evolution_of_the_frankensqlite_specs_document_from_inception.html dist/index.html
cp visualization_of_the_evolution_of_the_frankensqlite_specs_document_from_inception.html dist/spec_evolution.html
cp spec_evolution_v1.sqlite3 dist/
cp spec_evolution_v1.sqlite3.config.json dist/
cp og-image.png dist/
cp twitter-image.png dist/
cp frankensqlite_illustration.webp dist/
cp frankensqlite_diagram.webp dist/
cp _headers dist/
cp _routes.json dist/

# Deploy to Cloudflare Pages
echo "Deploying to Cloudflare Pages..."
npx wrangler pages deploy dist --project-name frankensqlite-spec-evolution --commit-dirty=true

# Post-deployment verification
echo ""
echo "Verifying deployment..."
sleep 5  # Give CDN time to propagate

verify_sqlite() {
    local url="$1"
    local max_retries=5
    local retry=0

    while [ $retry -lt $max_retries ]; do
        echo "  Checking $url (attempt $((retry + 1))/$max_retries)..."

        # Check Content-Type header
        content_type=$(curl -sI "$url" | grep -i "^content-type:" | tr -d '\r' | cut -d' ' -f2)

        # Check first 15 bytes of file for SQLite magic header
        magic=$(curl -s "$url" | head -c 15)

        if [[ "$content_type" == "application/octet-stream" ]] && [[ "$magic" == "SQLite format 3" ]]; then
            echo "  OK: Content-Type=$content_type, Magic header verified"
            return 0
        fi

        echo "  WARN: Content-Type=$content_type, Magic='$magic'"
        retry=$((retry + 1))
        sleep 3
    done

    return 1
}

if verify_sqlite "$DOMAIN/$SQLITE_FILE"; then
    echo ""
    echo "Deployment verified successfully!"
    exit 0
else
    echo ""
    echo "DEPLOYMENT VERIFICATION FAILED!"
    echo "The SQLite file is not being served correctly."
    echo "Check that _routes.json is properly deployed and excludes /*.sqlite3"
    exit 1
fi
