#!/bin/bash
set -e
echo "Starting manual deployment..."
mkdir -p dist
echo "Copying HTML..."
cp visualization_of_the_evolution_of_the_frankensqlite_specs_document_from_inception.html dist/index.html
echo "Copying SQLite..."
cp spec_evolution_v1.sqlite3 dist/
echo "Copying Config..."
cp spec_evolution_v1.sqlite3.config.json dist/
echo "Copying Images..."
cp og-image.png dist/
cp twitter-image.png dist/
cp frankensqlite_illustration.webp dist/
cp frankensqlite_diagram.webp dist/
echo "Deploying..."
npx wrangler pages deploy dist --project-name frankensqlite-spec-evolution --branch main --commit-dirty=true
echo "Done."
