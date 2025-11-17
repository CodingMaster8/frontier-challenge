#!/bin/bash

# Quick script to set GitHub Secrets using GitHub CLI
# This makes it easy to configure secrets for GitHub Actions
# Reads secrets from .env file in the root directory

set -e

echo "🔐 GitHub Secrets Setup"
echo "======================="
echo ""

# Check if .env file exists
ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
    echo ".env file not found!"
    echo ""
    echo "Please create a .env file in the root directory with the following format:"
    echo ""
    echo "OPENAI_API_KEY=your-openai-api-key"
    echo "PINECONE_API_KEY=your-pinecone-api-key"
    echo "PINECONE_INDEX_NAME=your-pinecone-index-name"
    echo "CODECOV_TOKEN=your-codecov-token  # Optional"
    echo ""
    exit 1
fi

echo "✅ Found .env file"
echo ""

# Load .env file
set -a
source "$ENV_FILE"
set +a

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo " GitHub CLI not found!"
    echo ""
    echo "Install it with:"
    echo "  macOS:   brew install gh"
    echo "  Linux:   See https://cli.github.com/"
    echo ""
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo " Not authenticated with GitHub"
    echo ""
    echo "Please run: gh auth login"
    exit 1
fi

echo " GitHub CLI is installed and authenticated"
echo ""

# Get repository info
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || echo "unknown")
echo "Repository: $REPO"
echo ""

# Function to set a secret from environment variable
set_secret() {
    local secret_name=$1
    local secret_description=$2
    local optional=$3
    local secret_value="${!secret_name}"  # Indirect variable expansion

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Setting: $secret_name"
    echo "Purpose: $secret_description"

    if [ -z "$secret_value" ]; then
        if [ "$optional" = "optional" ]; then
            echo "Status: Optional - Not found in .env, skipping"
            echo "⏭  Skipped"
            echo ""
            return 0
        else
            echo "Status: Required"
            echo " Error: $secret_name not found in .env file"
            echo ""
            return 1
        fi
    fi

    echo "Status: Found in .env"
    if gh secret set "$secret_name" -b"$secret_value" 2>/dev/null; then
        echo " $secret_name set successfully"
    else
        echo " Failed to set $secret_name"
        return 1
    fi
    echo ""
}

echo "This script will set up the following secrets from your .env file:"
echo ""
echo "Required:"
echo "  1. OPENAI_API_KEY       - Your OpenAI API key"
echo "  2. PINECONE_API_KEY     - Your Pinecone API key"
echo "  3. PINECONE_INDEX_NAME  - Your Pinecone index name"
echo ""
echo "Optional:"
echo "  4. CODECOV_TOKEN        - Codecov token for coverage reports"
echo ""

read -p "Continue? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Starting secret configuration..."
echo ""

# Set required secrets
set_secret "OPENAI_API_KEY" "OpenAI API key for LLM operations" "required" || exit 1
set_secret "PINECONE_API_KEY" "Pinecone API key for vector search" "required" || exit 1
set_secret "PINECONE_INDEX_NAME" "Pinecone index name for vector database" "required" || exit 1

# Set optional secrets
set_secret "CODECOV_TOKEN" "Codecov token for code coverage tracking" "optional"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo " Secret configuration complete!"
echo ""

# List configured secrets
echo "Current secrets in repository:"
gh secret list

echo ""
echo "🎉 Next steps:"
echo "   1. Verify secrets are correct: gh secret list"
echo "   2. Commit and push your workflows: git push"
echo "   3. Check Actions tab: gh repo view --web"
echo ""
