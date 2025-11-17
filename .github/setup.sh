#!/bin/bash

# GitHub Actions CI/CD Setup Script
# This script helps you set up GitHub Actions for your repository

set -e

echo "🚀 GitHub Actions CI/CD Setup Script"
echo "===================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: Not a git repository${NC}"
    echo "Please run this script from the root of your git repository."
    exit 1
fi

echo -e "${GREEN}✅ Git repository detected${NC}"
echo ""

# Check if GitHub CLI is installed
if command -v gh &> /dev/null; then
    echo -e "${GREEN}✅ GitHub CLI is installed${NC}"
    GH_CLI_AVAILABLE=true
else
    echo -e "${YELLOW}⚠️  GitHub CLI not found${NC}"
    echo "Install it from: https://cli.github.com/"
    echo "You'll need to configure secrets manually"
    GH_CLI_AVAILABLE=false
fi
echo ""

# Check current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: ${CURRENT_BRANCH}"
echo ""

# Prompt for secrets
echo "📝 Configuration"
echo "---------------"
echo ""
echo "You'll need to configure these secrets in GitHub:"
echo "  • OPENAI_API_KEY"
echo "  • PINECONE_API_KEY"
echo "  • PINECONE_INDEX_NAME"
echo "  • CODECOV_TOKEN (optional)"
echo ""

if [ "$GH_CLI_AVAILABLE" = true ]; then
    read -p "Would you like to set these secrets now? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Setting up secrets..."

        read -sp "Enter OPENAI_API_KEY: " OPENAI_KEY
        echo ""
        if [ -n "$OPENAI_KEY" ]; then
            gh secret set OPENAI_API_KEY -b"$OPENAI_KEY"
            echo -e "${GREEN}✅ OPENAI_API_KEY set${NC}"
        fi

        read -sp "Enter PINECONE_API_KEY: " PINECONE_KEY
        echo ""
        if [ -n "$PINECONE_KEY" ]; then
            gh secret set PINECONE_API_KEY -b"$PINECONE_KEY"
            echo -e "${GREEN}✅ PINECONE_API_KEY set${NC}"
        fi

        read -p "Enter PINECONE_INDEX_NAME: " PINECONE_INDEX
        echo ""
        if [ -n "$PINECONE_INDEX" ]; then
            gh secret set PINECONE_INDEX_NAME -b"$PINECONE_INDEX"
            echo -e "${GREEN}✅ PINECONE_INDEX_NAME set${NC}"
        fi

        read -p "Enter CODECOV_TOKEN (optional, press Enter to skip): " CODECOV_TOKEN
        echo ""
        if [ -n "$CODECOV_TOKEN" ]; then
            gh secret set CODECOV_TOKEN -b"$CODECOV_TOKEN"
            echo -e "${GREEN}✅ CODECOV_TOKEN set${NC}"
        fi
    fi
else
    echo "To set secrets manually:"
    echo "1. Go to: https://github.com/$(git remote get-url origin | sed 's/.*github.com[:/]\(.*\)\.git/\1/')/settings/secrets/actions"
    echo "2. Click 'New repository secret'"
    echo "3. Add each secret listed above"
fi
echo ""

# Check if workflows exist
echo "📋 Checking workflow files..."
WORKFLOW_DIR=".github/workflows"
if [ -d "$WORKFLOW_DIR" ]; then
    echo -e "${GREEN}✅ Workflows directory exists${NC}"
    echo "Workflows found:"
    ls -1 "$WORKFLOW_DIR"/*.yml 2>/dev/null || echo "  (none)"
else
    echo -e "${RED}❌ Workflows directory not found${NC}"
    exit 1
fi
echo ""

# Enable GitHub Pages
echo "📚 GitHub Pages Setup"
echo "--------------------"
if [ "$GH_CLI_AVAILABLE" = true ]; then
    echo "Enabling GitHub Pages with source: GitHub Actions"
    gh api repos/{owner}/{repo}/pages \
        -X POST \
        -F source[branch]=gh-pages \
        -F source[path]=/ 2>/dev/null || \
        echo -e "${YELLOW}Note: GitHub Pages may already be configured or needs manual setup${NC}"
else
    echo "To enable GitHub Pages manually:"
    echo "1. Go to: Settings → Pages"
    echo "2. Set Source to 'GitHub Actions'"
    echo "3. Save changes"
fi
echo ""

# Branch protection
echo "🔒 Branch Protection"
echo "-------------------"
echo "Setting up branch protection for 'main' branch..."
echo ""
echo "Recommended settings:"
echo "  ✅ Require pull request before merging"
echo "  ✅ Require status checks: lint, test, docs, build"
echo "  ✅ Require branches to be up to date"
echo "  ✅ Require conversation resolution"
echo ""

if [ "$GH_CLI_AVAILABLE" = true ]; then
    read -p "Would you like to enable basic branch protection? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        gh api repos/{owner}/{repo}/branches/main/protection \
            -X PUT \
            -f required_status_checks[strict]=true \
            -f 'required_status_checks[contexts][]=lint' \
            -f 'required_status_checks[contexts][]=test' \
            -f 'required_status_checks[contexts][]=docs' \
            -f 'required_status_checks[contexts][]=build' \
            -f required_pull_request_reviews[required_approving_review_count]=1 \
            -f required_pull_request_reviews[dismiss_stale_reviews]=true \
            -F enforce_admins=false \
            2>/dev/null && \
            echo -e "${GREEN}✅ Branch protection enabled${NC}" || \
            echo -e "${YELLOW}⚠️  Could not enable automatically. Please configure manually.${NC}"
    fi
else
    echo "To configure branch protection manually:"
    echo "1. Go to: Settings → Branches → Branch protection rules"
    echo "2. Add rule for 'main' branch"
    echo "3. Apply the recommended settings above"
fi
echo ""

# Test the setup
echo "🧪 Testing Setup"
echo "---------------"
read -p "Would you like to test the workflows by pushing a commit? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating test commit..."
    git add .github/
    git commit -m "ci: Add GitHub Actions CI/CD workflows" || echo "No changes to commit"

    read -p "Push to remote? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git push origin "$CURRENT_BRANCH"
        echo -e "${GREEN}✅ Pushed to remote${NC}"
        echo ""
        echo "Check the Actions tab to see the workflows running:"
        if [ "$GH_CLI_AVAILABLE" = true ]; then
            echo "$(gh repo view --json url -q .url)/actions"
        else
            echo "https://github.com/$(git remote get-url origin | sed 's/.*github.com[:/]\(.*\)\.git/\1/')/actions"
        fi
    fi
fi
echo ""

# Final summary
echo "✨ Setup Complete!"
echo "================="
echo ""
echo "Next steps:"
echo "1. Check the Actions tab to see workflows running"
echo "2. Review and adjust branch protection rules if needed"
echo "3. Configure any remaining secrets"
echo "4. Create a test pull request to see PR checks in action"
echo ""
echo "Documentation: .github/README.md"
echo ""
echo "Happy coding! 🚀"
