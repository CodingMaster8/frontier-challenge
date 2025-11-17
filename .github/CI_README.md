# GitHub Actions CI/CD Workflows

This directory contains automated workflows for continuous integration, deployment, and quality assurance.

## Workflows Overview

### 1. **CI/CD Pipeline** (`ci.yml`)
Main workflow that runs on every push and pull request to main/master/develop branches.

**Jobs:**
- **Lint & Format Check**: Runs Ruff formatter and linter
- **Type Checking**: Runs MyPy for static type analysis
- **Tests**: Runs pytest with coverage on Python 3.12 and 3.13
- **Security Scan**: Runs Bandit and Safety checks
- **Documentation Build**: Builds MkDocs documentation
- **Package Build**: Builds distributable package
- **Deploy Docs**: Deploys documentation to GitHub Pages (on main branch only)

**Triggers:**
- Push to main, master, or develop branches
- Pull requests to main, master, or develop branches
- Manual trigger via workflow_dispatch

### 2. **Pull Request Checks** (`pr-checks.yml`)
Specialized workflow for pull request validation with smart file detection.

**Features:**
- **Smart File Detection**: Only runs relevant checks based on changed files
-  **Coverage Comments**: Posts coverage reports directly on PR
-  **Size Check**: Warns if PR is too large
-  **Status Comments**: Posts status updates on the PR

**Jobs:**
- Detects which files changed (Python, docs, tests)
- Runs only relevant checks to save CI time
- Posts coverage reports and status updates

### 3. **Release & Deploy** (`release.yml`)
Automates the release process when a version tag is pushed.

**Jobs:**
- **Pre-release Tests**: Ensures all tests pass before release
-  **Build Distribution**: Creates wheel and source distributions
-  **Create GitHub Release**: Generates changelog and creates release
-  **Publish to PyPI**: Uploads package to PyPI (currently TestPyPI)
-  **Deploy Documentation**: Updates documentation site


**Jobs:**
-  **Comprehensive Tests**: Full test suite with parallel execution
-  **Integration Tests**: Runs evaluation suite
-  **Dependency Security**: Checks for vulnerable dependencies
-  **Report Generation**: Creates summary of all checks

**Schedule:**
- Runs daily at 2 AM UTC
- Can be triggered manually

##  Setup Instructions

### 1. Configure GitHub Secrets

Go to **Settings → Secrets and variables → Actions** and add:

```
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=your-pinecone-index-name
CODECOV_TOKEN=your-codecov-token (optional)
```

### 2. Enable GitHub Pages

1. Go to **Settings → Pages**
2. Set Source to "GitHub Actions"
3. Save changes

Your documentation will be available at: `https://<username>.github.io/<repository>/`

### 3. Configure Branch Protection

Go to **Settings → Branches → Branch protection rules** for `main`:

**Required:**
-  Require a pull request before merging
-  Require status checks to pass before merging
  - Select: `lint`, `test`, `docs`, `build`
-  Require branches to be up to date before merging
-  Require conversation resolution before merging

**Recommended:**
-  Require approvals (1 or more)
-  Dismiss stale pull request approvals
-  Require review from Code Owners
-  Include administrators

### 4. Enable Dependabot

Dependabot is pre-configured in `.github/dependabot.yml` and will:
- Check for dependency updates weekly
- Create PRs for updates automatically
- Label PRs appropriately

No additional setup needed!

##  Monitoring & Reports

### Code Coverage
- Coverage reports are generated for every test run
- Reports are uploaded to Codecov (if token is configured)
- PR comments show coverage diff

### Test Results
- Test results are visible in the Actions tab
- Failed tests show detailed error messages
- Coverage reports available as artifacts

### Security
- Bandit scans for security issues in code
- Safety checks for vulnerable dependencies
- pip-audit runs nightly for dependency vulnerabilities

##  Best Practices

### For Contributors

1. **Before Pushing:**
   ```bash
   # Run tests locally
   pytest

   # Run linting
   ruff format .
   ruff check .

   # Run type checking
   mypy frontier_challenge/
   ```

2. **Pull Request Size:**
   - Keep PRs under 50 files when possible
   - Keep line changes under 1000 when possible
   - Break large changes into smaller, logical PRs

3. **Commit Messages:**
   - Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `chore:`
   - Be descriptive: "Add semantic search functionality" not "Add feature"

## 🔍 Troubleshooting

### Tests Failing in CI but Pass Locally

1. Check environment variables are set correctly
2. Verify Python version matches CI (3.12)
3. Check for file path issues (absolute vs relative)
4. Look for missing dependencies

### Workflow Not Triggering

1. Check branch name matches trigger pattern
2. Verify workflow file is in `.github/workflows/`
3. Check for YAML syntax errors
4. Ensure you have pushed to the correct branch
