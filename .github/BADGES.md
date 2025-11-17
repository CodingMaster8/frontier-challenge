# GitHub Actions Status Badges

Add these badges to your main README.md file to show your CI/CD status:

## Basic Badges

```markdown
![CI/CD](https://github.com/codingmaster8/frontier-challenge/actions/workflows/ci.yml/badge.svg)
![PR Checks](https://github.com/codingmaster8/frontier-challenge/actions/workflows/pr-checks.yml/badge.svg)
![Tests](https://github.com/codingmaster8/frontier-challenge/actions/workflows/nightly.yml/badge.svg)
```

## Badges with Branch

```markdown
![CI/CD](https://github.com/codingmaster8/frontier-challenge/actions/workflows/ci.yml/badge.svg?branch=main)
```

## All Recommended Badges

Copy this block to add to your README.md:

```markdown
<!-- CI/CD Status Badges -->
[![CI/CD Pipeline](https://github.com/codingmaster8/frontier-challenge/actions/workflows/ci.yml/badge.svg)](https://github.com/codingmaster8/frontier-challenge/actions/workflows/ci.yml)
[![PR Checks](https://github.com/codingmaster8/frontier-challenge/actions/workflows/pr-checks.yml/badge.svg)](https://github.com/codingmaster8/frontier-challenge/actions/workflows/pr-checks.yml)
[![Nightly Tests](https://github.com/codingmaster8/frontier-challenge/actions/workflows/nightly.yml/badge.svg)](https://github.com/codingmaster8/frontier-challenge/actions/workflows/nightly.yml)
[![codecov](https://codecov.io/gh/codingmaster8/frontier-challenge/branch/main/graph/badge.svg)](https://codecov.io/gh/codingmaster8/frontier-challenge)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/github/license/codingmaster8/frontier-challenge)](LICENSE)
```

## Custom Badge Colors

You can customize badge colors:

```markdown
![Tests](https://github.com/codingmaster8/frontier-challenge/actions/workflows/ci.yml/badge.svg?style=flat-square)
![Tests](https://github.com/codingmaster8/frontier-challenge/actions/workflows/ci.yml/badge.svg?style=for-the-badge)
```

Available styles:
- `flat` (default)
- `flat-square`
- `plastic`
- `for-the-badge`
- `social`

## Example README Section

Here's a complete example of how to structure your README with badges:

```markdown
# Frontier Challenge

> AI-powered Brazilian Funds Database Analysis

[![CI/CD Pipeline](https://github.com/codingmaster8/frontier-challenge/actions/workflows/ci.yml/badge.svg)](https://github.com/codingmaster8/frontier-challenge/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/codingmaster8/frontier-challenge/branch/main/graph/badge.svg)](https://codecov.io/gh/codingmaster8/frontier-challenge)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/github/license/codingmaster8/frontier-challenge)](LICENSE)

## Features

- 🤖 AI-powered fund analysis
- 📊 Interactive visualizations
- 🔍 Semantic search
- 💾 Comprehensive database

## Quick Start

... (rest of your README)
```

## Shields.io Custom Badges

You can also create custom badges at [shields.io](https://shields.io/):

```markdown
![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen)
![Passing](https://img.shields.io/badge/tests-passing-success)
![Status](https://img.shields.io/badge/status-production-blue)
```

## Notes

- Badges update automatically when workflows run
- Clicking badges links to workflow runs
- Green badge = passing, Red badge = failing
- Badges cache for ~5 minutes
