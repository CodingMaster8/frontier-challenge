#!/bin/bash
# Evaluation runner script for the Financial Agent
# Usage: ./evals/run.sh [command]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Print colored message
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check requirements
check_requirements() {
    print_info "Checking requirements..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi

    # Check environment variables
    if [ -z "$OPENAI_API_KEY" ]; then
        print_warning "OPENAI_API_KEY is not set (required for semantic search)"
    fi

    if [ -z "$PINECONE_API_KEY" ]; then
        print_warning "PINECONE_API_KEY is not set (required for semantic search)"
    fi

    # Check database
    if [ ! -f "data/br_funds.db" ]; then
        print_error "Database not found at data/br_funds.db"
        exit 1
    fi

    print_success "Requirements check passed"
}

# Create results directory
setup() {
    print_info "Setting up evaluation environment..."
    mkdir -p evals/results
    print_success "Setup complete"
}

# Test the framework
test_framework() {
    print_info "Testing evaluation framework..."
    python3 evals/test_framework.py
}

# Run semantic search evaluation
run_semantic() {
    print_info "Running semantic search evaluation..."
    python3 -m evals.run_semantic_eval
}

# Run SQL filter evaluation
run_sql() {
    print_info "Running SQL filter evaluation..."
    python3 -m evals.run_sql_eval
}

# Run holdings search evaluation
run_holdings() {
    print_info "Running holdings search evaluation..."
    python3 -m evals.run_holdings_eval
}

# Run all evaluations
run_all() {
    print_info "Running comprehensive evaluation..."
    python3 -m evals.run_all_evals
}

# Quick test (basic cases only)
run_quick() {
    print_info "Running quick test (basic cases only)..."
    python3 evals/quickstart.py --quick
}

# Show results
show_results() {
    print_info "Evaluation results:"
    echo ""

    if [ -d "evals/results" ]; then
        ls -lht evals/results/ | head -10

        # Count files
        json_files=$(find evals/results -name "*.json" | wc -l | tr -d ' ')
        html_files=$(find evals/results -name "*.html" | wc -l | tr -d ' ')

        echo ""
        print_success "Found $json_files JSON result files"
        print_success "Found $html_files HTML report files"

        # Show latest HTML report
        latest_html=$(ls -t evals/results/*.html 2>/dev/null | head -1)
        if [ -n "$latest_html" ]; then
            echo ""
            print_info "Latest HTML report: $latest_html"
            print_info "Open with: open $latest_html"
        fi
    else
        print_warning "No results directory found. Run evaluations first."
    fi
}

# Clean results
clean() {
    print_info "Cleaning evaluation results..."
    rm -rf evals/results/*.json
    rm -rf evals/results/*.html
    print_success "Results cleaned"
}

# Show usage
usage() {
    cat << EOF
${BLUE}Financial Agent Evaluation Runner${NC}

${GREEN}Usage:${NC}
    ./evals/run.sh [command]

${GREEN}Commands:${NC}
    test            Test the evaluation framework
    semantic        Run semantic search evaluation
    sql             Run SQL filter evaluation
    holdings        Run holdings search evaluation
    all             Run all evaluations (comprehensive)
    quick           Run quick test (basic cases only)

    results         Show evaluation results
    clean           Clean result files
    setup           Setup evaluation environment
    check           Check requirements

    help            Show this help message

${GREEN}Examples:${NC}
    ./evals/run.sh test          # Test the framework
    ./evals/run.sh semantic      # Evaluate semantic search
    ./evals/run.sh all           # Run comprehensive evaluation
    ./evals/run.sh results       # View results

${GREEN}Environment Variables:${NC}
    OPENAI_API_KEY      Required for semantic search and SQL tools
    PINECONE_API_KEY    Required for semantic search

${GREEN}Output:${NC}
    Results are saved to: evals/results/
    - JSON files: Detailed test results
    - HTML files: Interactive reports

EOF
}

# Main command dispatcher
main() {
    case "${1:-help}" in
        test)
            check_requirements
            setup
            test_framework
            ;;
        semantic)
            check_requirements
            setup
            run_semantic
            ;;
        sql)
            check_requirements
            setup
            run_sql
            ;;
        holdings)
            check_requirements
            setup
            run_holdings
            ;;
        all)
            check_requirements
            setup
            run_all
            ;;
        quick)
            check_requirements
            setup
            run_quick
            ;;
        results)
            show_results
            ;;
        clean)
            clean
            ;;
        setup)
            setup
            ;;
        check)
            check_requirements
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            usage
            exit 1
            ;;
    esac
}

# Run main with all arguments
main "$@"
