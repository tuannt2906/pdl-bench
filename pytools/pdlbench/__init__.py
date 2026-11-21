"""pdlbench — tooling for pdl-bench (Parallel & Distributed Deep Learning Benchmark).

Subpackages/modules:
    data        dataset download + conversion to the .pdlds format
    envinfo     host environment capture
    config      YAML experiment schema and run expansion
    runner      experiment execution (mpirun orchestration, manifests, resume)
    aggregate   result loading, flattening and statistics
    plots       figures for reports (PNG + PDF)
    perfmodel   alpha-beta communication model and training-time model
    report      Markdown report generation
    codemetrics source-code metrics for the AI coding-agent evaluation
"""

__version__ = "1.0.0"
