# Layout Optimization

This project explores algorithm-driven layout optimization and geometry processing under manufacturing constraints.

It focuses on enforcing minimum feature width, improving geometric smoothness, and detecting structural irregularities through a combination of computational geometry and constraint-aware optimization.

## Features

- Constraint-aware geometry correction (minimum width enforcement)
- Laplacian smoothing for boundary regularization
- Geometry quality metrics (width, curvature, violations)
- Rasterization-based analysis for defect detection
- Synthetic test cases for validation and benchmarking

## Motivation

In semiconductor and advanced manufacturing, layout patterns must satisfy strict geometric and process constraints such as minimum feature size, smooth transitions, and structural integrity.

This project explores how algorithmic approaches can be used to analyze, regularize, and optimize geometric layouts under such constraints. It serves as a simplified prototype of constraint-driven layout optimization workflows commonly found in EDA and manufacturing systems.

## Project Structure

```
layout-optimization/
│
├── src/
│   ├── main.py
│   ├── io.py
│   ├── geometry/
│   │   ├── smoothing.py
│   │   ├── correction.py
│   │   ├── metrics.py
│   │
│   ├── cv/
│   │   ├── rasterize.py
│   │   ├── analysis.py
│   │
│   ├── optimization/
│   │   ├── constraint_solver.py
│   │   ├── cost_function.py
│   │
│   ├── visualization.py
│   └── utils.py
│
├── tests/
│   ├── test_cases.py
│
├── examples/
│   ├── simple_case.json
│   ├── narrow_region.json
│
├── README.md
├── requirements.txt
```


