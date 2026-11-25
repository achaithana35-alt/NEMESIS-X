# Contributing to Real-Time Video Anomaly Detection

Thank you for your interest in contributing. This project welcomes contributions of all kinds.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/Real-Time-Video-Anomaly-Detection.git`
3. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a feature branch: `git checkout -b feature/your-feature`

## Development Guidelines

### Code Style
- Follow PEP 8 conventions
- Use type hints for all function signatures
- Write docstrings for all public modules, classes, and functions
- Keep functions focused and under 50 lines where possible

### Testing
- Write unit tests for all new functionality
- Run tests before submitting: `python -m pytest tests/ -v`
- Ensure all existing tests pass

### Pull Requests
- Provide a clear description of the changes
- Reference any related issues
- Include test results and/or screenshots for visual changes

## Areas for Contribution

- **Model improvements**: Better anomaly classifiers, new architectures
- **Optimization**: TensorRT, quantization-aware training
- **Datasets**: Support for additional anomaly detection benchmarks
- **Visualization**: Enhanced dashboard features
- **Documentation**: Tutorials, examples, API docs

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
