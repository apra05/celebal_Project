# Contributing to Satellite Land-Use Classifier & Temporal Change Detector

First off, thank you for considering contributing to this project! It's people like you that make open-source projects thrive.

## How to Contribute

### 1. Fork and Clone
Fork this repository to your own GitHub account and then clone it to your local machine:
```bash
git clone https://github.com/YOUR_USERNAME/celebal_Project.git
cd celebal_Project
```

### 2. Set Up Your Environment
We recommend using a Python virtual environment to manage dependencies.
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 3. Make Your Changes
- Create a new branch for your feature, enhancement, or bugfix: 
  ```bash
  git checkout -b feature/your-feature-name
  ```
- Make your code changes.
- Ensure your code follows standard Python PEP-8 formatting.

### 4. Run the Tests
We use `pytest` for unit testing. Before submitting a PR, please make sure all existing and new tests pass locally:
```bash
pytest tests/ -v
```

### 5. Submit a Pull Request
- Push your branch to your fork: 
  ```bash
  git push origin feature/your-feature-name
  ```
- Open a Pull Request against the `main` branch of this repository.
- Provide a clear and detailed description of the problem you are solving and how your code addresses it.

## Bug Reports and Feature Requests
If you spot a bug or have an idea for a new feature, feel free to open an Issue on GitHub. Please include as much detail as possible (e.g., steps to reproduce, expected vs actual behavior, and system environment).

## Code of Conduct
Please note that this project expects all contributors to maintain a welcoming and professional environment. By participating in this project, you agree to treat everyone with respect and abide by standard open-source community guidelines.
