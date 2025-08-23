# File Descriptions

- `custom_functions.py`: Contains custom utility functions used across the project.
- `data_loader.py`: Handles loading and preprocessing of data.
- `model_trainer.py`: Includes functions to train machine learning models.
- `evaluator.py`: Provides evaluation metrics and functions to assess model performance.
- `visualizer.py`: Contains functions for data and result visualization.

---

# Jay GenAI Portfolio - Custom Functions Module

## Project Overview

Welcome to the Custom Functions module of the Jay GenAI Portfolio. This module contains essential utility scripts that support the main application, facilitating data handling, model training, evaluation, and visualization.

## Features

- Modular utility functions for easy reuse
- Data loading and preprocessing capabilities
- Model training and evaluation helpers
- Visualization tools for insights and results

## Repo Structure (Custom_Functions folder)

```
Custom_Functions/
├── __init__.py
├── index/
├── app.log
├── decorators_demo.py
├── faiss_query_engine.py
├── faiss_rag_demo.py
├── file_utils.py
├── log_utils.py
├── query_engine.py
├── README.md
├── requirements.txt
├── streamlit_app.py
└── test_query_engine.py
```

## Requirements Example

```bash
streamlit
numpy
pandas
scikit-learn
matplotlib
```

## .gitignore Essentials

```
__pycache__/
*.pyc
.env
.DS_Store
```

## Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/yourusername/jay-genai-portfolio.git
cd jay-genai-portfolio/Custom_Functions
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r ../requirements.txt
```

## Running Instructions

### Streamlit App

From the root directory, run:

```bash
streamlit run app.py
```

### Tester Script

Run the tester script to validate functions:

```bash
python test_custom_functions.py
```

### RAG Demo

Execute the Retrieval-Augmented Generation demo:

```bash
python rag_demo.py
```

## How It Works

The Custom Functions module provides foundational support to the main portfolio application by:

- Loading and preprocessing datasets
- Training machine learning models with configurable parameters
- Evaluating model performance with standard metrics
- Visualizing data trends and model results for better understanding

## Common Commands

- `python <script>.py` - Run any script
- `streamlit run app.py` - Launch the Streamlit web app
- `pytest` - Run tests (if configured)

## Deployment Options

### Streamlit Cloud

- Push your code to GitHub
- Connect your repo to Streamlit Cloud for automatic deployment

### Docker

- Build a Docker image:

```bash
docker build -t jay-genai-portfolio .
```

- Run the container:

```bash
docker run -p 8501:8501 jay-genai-portfolio
```

## Pushing to GitHub

```bash
git add .
git commit -m "Add Custom Functions module"
git push origin main
```

## Troubleshooting

- Ensure all dependencies are installed
- Check Python version compatibility (recommended 3.8+)
- Review error messages for missing files or modules
- Use virtual environments to avoid conflicts

## Next Steps

- Integrate additional utility functions as needed
- Expand testing coverage for robustness
- Optimize performance of data processing functions
- Enhance documentation with usage examples and tutorials

Thank you for exploring the Custom Functions module of the Jay GenAI Portfolio!
