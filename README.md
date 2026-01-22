# Mobile Price Range Predictor

This project is a machine learning application that predicts the price range of mobile phones based on their specifications. It was developed as a final assignment for a Machine Learning course, implementing a complete ML pipeline from data preprocessing to deployment.

## Features

- **Data Analysis & Preprocessing**: Comprehensive data cleaning, feature engineering, and preprocessing steps
- **Machine Learning Pipeline**: Integrated preprocessing and model training pipeline
- **Model Training**: Random Forest classifier with hyperparameter tuning
- **Web Interface**: User-friendly Gradio web app for predictions
- **Deployment Ready**: Deployed on Hugging Face Spaces

## Dataset

The project uses the Mobile Price Classification dataset, which contains specifications of mobile phones and their price ranges (0-3, where 0 is low cost and 3 is very high cost).

- `train.csv`: Training dataset
- `test.csv`: Test dataset

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mobile-price-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Web App Locally

```bash
python app.py
```

The Gradio interface will launch in your browser, allowing you to input mobile specifications and get price range predictions.

### Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: Battery power, Bluetooth, clock speed, dual SIM, camera specs, memory, dimensions, and more
- **Engineered Features**: Pixel area, screen area, total camera MP, advanced features count

### Model Performance

- **Test Accuracy**: 90.5%
- **Cross-Validation Score**: Average accuracy with standard deviation (from notebook)
- **Key Features**: RAM, battery power, pixel resolution are most important predictors

## Project Structure

- `app.py`: Main Gradio web application
- `mobile_price_classification_assignment.ipynb`: Jupyter notebook with complete ML pipeline
- `mobile_price_model.pkl`: Trained model file
- `requirements.txt`: Python dependencies
- `train.csv` & `test.csv`: Dataset files

## Deployment

The application is deployed on Hugging Face Spaces. Access it at: https://huggingface.co/spaces/Hasan0423/Mobile_Price_Prediction_App

## Submission Links

- **GitHub Repository**: https://github.com/hasan042021/Mobile-Price-Predictor-Model
- **Google Colab Notebook**: https://colab.research.google.com/drive/1zogwSs17DIoZriiTCjRxWbfqn81gla1y?usp=sharing
- **Hugging Face App**: https://huggingface.co/spaces/Hasan0423/Mobile_Price_Prediction_App

## Technologies Used

- Python
- Scikit-learn
- Pandas & NumPy
- Gradio
- Jupyter Notebook

## License

This project is for educational purposes as part of a machine learning course assignment.
