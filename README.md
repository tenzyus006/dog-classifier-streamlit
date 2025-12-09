# API-REST-avec-FastAPI

A RESTful API for text classification/tagging built with FastAPI.  
This project demonstrates how to serve a machine learning model and label binarizer using FastAPI, with deployment instructions for Heroku.

---

## Features

- **Prediction endpoint**: Quickly get tag predictions for input text via `/predict`.
- **Pre-trained ML Pipeline**: Uses a scikit-learn pipeline and a MultiLabelBinarizer.
- **Automatic model file download**: On startup, downloads model and label binarizer from URLs.

---

## Running Locally

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd API-REST-avec-FastAPI
   ```

2. **Create a virtual environment and activate it:**
   ```sh
   python3.12 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Set environment variables for model files**  
   Obtain public URLs to your model files (see below), then:
   ```sh
   export LOGISTIC_MODEL_URL=<your-logistic-model-url>
   export MLB_URL=<your-mlb-url>
   ```

5. **Run the API:**
   ```sh
   uvicorn api:app --reload
   ```

---

## Deploying to Heroku

1. **Set config vars for model URLs in Heroku dashboard**  
   Go to your app → Settings → Reveal Config Vars, and set:
   - `LOGISTIC_MODEL_URL`
   - `MLB_URL`

2. **Deploy:**
   ```sh
   git push heroku main
   ```

---

## Model File Setup

- `logistic_model_tfidf.pkl`: Your trained scikit-learn pipeline.
- `mlb.pkl`: Your pickled MultiLabelBinarizer.
- Upload these files to a public storage service (S3, Hugging Face, etc.), and use the direct links for your environment variables.

---

## API Usage

- **POST /predict**
  - Request body: `{ "text": "Your sample text here" }`
  - Response: `{ "tags": ["tag1", "tag2", ...] }`

---

## Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{"text": "Some text to classify"}'
```

---

## License

MIT License

---

## Author

Custom profile by [Your Name]