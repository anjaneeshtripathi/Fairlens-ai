# 🚀 FairLens AI — Complete Deployment Guide

---

## PART 1: GET YOUR GEMINI API KEY (Free, 2 minutes)

### Step-by-Step

1. Open your browser and go to:
   **https://aistudio.google.com/app/apikey**

2. Sign in with your **Google account** (Gmail works).

3. Click **"Create API Key"** → Select your project (or create a new one).

4. Copy the key — it looks like: `AIzaSy...` (39 characters).

5. **Keep it safe** — treat it like a password. Never commit it to GitHub.

### Free Tier Limits
| Limit | Value |
|-------|-------|
| Requests per minute | 15 |
| Requests per day | 1,500 |
| Tokens per minute | 1,000,000 |
| Cost | **FREE** |

This is more than enough for a hackathon demo.

---

## PART 2: RUN LOCALLY

```bash
# 1. Unzip the project
unzip fairlens_ai_v3.zip
cd fairlens_ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Set Gemini key as environment variable
#    This pre-fills the key in the app so you don't type it every time
export GEMINI_API_KEY="AIzaSy..."    # Linux/Mac
set GEMINI_API_KEY=AIzaSy...         # Windows CMD
$env:GEMINI_API_KEY="AIzaSy..."      # Windows PowerShell

# 5. Launch the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## PART 3: DEPLOY TO STREAMLIT CLOUD (Easiest — 5 minutes, FREE)

This is the recommended method for the hackathon submission.

### Step 1: Push to GitHub

```bash
# If you don't have git initialized:
git init
git add .
git commit -m "FairLens AI v3 with Gemini"

# Create a new repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/fairlens-ai.git
git push -u origin main
```

### Step 2: Connect Streamlit Cloud

1. Go to **https://share.streamlit.io**
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository: `YOUR_USERNAME/fairlens-ai`
5. Set **Main file path**: `fairlens_ai/app.py`
6. Click **"Deploy!"**

### Step 3: Add Your Gemini API Key as a Secret

1. In your deployed app dashboard, click **"⋮" → "Settings" → "Secrets"**
2. Paste this:
   ```toml
   GEMINI_API_KEY = "AIzaSy..."
   ```
3. Click **"Save"** — app restarts automatically.

✅ **Your app is now live at**: `https://YOUR_USERNAME-fairlens-ai-app-XXXX.streamlit.app`

---

## PART 4: DEPLOY TO GOOGLE CLOUD RUN (Production-grade)

This satisfies the "cloud deployment" requirement with a Google Cloud service.

### Prerequisites
- Google Cloud account (free tier: $300 credit for new accounts)
- Google Cloud CLI (`gcloud`) installed

### Step 1: Install gcloud CLI

```bash
# Mac (with Homebrew)
brew install google-cloud-sdk

# Linux
curl https://sdk.cloud.google.com | bash

# Windows — download installer from:
# https://cloud.google.com/sdk/docs/install
```

### Step 2: Authenticate and Set Up Project

```bash
# Login
gcloud auth login

# Create a new project (or use existing)
gcloud projects create fairlens-ai-YOURNAME --name="FairLens AI"

# Set the project
gcloud config set project fairlens-ai-YOURNAME

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

### Step 3: Build and Deploy

```bash
cd fairlens_ai

# Build the Docker image on Google Cloud Build (no local Docker needed!)
gcloud builds submit --tag gcr.io/fairlens-ai-YOURNAME/fairlens-ai .

# Deploy to Cloud Run
gcloud run deploy fairlens-ai \
  --image gcr.io/fairlens-ai-YOURNAME/fairlens-ai \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --set-env-vars GEMINI_API_KEY="AIzaSy..."
```

### Step 4: Get Your URL

```bash
gcloud run services describe fairlens-ai \
  --region us-central1 \
  --format 'value(status.url)'
```

✅ **Your app is live at**: `https://fairlens-ai-XXXX-uc.a.run.app`

### Estimated Cost
| Service | Monthly Cost |
|---------|-------------|
| Cloud Run (low traffic) | ~$0–5 |
| Artifact Registry | ~$0.10 |
| Cloud Build | Free (120 min/day) |
| **Total** | **< $5/month** |

---

## PART 5: AUTOMATED CI/CD WITH GITHUB ACTIONS

The `.github/workflows/deploy.yml` file auto-deploys every time you push to `main`.

### Setup

1. Create a Service Account in GCP:
```bash
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions Deployer"

gcloud projects add-iam-policy-binding fairlens-ai-YOURNAME \
  --member="serviceAccount:github-actions@fairlens-ai-YOURNAME.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding fairlens-ai-YOURNAME \
  --member="serviceAccount:github-actions@fairlens-ai-YOURNAME.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

# Download the key
gcloud iam service-accounts keys create key.json \
  --iam-account=github-actions@fairlens-ai-YOURNAME.iam.gserviceaccount.com
```

2. Add these as GitHub repository secrets (Settings → Secrets → Actions):
   - `GCP_PROJECT_ID` = `fairlens-ai-YOURNAME`
   - `GCP_SA_KEY` = contents of `key.json` (paste the entire JSON)
   - `GEMINI_API_KEY` = `AIzaSy...`

3. Push to main — GitHub Actions will build and deploy automatically!

---

## PART 6: ENVIRONMENT VARIABLE REFERENCE

| Variable | Where to set | Description |
|----------|-------------|-------------|
| `GEMINI_API_KEY` | `.env`, Cloud Run, Streamlit Secrets | Your Google AI Studio key |

The app also accepts the key via the UI text input, so it works even without the env var.

---

## PART 7: PROJECT STRUCTURE

```
fairlens_ai/
├── app.py                    ← Main Streamlit app (entry point)
├── requirements.txt          ← Python dependencies
├── Dockerfile                ← Container for Cloud Run
├── .streamlit/
│   ├── config.toml           ← Dark theme + server config
│   └── secrets.toml.template ← Template for secrets (don't commit!)
├── utils/
│   ├── __init__.py
│   ├── data_loader.py        ← Load Adult Income Dataset
│   ├── preprocessor.py       ← Clean, encode, split, scale
│   ├── model_trainer.py      ← Train Logistic Regression
│   ├── bias_detector.py      ← DPD, EOD, Bias Score 0–100
│   ├── gemini_advisor.py     ← Google Gemini AI integration ← NEW
│   ├── whatif_simulator.py   ← Flip attribute → re-predict
│   ├── mitigator.py          ← Reweighting & remove-sensitive
│   ├── report_generator.py   ← Exportable .txt audit report
│   └── visualizer.py         ← All matplotlib charts
├── data/                     ← Auto-downloaded dataset cache
├── models/                   ← Saved model artifacts
└── reports/                  ← Generated fairness reports
```

---

## PART 8: QUICK CHECKLIST FOR SUBMISSION

- [ ] App runs locally: `streamlit run app.py`
- [ ] Gemini API key works (test with a quick question in the app)
- [ ] App deployed to Streamlit Cloud OR Google Cloud Run
- [ ] GitHub repository is **public**
- [ ] Demo video recorded (3 minutes max — show all 6 key screens)
- [ ] All 4 links ready: GitHub, Demo Video, MVP Link, Prototype Link

---

*FairLens AI — Built for Hack2Skill Solution Challenge 2026*
