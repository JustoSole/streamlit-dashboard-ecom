# Streamlit BigQuery Deployment Guide

## Overview
This guide will help you deploy your Customer Analytics Dashboard to Streamlit Cloud with BigQuery integration.

## Prerequisites
- Google Cloud Project with BigQuery enabled
- Service account with BigQuery permissions
- Streamlit Cloud account

## Local Development Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Local Secrets** (Optional for local testing)
   - Copy `.streamlit/secrets.toml.template` to `.streamlit/secrets.toml`
   - The template already contains your service account details
   - This file is already in `.gitignore` so it won't be committed

3. **Run Locally**
   ```bash
   streamlit run dashboard_final.py
   ```

## Deployment to Streamlit Cloud

### Step 1: Push Code to GitHub
Make sure your code is pushed to a GitHub repository with:
- All Python files
- `requirements.txt`
- `.gitignore` (already configured)
- **DO NOT** include `service_account.json` or `.streamlit/secrets.toml`

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repository
3. Set the main file path: `dashboard_final.py`
4. Click "Deploy"

### Step 3: Configure Secrets
1. In your Streamlit Cloud app dashboard, click "Settings"
2. Go to the "Secrets" tab
3. Add the following configuration:

```toml
[gcp_service_account]
type = "service_account"
project_id = "racket-central-gcp"
private_key_id = "42d29c8dfca2b2f4c549f37e31a46eab356051eb"
private_key = "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC9quYzprwSVIcd\nD1GUaaPcYhseSCqXQ++/tw1MFudM2AuxF8JtGrhV+4oDBPA5yej+0bo0Phwzr9qV\nmEaFrscU9REc9Cz2Gg1nr/D8GAZisfdKQ49++xkjYJJv5D4H4dt41JoCinwRcsWL\nu+EReorSFjxGOUuzNvB+rGofyXwzwnzN5MxDvrR/2dtV+O+hBx5joWzoMU2wKhUA\nkceUtcSB59ew4BiL7B+etBVdvpdKXmx+5hlfV7AvlBhmaS1OoI48kic+SmQtP8kq\ns6aitti+Vb86eRIlfkCf08HK0yiSD/wJUNqTVNhr33AKBCBr+hr/gtJF6JfYXe9m\nwJSo61hNAgMBAAECggEADOV5bfVTisAySGC5xdMhtVhQYBCSdiLu7YS8ro9MyUCp\nbtsJjnycyxqQPvX6KvEWhXkXkE98Kxbf0ICATxJcGV8z0NmuKEpg3fYsdx+1E3Hm\nuXx63ddPVW9OwF/vHdRsJWejVIfmZBNUk6nskFEvmOsHzH3p2Uo43mF1J9XTOvug\n+Z2qXL0k8GSuvdcVexYrSW5SgHVYzA4IKN+bBhLtI+j90aMBaJV9IFh9aArsOkgz\n0U1GGMfyZUCqp1uVvA/ZDbjnfa7gynZ6n4gei/oDBkpL+/E0SMlP3Cz96didAVYZ\nbTqmhWgtgxL5+xh6JPmQWiRF30NfD08TpAZDgVnuSwKBgQDeLdqq9hKYPSFQxpbv\nAKdhHLgLcYUsgKkY97f3Mm3tfdNtIAnMrpF3tICmdLWuH4MW9b+sv+RkFs/6KIFR\nNvN6CPmRYy8cQ+rMlHYE+ZXdyI+VFqvr8q9ewqoJaqxw6kxOlcqWoBs3NLqQhfqW\niU5g192fGBS20UruGokXr2n22wKBgQDaihiDOtahljmNn2lRebPZ6MkeIJ0aDGcv\n+TM1sY2FraQU7nMtDEkt86AXljQzrQeLaZ/xeCS+1q/ruVu80o+SjjHYfbjVuPuw\nCUIArMhzv03VJALBOUZvCOFUe2PNodw9pbEUMpHdnhoZLins/xOh94GBJOPXOOmb\nVJcvof7x9wKBgHpexxFidttiz/atanQ45/eU2clzvOXF91zJE4oTPHiR1OFFxB/4\nBiboQ/NqVKaStKDwuaFsD18RMXuW06LnoTKVvt9UwZ3PyoLjQh17Wg/NZ0e2NPq/\nr9eBYCXPmyqV4XFnDy3nARZm9FqlcF95QLIWMvptSPtoStzZwKhK9RIBAoGBAI0o\nI7sn1xiaKuSSMfnBbWz0EmvWTwNTPZdcFDq7S5kr1k760gQn6mC2+xIhH+i8+6GO\nARR8MOffTdQpbtrg+oGEPSgD1M6fZFqJMEu1TuiMiZ6BWxIph5gSmVDzPjzFLrfW\n/TD5lQQbqenXypbdD3ZPoyOii1Qp26JMGjdXIJBzAoGBAM2aNBaWMv2GUhci3a/u\n1DcI81jJ9Z5N6UKHU0SdN3AcSzaTTAU5nEP6GHcFBKJyqrL1Mv6LJErlsPARht6z\nQiarQAF2JYymvNnRTrEaDNCBrUipL8E9IDrHz/tjT21jnjjq0yoK5JmryX+qPxjx\n8NebiOx3cXtAf4c1DcyTv9lM\n-----END PRIVATE KEY-----\n"
client_email = "bigquery-streamlit@racket-central-gcp.iam.gserviceaccount.com"
client_id = "115054804142351122236"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/bigquery-streamlit%40racket-central-gcp.iam.gserviceaccount.com"
universe_domain = "googleapis.com"
```

4. Click "Save"
5. Your app will automatically redeploy with the new secrets

## How the Authentication Works

The updated `utils.py` file now:

1. **First tries Streamlit secrets** (for deployed environment)
   - Checks if `st.secrets` contains `gcp_service_account`
   - Uses these credentials if available
   - Shows success message: "✅ Connected to BigQuery using Streamlit secrets"

2. **Falls back to local service account** (for development)
   - Looks for `service_account.json` file
   - Uses local file if secrets are not available
   - Shows success message: "✅ Connected to BigQuery using local service account"

3. **Provides clear error messages** if neither option works

## Troubleshooting

### Common Issues:

1. **"Service account file not found"**
   - For local development: Ensure `service_account.json` exists
   - For deployment: Add secrets to Streamlit Cloud settings

2. **"Failed to connect using Streamlit secrets"**
   - Check that all fields in secrets are correctly formatted
   - Ensure private key includes `\n` characters properly
   - Verify project_id matches your GCP project

3. **Permission errors**
   - Ensure service account has BigQuery Data Viewer and Job User roles
   - Check that the dataset `racket_central_dev` exists and is accessible

### Testing the Connection:
The app will show a green success message when BigQuery connection is established successfully.

## Security Notes
- Never commit `service_account.json` to version control
- Never commit `.streamlit/secrets.toml` to version control  
- Both files are already in `.gitignore`
- Use Streamlit secrets for all deployed environments 