# Flip App – Price Compare & ROI

Deploy on **Streamlit Community Cloud**:

1. Create a public GitHub repo and add two files:
   - `flip_app.py` (this app)
   - `requirements.txt` (see below)
2. Go to https://share.streamlit.io/new → *Deploy a public app from GitHub*.
3. Select your repo and set **Main file path** to `flip_app.py` → Deploy.

## requirements.txt
```
streamlit>=1.36
requests>=2.31
beautifulsoup4>=4.12
lxml>=4.9
pandas>=2.0
numpy>=1.24
```

## Local CLI (optional)
```
python flip_app.py --query "iPhone 12 64GB" --listing-price 180 --shipping 5 --fees-pct 12 --extra-costs 10
python flip_app.py --run-tests
```
