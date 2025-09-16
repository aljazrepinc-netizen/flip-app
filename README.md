# Flip App – Streamlit + CLI

## Deploy on Streamlit Cloud
1. Upload `flip_app.py` and `requirements.txt` to your GitHub repo.
2. Go to https://share.streamlit.io/new → Deploy from GitHub.
3. Select your repo, set **Main file path** = `flip_app.py`, then Deploy.

## Requirements
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
