# flip_app.py
# --------------------------------------------------------------
# Mini aplikacija za primerjavo cen + ROI z **Streamlit (če je na voljo)** ali **CLI** načinom.
# Popravljeno, da se ob manjkajočih argumentih ne konča z `SystemExit: 2` (argparse),
# ampak ponudi interaktivni vnos ali prijazno sporočilo z navodili.
#
# ➤ UI način (Streamlit Cloud/Spaces/local):
#     pip install -r requirements.txt
#     streamlit run flip_app.py
# ➤ CLI način (lokalno/Colab terminal):
#     python flip_app.py --query "iPhone 12 64GB" --listing-price 180 --shipping 5 --fees-pct 12 --extra-costs 10
#     python flip_app.py --url "https://..." --listing-price 180 ...
# ➤ Brez argumentov: CLI ponudi **interaktivni vnos** za query.
#
# POMEMBNO: Scraping lahko krši pogoje uporabe posameznih strani. Uporabi odgovorno in za osebno rabo.

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from statistics import mean
from typing import List, Dict, Optional, Tuple

import requests
from bs4 import BeautifulSoup

# Poskusi uvoziti dodatne knjižnice; delovanje brez njih naj ostane možno
try:
    import pandas as pd  # type: ignore
    PANDAS_AVAILABLE = True
except Exception:
    pd = None  # type: ignore
    PANDAS_AVAILABLE = False

try:
    import numpy as np  # type: ignore
    NUMPY_AVAILABLE = True
except Exception:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

try:
    import streamlit as st  # type: ignore
    STREAMLIT_AVAILABLE = True
except Exception:
    st = None  # type: ignore
    STREAMLIT_AVAILABLE = False

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# --------------------------------------------------------------
# Helperji
# --------------------------------------------------------------

def clean_price(text: str) -> Optional[float]:
    """Pretvori različne oblike cen v float. Vrne None, če ne uspe."""
    if not text:
        return None
    t = re.sub(r"[\n\r\t]", " ", text).replace("\xa0", " ")
    m = re.findall(r"[0-9][0-9\.,\s'–-]*", t)
    if not m:
        return None
    cand = m[0].strip().replace(" ", "").replace("–", "").replace("-", "")
    # švicarski apostrof tisočice
    cand = cand.replace("'", "")
    if "," in cand and "." in cand:
        if cand.rfind(",") > cand.rfind("."):
            cand = cand.replace(".", "").replace(",", ".")
        else:
            cand = cand.replace(",", "")
    else:
        if cand.count(",") == 1 and cand.count(".") == 0:
            cand = cand.replace(",", ".")
        else:
            cand = cand.replace(",", "")
    try:
        return float(cand)
    except Exception:
        return None


def get_title_from_url(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "lxml")
        title = soup.title.text.strip() if soup.title else None
        if title:
            title = re.sub(r"\s*[|\-–]\s*.*$", "", title)
        return title
    except Exception:
        return None


@dataclass
class DealCosts:
    listing_price: float = 0.0
    shipping: float = 0.0
    fees_pct: float = 12.0
    extra_costs: float = 0.0

    def estimate_fees(self) -> float:
        return self.listing_price * self.fees_pct / 100.0

    def total_cost(self) -> float:
        return self.listing_price + self.shipping + self.estimate_fees() + self.extra_costs


# --------------------------------------------------------------
# Viri podatkov
# --------------------------------------------------------------

def search_ebay(query: str, sold: bool = False, limit: int = 10) -> List[Dict]:
    """Vrne sezname compov: title, price, url, source."""
    q = requests.utils.quote(query)
    if sold:
        url = f"https://www.ebay.com/sch/i.html?_nkw={q}&LH_Sold=1&LH_Complete=1"
    else:
        url = f"https://www.ebay.com/sch/i.html?_nkw={q}"
    out: List[Dict] = []
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return out
        soup = BeautifulSoup(r.text, "lxml")
        for li in soup.select("li.s-item"):
            title_el = li.select_one("h3.s-item__title")
            price_el = li.select_one("span.s-item__price")
            link_el = li.select_one("a.s-item__link")
            if not title_el or not price_el or not link_el:
                continue
            title = title_el.get_text(strip=True)
            price = clean_price(price_el.get_text())
            url_item = link_el.get("href")
            if price is None:
                continue
            out.append({
                "title": title,
                "price": price,
                "url": url_item,
                "source": f"eBay {'Sold' if sold else 'Active'}",
            })
            if len(out) >= limit:
                break
        return out
    except Exception:
        return out


def search_bolha(query: str, limit: int = 10) -> List[Dict]:
    q = requests.utils.quote(query)
    url = f"https://www.bolha.com/?ctl=search_ads&keywords={q}"
    out: List[Dict] = []
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return out
        soup = BeautifulSoup(r.text, "lxml")
        cards = soup.select("article.EntityList-item, li.EntityList-item")
        for card in cards:
            title_el = card.select_one("a.link, a.link.title") or card.select_one("a")
            price_el = card.select_one("strong.price") or card.select_one("span.price")
            if not title_el or not price_el:
                continue
            title = title_el.get_text(strip=True)
            href = title_el.get("href", "")
            if href and href.startswith("/"):
                href = "https://www.bolha.com" + href
            price = clean_price(price_el.get_text())
            if price is None:
                continue
            out.append({
                "title": title,
                "price": price,
                "url": href,
                "source": "Bolha (active)",
            })
            if len(out) >= limit:
                break
        return out
    except Exception:
        return out


# --------------------------------------------------------------
# Izračuni
# --------------------------------------------------------------

def market_average(comps: List[Dict]) -> Tuple[Optional[float], str]:
    """Povprečje iz eBay Sold, če obstaja; sicer iz vseh compov."""
    if not comps:
        return None, "Ni compov"
    sold_prices = [c["price"] for c in comps if isinstance(c.get("price"), (int, float)) and "Sold" in str(c.get("source", ""))]
    if sold_prices:
        return float(mean(sold_prices)), "eBay Sold"
    prices = [c["price"] for c in comps if isinstance(c.get("price"), (int, float))]
    if not prices:
        return None, "Ni veljavnih cen"
    return float(mean(prices)), "Vse najdbe"


def roi_summary(costs: DealCosts, market_avg: Optional[float]) -> Dict[str, float]:
    est_fees = costs.estimate_fees()
    total = costs.total_cost()
    if market_avg is None:
        return {"market_avg": float("nan"), "est_fees": est_fees, "total_cost": total, "profit": float("nan"), "roi": float("nan")}
    profit = market_avg - total
    roi = (profit / total * 100.0) if total > 0 else float("nan")
    return {
        "market_avg": market_avg,
        "est_fees": est_fees,
        "total_cost": total,
        "profit": profit,
        "roi": roi,
    }


# --------------------------------------------------------------
# STREAMLIT UI (le če je nameščen)
# --------------------------------------------------------------

def run_streamlit_app() -> None:
    st.set_page_config(page_title="Flip Compare & ROI", page_icon="💸", layout="wide")
    st.title("💸 Mini Flip App — primerjava cen + ROI")
    st.caption("Vnesi URL ali ime izdelka. App poišče primerljive cene (eBay Sold, eBay Active, Bolha) in izračuna ROI.")

    with st.sidebar:
        st.subheader("Nastavitve stroškov")
        listing_price = st.number_input("Cena oglasa (EUR)", min_value=0.0, step=1.0, value=0.0)
        shipping = st.number_input("Poštnina (EUR)", min_value=0.0, step=1.0, value=5.0)
        fees_pct = st.number_input("Provizije platforme (%)", min_value=0.0, step=0.5, value=12.0)
        extra_costs = st.number_input("Dodatni stroški (EUR)", min_value=0.0, step=1.0, value=10.0)
        min_profit = st.number_input("Ciljni dobiček (EUR)", min_value=0.0, step=5.0, value=40.0)
        min_roi = st.number_input("Ciljni ROI (%)", min_value=0.0, step=1.0, value=20.0)

    tab1, tab2 = st.tabs(["🔗 URL izdelka", "🔎 Iskanje po imenu"])
    query = None

    with tab1:
        url_input = st.text_input("Prilepi URL oglasa (FB Marketplace, Bolha, trgovina…)")
        if st.button("Analiziraj URL", type="primary"):
            title = get_title_from_url(url_input) if url_input else None
            if not title:
                st.warning("Naslova ni bilo mogoče prebrati. Vpiši ime izdelka v drugi zavihek.")
            else:
                st.write(f"📝 Najden naslov: **{title}** (po potrebi ga skrajšaj)")
                query = title

    with tab2:
        name_input = st.text_input("Vpiši ime/model (npr. 'iPhone 12 64GB')")
        if st.button("Poišči primerjave"):
            query = name_input

    if query:
        st.markdown(f"### Rezultati za: **{query}**")
        with st.spinner("Iščem primerljive cene…"):
            comps: List[Dict] = []
            comps += search_ebay(query, sold=True, limit=10)
            time.sleep(0.5)
            comps += search_ebay(query, sold=False, limit=10)
            time.sleep(0.5)
            comps += search_bolha(query, limit=10)

        if not comps:
            st.error("Ni najdenih primerjav (poskusi drugačen izraz ali ročno preverjanje).")
        else:
            avg, src = market_average(comps)
            costs = DealCosts(listing_price, shipping, fees_pct, extra_costs)
            summary = roi_summary(costs, avg)

            c1, c2 = st.columns([2, 1])
            with c1:
                st.subheader("Primerjave (comps)")
                if PANDAS_AVAILABLE:
                    df = pd.DataFrame(comps)
                    st.dataframe(df[["source", "title", "price", "url"]])
                else:
                    st.write(comps)

            with c2:
                st.subheader("Izračun")
                if summary["market_avg"] == summary["market_avg"]:
                    st.metric("Povprečna tržna cena", f"€ {summary['market_avg']:.2f}", help=f"Vir: {src}")
                else:
                    st.metric("Povprečna tržna cena", "N/A", help=f"Vir: {src}")
                st.metric("Skupni strošek", f"€ {summary['total_cost']:.2f}")
                if summary["profit"] == summary["profit"]:
                    st.metric("Potencialni dobiček", f"€ {summary['profit']:.2f}")
                    st.metric("ROI", f"{summary['roi']:.1f}%")
                else:
                    st.metric("Potencialni dobiček", "N/A")
                    st.metric("ROI", "N/A")

            decision = (
                "GO ✅" if (
                    summary["profit"] == summary["profit"]
                    and summary["profit"] >= min_profit
                    and summary["roi"] >= min_roi
                ) else "PASS ❌"
            )
            st.success(f"Odločitev: **{decision}** · Cilji: dobiček ≥ €{min_profit:.0f}, ROI ≥ {min_roi:.0f}%")

        st.caption("⚠️ Dinamična spletna mesta se spreminjajo. Če se parsing pokvari, posodobi selektorje v kodi.")


# --------------------------------------------------------------
# CLI način
# --------------------------------------------------------------

def _print_cli_examples() -> None:
    print("\nUporaba (primeri):")
    print('  python flip_app.py --query "iPhone 12 64GB" --listing-price 180 --shipping 5 --fees-pct 12 --extra-costs 10')
    print('  python flip_app.py --url "https://primer.si/oglas" --listing-price 180 --shipping 5 --fees-pct 12 --extra-costs 10')
    print("  python flip_app.py --run-tests")


def _interactive_prompt() -> Optional[str]:
    """Vedno poskusi interaktivni vnos za query, če argumenti manjkajo.
    Vrne niz ali None, če vnos ni mogoč (npr. EOFError)."""
    try:
        print("[?] Manjkajo argumenti --query/--url. Vnesi ime/model (pusti prazno za preklic):")
        q = input("> ").strip()
        return q or None
    except (EOFError, KeyboardInterrupt):
        return None


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Price compare + ROI (CLI)", add_help=True)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--query", type=str, help="Ime/model izdelka (npr. 'iPhone 12 64GB')")
    src.add_argument("--url", type=str, help="URL oglasa (poskusi prebrati naslov strani)")

    parser.add_argument("--listing-price", type=float, default=0.0, help="Cena oglasa (EUR)")
    parser.add_argument("--shipping", type=float, default=5.0, help="Poštnina (EUR)")
    parser.add_argument("--fees-pct", type=float, default=12.0, help="Provizije (%)")
    parser.add_argument("--extra-costs", type=float, default=10.0, help="Dodatni stroški (EUR)")

    parser.add_argument("--min-profit", type=float, default=40.0, help="Ciljni dobiček (EUR)")
    parser.add_argument("--min-roi", type=float, default=20.0, help="Ciljni ROI (%)")

    parser.add_argument("--limit", type=int, default=10, help="Max zadetkov iz vsakega vira")
    parser.add_argument("--save-csv", type=str, default=None, help="Shrani comps v CSV datoteko")
    parser.add_argument("--run-tests", action="store_true", help="Zaženi vgrajene enote teste in izhod")

    args, _unknown = parser.parse_known_args()

    if args.run_tests:
        run_tests()
        return

    query = args.query

    if not query and args.url:
        query = get_title_from_url(args.url)
        if not query:
            print("[!] Naslova iz URL ni bilo mogoče prebrati.")
            query = _interactive_prompt()
            if not query:
                _print_cli_examples()
                return

    if not query and not args.url:
        query = _interactive_prompt()
        if not query:
            print("[!] Ni bilo vnosa.")
            _print_cli_examples()
            return

    print(f"[i] Iskanje compov za: {query}")
    comps: List[Dict] = []
    comps += search_ebay(query, sold=True, limit=args.limit)
    time.sleep(0.3)
    comps += search_ebay(query, sold=False, limit=args.limit)
    time.sleep(0.3)
    comps += search_bolha(query, limit=args.limit)

    if not comps:
        print("[!] Ni najdenih primerjav.")
        return

    avg, src = market_average(comps)
    costs = DealCosts(args.listing_price, args.shipping, args.fees_pct, args.extra_costs)
    summary = roi_summary(costs, avg)

    print("\n=== Povzetek ===")
    print(f"Vir povprečja: {src}")
    if summary["market_avg"] == summary["market_avg"]:
        print(f"Povprečna tržna cena: € {summary['market_avg']:.2f}")
    else:
        print("Povprečna tržna cena: N/A")
    print(f"Skupni strošek:       € {summary['total_cost']:.2f}")
    if summary["profit"] == summary["profit"]:
        print(f"Potencialni dobiček:  € {summary['profit']:.2f}")
        print(f"ROI:                  {summary['roi']:.1f}%")
    else:
        print("Potencialni dobiček:  N/A")
        print("ROI:                  N/A")

    decision = (
        "GO ✅" if (
            summary["profit"] == summary["profit"]
            and summary["profit"] >= args.min_profit
            and summary["roi"] >= args.min_roi
        ) else "PASS ❌"
    )
    print(f"Odločitev: {decision} · Cilji: dobiček ≥ €{args.min_profit:.0f}, ROI ≥ {args.min_roi:.0f}%\n")

    if args.save_csv:
        if not PANDAS_AVAILABLE:
            print("[!] pandas ni na voljo – CSV ne bo shranjen. Namesti z: pip install pandas")
        else:
            df = pd.DataFrame(comps)
            df.to_csv(args.save_csv, index=False)
            print(f"[✓] Shranjeno: {args.save_csv}")


# --------------------------------------------------------------
# TESTI (brez mreže) – zaženemo z: python flip_app.py --run-tests
# --------------------------------------------------------------

def run_tests() -> None:
    print("[tests] Začenjam teste…")

    # clean_price – osnovni primeri
    cases = {
        "€ 1.234,56": 1234.56,
        "$1,299.99": 1299.99,
        "199 €": 199.0,
        "1 299,00 kn": 1299.0,
        "abc": None,
        "": None,
    }
    for raw, exp in cases.items():
        got = clean_price(raw)
        assert (got == exp) or (got is None and exp is None), f"clean_price('{raw}') -> {got}, pričakovano {exp}"

    # clean_price – dodatni robni primeri
    extra_cases = {
        "CHF 1'249.00": 1249.0,
        "£79.99": 79.99,
        "1.299,00 €": 1299.0,
        "EUR 49,–": 49.0,
        "2.499,– €": 2499.0,
        "€1.299": 1299.0,
        "---": None,
    }
    for raw, exp in extra_cases.items():
        got = clean_price(raw)
        assert (got == exp) or (got is None and exp is None), f"clean_price('{raw}') -> {got}, pričakovano {exp}"

    # market_average – preferira Sold
    comps_all = [
        {"price": 200, "source": "eBay Active"},
        {"price": 210, "source": "eBay Active"},
        {"price": 190, "source": "eBay Sold"},
    ]
    avg, src = market_average(comps_all)
    assert round(avg, 2) == 190.00 and src == "eBay Sold", f"market_average -> {avg}, {src}"

    # market_average – brez Sold, vzemi povprečje vseh
    comps_active = [
        {"price": 100, "source": "eBay Active"},
        {"price": 150, "source": "eBay Active"},
    ]
    avg2, src2 = market_average(comps_active)
    assert round(avg2, 2) == 125.00 and src2 == "Vse najdbe", f"market_average (active only) -> {avg2}, {src2}"

    # market_average – prazno
    avg3, src3 = market_average([])
    assert avg3 is None and src3 == "Ni compov", f"market_average (empty) -> {avg3}, {src3}"

    # roi_summary – standardni primer
    costs = DealCosts(listing_price=150, shipping=5, fees_pct=10, extra_costs=5)
    summary = roi_summary(costs, market_avg=220)
    # fees = 15, total = 175, profit = 45, roi ≈ 25.714%
    assert round(summary["est_fees"], 2) == 15.00
    assert round(summary["total_cost"], 2) == 175.00
    assert round(summary["profit"], 2) == 45.00
    assert round(summary["roi"], 2) == 25.71

    # roi_summary – total_cost == 0 (ROI mora biti NaN)
    zero_costs = DealCosts(listing_price=0, shipping=0, fees_pct=0, extra_costs=0)
    summary2 = roi_summary(zero_costs, market_avg=100)
    assert summary2["total_cost"] == 0
    assert summary2["roi"] != summary2["roi"], "ROI naj bo NaN, ko je total_cost 0"

    print("[tests] Vsi testi so OK ✅")


# --------------------------------------------------------------
# Vstopna točka
# --------------------------------------------------------------
if __name__ == "__main__":
    # Če teče pod Streamlitom, bo UI; sicer CLI
    if STREAMLIT_AVAILABLE and hasattr(st, "_is_running_with_streamlit"):
        run_streamlit_app()
    else:
        run_cli()
