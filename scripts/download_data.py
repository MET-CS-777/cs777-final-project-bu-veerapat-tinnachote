"""
download_data.py

Pulls the raw StatsBomb open-data JSON files (matches, events, lineups)
needed for the competitions listed in config.py, and saves them locally
under data/. Run this once before the Spark pipeline.

Source: https://github.com/statsbomb/open-data (free, no API key required)

Usage:
    python scripts/download_data.py

Note: this script must be run from a machine with normal internet access
to raw.githubusercontent.com. (It will not work from a network-restricted
sandbox -- that's expected; run it on your own laptop per the assignment
instructions, which explicitly allow local execution.)
"""
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_RAW_URL, COMPETITIONS, DATA_DIR, MATCHES_DIR, EVENTS_DIR, LINEUPS_DIR


def fetch_json(url, retries=3, backoff=2.0):
    for attempt in range(retries):
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if attempt < retries - 1:
            time.sleep(backoff * (attempt + 1))
    resp.raise_for_status()


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1) competitions.json (reference / documentation)
    print("Downloading competitions.json ...")
    competitions = fetch_json(f"{BASE_RAW_URL}/competitions.json")
    save_json(competitions, f"{DATA_DIR}/competitions.json")

    all_match_ids = []

    for comp_id, season_id, name in COMPETITIONS:
        print(f"\n=== {name} (competition_id={comp_id}, season_id={season_id}) ===")

        matches_url = f"{BASE_RAW_URL}/matches/{comp_id}/{season_id}.json"
        matches = fetch_json(matches_url)
        save_json(matches, f"{MATCHES_DIR}/{comp_id}/{season_id}.json")
        print(f"  {len(matches)} matches")

        for i, m in enumerate(matches, 1):
            match_id = m["match_id"]
            all_match_ids.append(match_id)

            events_path = f"{EVENTS_DIR}/{match_id}.json"
            lineups_path = f"{LINEUPS_DIR}/{match_id}.json"

            if not os.path.exists(events_path):
                events = fetch_json(f"{BASE_RAW_URL}/events/{match_id}.json")
                save_json(events, events_path)

            if not os.path.exists(lineups_path):
                lineups = fetch_json(f"{BASE_RAW_URL}/lineups/{match_id}.json")
                save_json(lineups, lineups_path)

            if i % 10 == 0 or i == len(matches):
                print(f"  fetched {i}/{len(matches)} matches")

    print(f"\nDone. {len(all_match_ids)} total matches downloaded to {DATA_DIR}/")


if __name__ == "__main__":
    main()
