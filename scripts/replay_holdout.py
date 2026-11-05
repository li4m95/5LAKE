"""Replay recent real sales through the API, then send their true prices as feedback.

Usage: uv run python scripts/replay_holdout.py --api-key KEY [--rows 200] [--price-shift 1.0]
"""

import argparse

import httpx
import pandas as pd

from mlops import settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument(
        "--price-shift",
        type=float,
        default=1.0,
        help="Multiply real prices by this factor to simulate a market change.",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(settings.PROCESSED_PATH)
    recent = frame.sort_values(settings.TIMESTAMP).tail(args.rows)

    headers = {"X-API-Key": args.api_key}
    with httpx.Client(base_url=args.base_url, timeout=10, headers=headers) as client:
        for record in recent.to_dict("records"):
            payload = {name: record[name] for name in settings.FEATURES}
            response = client.post("/predict", json=payload)
            response.raise_for_status()
            feedback = {
                "prediction_id": response.json()["prediction_id"],
                "actual_sale_price": record[settings.TARGET] * args.price_shift,
            }
            client.post("/feedback", json=feedback).raise_for_status()
    print(f"Sent {len(recent)} predictions with feedback (price shift {args.price_shift}).")


if __name__ == "__main__":
    main()
