"""Smoke test against a running stack.

Usage: uv run python scripts/smoke_test.py --api-key KEY [--base-url http://127.0.0.1:8000]
"""

import argparse
import sys

import httpx

SAMPLE = {
    "OverallQual": 7,
    "GrLivArea": 1800,
    "GarageCars": 2,
    "TotalBsmtSF": 900,
    "YearBuilt": 2005,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    results: list[bool] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail if not ok else ''}")
        results.append(ok)

    headers = {"X-API-Key": args.api_key}
    with httpx.Client(base_url=args.base_url, timeout=10) as client:
        check("health", client.get("/health").status_code == 200)

        response = client.get("/ready")
        check("ready (champion loaded)", response.status_code == 200, response.text)

        response = client.post("/predict", json=SAMPLE, headers=headers)
        check("predict", response.status_code == 200, response.text)
        if response.status_code == 200:
            prediction_id = response.json()["prediction_id"]
            body = {"prediction_id": prediction_id, "actual_sale_price": 210_000}
            response = client.post("/feedback", json=body, headers=headers)
            check("feedback", response.status_code == 201, response.text)

        response = client.post("/predict", json=SAMPLE, headers={"X-API-Key": "wrong"})
        check("rejects a wrong API key", response.status_code == 401)

        response = client.post("/predict", json={**SAMPLE, "OverallQual": 100}, headers=headers)
        check("rejects invalid input", response.status_code == 422)

        check("metrics exposed", "http_requests_total" in client.get("/metrics").text)

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
