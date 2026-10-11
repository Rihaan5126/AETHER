"""Black-box checks for a deployed AETHER dashboard and (optionally) API.

Uses only the standard library, so it runs anywhere. Every check prints
PASS / FAIL / SKIP with the observed value; nothing is assumed.

    python scripts/verify_deployment.py --web https://aetherbyrihaan.vercel.app
    python scripts/verify_deployment.py --api http://127.0.0.1:8001 --api-key KEY \\
        --allowed-origin https://aetherbyrihaan.vercel.app

The API checks never start training: the only write requests are deliberately
invalid (rejected before any job is created) or unauthenticated.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class Result:
    name: str
    status: str  # PASS | FAIL | SKIP
    detail: str


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    follow: bool = True,
) -> tuple[int, dict[str, str], bytes]:
    opener = urllib.request.build_opener() if follow else urllib.request.build_opener(NoRedirect)
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with opener.open(req, timeout=20) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, {k.lower(): v for k, v in err.headers.items()}, err.read()


def check(results: list[Result], name: str, ok: bool, detail: str) -> None:
    results.append(Result(name, "PASS" if ok else "FAIL", detail))


def verify_web(base: str, results: list[Result]) -> None:
    base = base.rstrip("/")
    status, headers, _ = request(base + "/")
    check(results, "web: HTTPS page loads", status == 200, f"status {status}")
    if base.startswith("https://"):
        s, h, _ = request("http://" + base.removeprefix("https://") + "/", follow=False)
        check(
            results,
            "web: HTTP redirects to HTTPS",
            s in (301, 307, 308) and h.get("location", "").startswith("https://"),
            f"status {s} -> {h.get('location')}",
        )
    hsts = headers.get("strict-transport-security", "")
    check(results, "web: HSTS", "max-age=" in hsts, hsts or "missing")
    check(
        results,
        "web: X-Frame-Options",
        headers.get("x-frame-options") == "DENY",
        str(headers.get("x-frame-options")),
    )
    check(
        results,
        "web: X-Content-Type-Options",
        headers.get("x-content-type-options") == "nosniff",
        str(headers.get("x-content-type-options")),
    )
    check(
        results,
        "web: Referrer-Policy",
        bool(headers.get("referrer-policy")),
        str(headers.get("referrer-policy")),
    )
    csp = headers.get("content-security-policy", "")
    check(
        results,
        "web: Content-Security-Policy",
        "default-src 'self'" in csp and "frame-ancestors 'none'" in csp,
        csp or "missing",
    )
    check(
        results,
        "web: no framework banner",
        "x-powered-by" not in headers,
        str(headers.get("x-powered-by")),
    )
    s, _, raw = request(base + "/snapshot/meta.json")
    if s == 200:
        meta = json.loads(raw)
        check(
            results,
            "web: snapshot metadata",
            "generated_at" in meta,
            f"exported {meta.get('generated_at')}, {len(meta.get('experiments', []))} experiments",
        )
        s2, _, raw2 = request(base + "/snapshot/system/status.json")
        status_doc = json.loads(raw2) if s2 == 200 else {}
        check(
            results,
            "web: snapshot is read-only",
            status_doc.get("write_protected") is True
            and status_doc.get("environment") == "snapshot",
            f"write_protected={status_doc.get('write_protected')}",
        )
        big = next(iter(meta.get("experiments", [])), None)
        if big:
            s3, h3, _ = request(
                base + f"/snapshot/experiments/{big['id']}/analytics.json",
                headers={"Accept-Encoding": "br, gzip"},
            )
            enc = h3.get("content-encoding", "none")
            check(
                results,
                "web: JSON served compressed",
                s3 == 200 and enc in ("br", "gzip"),
                f"status {s3}, content-encoding {enc}",
            )
    else:
        results.append(
            Result("web: snapshot metadata", "SKIP", f"no snapshot (status {s}); live-API mode")
        )


def verify_api(
    base: str, key: str | None, allowed_origin: str | None, results: list[Result]
) -> None:
    base = base.rstrip("/") + "/api/v1"
    s, _, raw = request(base + "/health")
    check(results, "api: health", s == 200 and json.loads(raw).get("status") == "ok", f"status {s}")
    s, _, raw = request(base + "/system/status")
    doc = json.loads(raw) if s == 200 else {}
    check(
        results,
        "api: system status",
        s == 200,
        f"environment={doc.get('environment')}, db ok={doc.get('database', {}).get('ok')}",
    )
    check(
        results,
        "api: write protection on",
        doc.get("write_protected") is True,
        f"write_protected={doc.get('write_protected')}",
    )
    check(
        results,
        "api: job queue bounded",
        isinstance(doc.get("limits", {}).get("max_pending_jobs"), int),
        f"max_pending_jobs={doc.get('limits', {}).get('max_pending_jobs')}",
    )

    if allowed_origin:
        pre = {
            "Origin": allowed_origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-api-key",
        }
        s, h, _ = request(base + "/experiments", method="OPTIONS", headers=pre)
        check(
            results,
            "api: CORS allows the dashboard origin",
            h.get("access-control-allow-origin") == allowed_origin,
            f"status {s}, allow-origin={h.get('access-control-allow-origin')}",
        )
        evil = {**pre, "Origin": "https://evil.example"}
        s, h, _ = request(base + "/experiments", method="OPTIONS", headers=evil)
        check(
            results,
            "api: CORS rejects other origins",
            "access-control-allow-origin" not in h,
            f"status {s}, allow-origin={h.get('access-control-allow-origin')}",
        )

    invalid = json.dumps({"name": "verify-deployment probe"}).encode()  # missing required fields
    s, _, raw = request(
        base + "/experiments",
        method="POST",
        headers={"Content-Type": "application/json"},
        body=invalid,
    )
    check(results, "api: write without key is refused", s == 401, f"status {s} {raw[:80]!r}")
    if key:
        s, _, raw = request(
            base + "/experiments",
            method="POST",
            headers={"Content-Type": "application/json", "X-API-Key": key},
            body=invalid,
        )
        check(results, "api: authenticated invalid write -> 422 (no job)", s == 422, f"status {s}")
        s, _, raw = request(
            base + "/experiments",
            method="POST",
            headers={"Content-Type": "application/json", "X-API-Key": key + "x"},
            body=invalid,
        )
        check(results, "api: wrong key is refused", s == 401, f"status {s}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--web")
    ap.add_argument("--api")
    ap.add_argument("--api-key")
    ap.add_argument("--allowed-origin")
    ap.add_argument("--json", action="store_true", help="print results as JSON")
    args = ap.parse_args()
    results: list[Result] = []
    if args.web:
        verify_web(args.web, results)
    if args.api:
        verify_api(args.api, args.api_key, args.allowed_origin, results)
    if args.json:
        print(json.dumps([r.__dict__ for r in results], indent=2))
    else:
        for r in results:
            print(f"{r.status:4}  {r.name:48}  {r.detail}")
    sys.exit(1 if any(r.status == "FAIL" for r in results) else 0)


if __name__ == "__main__":
    main()
