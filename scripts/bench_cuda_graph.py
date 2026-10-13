"""CUDA Graph有効/無効時の音声合成速度を比較するベンチマーク。
事前にサーバーを起動しておくこと(別ターミナルで):

    IRODORI_USE_CUDA_GRAPH=false uv run --no-sync python -m irodori_openai_tts --port 8088
    python scripts/bench_cuda_graph.py --base-url http://localhost:8088 --label baseline

その後サーバーを再起動して:

    IRODORI_USE_CUDA_GRAPH=true uv run --no-sync python -m irodori_openai_tts --port 8088
    python scripts/bench_cuda_graph.py --base-url http://localhost:8088 --label cuda-graph

出力されたp50/p95を見比べる。SyncDesk本番の運用特性(文単位の短いテキストが多い)に
近づけたい場合は--textで実際のチャット発話に近い短文を渡すこと。
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import httpx

DEFAULT_TEXT = "これはCUDA Graphのベンチマーク用の固定文章です。生成速度を測定します。"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8088")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5, help="計測から除外する先頭の実行回数")
    parser.add_argument("--label", default="run")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("IRODORI_API_KEY"),
        help="サーバー起動時にIRODORI_API_KEYを設定している場合に指定する"
        "(環境変数IRODORI_API_KEYからも読む)。/v1/audio/speechはAPIキーが"
        "設定されていると認証必須になるため、省略すると401 Unauthorizedになる。",
    )
    args = parser.parse_args()

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    client = httpx.Client(base_url=args.base_url, headers=headers, timeout=120.0)

    health_before = client.get("/health").json()
    print(f"[{args.label}] health.model: {health_before['model']}")
    print(f"[{args.label}] health.cuda_graph (開始時): {health_before.get('cuda_graph')}")

    durations: list[float] = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        resp = client.post(
            "/v1/audio/speech",
            json={
                "model": "irodori-tts",
                "input": args.text,
                "voice": "none",
                "response_format": "wav",
            },
        )
        resp.raise_for_status()
        elapsed = time.perf_counter() - t0
        if i >= args.warmup:
            durations.append(elapsed)
        print(f"[{args.label}] run {i}: {elapsed * 1000:.1f}ms (bytes={len(resp.content)})")

    health_after = client.get("/health").json()
    print(f"[{args.label}] health.cuda_graph (終了時): {health_after.get('cuda_graph')}")

    if durations:
        p50 = statistics.median(durations)
        sorted_durations = sorted(durations)
        p95_index = max(0, int(len(sorted_durations) * 0.95) - 1)
        p95 = sorted_durations[p95_index]
        print(
            f"[{args.label}] p50={p50 * 1000:.1f}ms p95={p95 * 1000:.1f}ms "
            f"(n={len(durations)}, warmup={args.warmup}件を除外)"
        )
    else:
        print(f"[{args.label}] 計測対象のrunが無い(--warmup >= --runsになっていないか確認)")


if __name__ == "__main__":
    main()
