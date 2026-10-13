"""CUDA Graph有効化による推論結果が、無効時のeager実行結果と完全に一致するかを
検証するスタンドアロンスクリプト。開発PC(Windows、Smart App Controlでpyarrowが
ブロックされ、transformers経由のテキストエンコーダが使えずサーバー自体を
起動できない)では実行できないため、別PC(GPU環境)で実行することを前提にする。

実行方法(irodori-tts-serverディレクトリで):
    uv run --no-sync python scripts/verify_cuda_graph_parity.py

事前に.envでIRODORI_HF_CHECKPOINT等、通常のサーバー起動と同じ設定をしておくこと。
このスクリプトはHTTPサーバーを起動せず、InferenceRuntimeを直接呼ぶ
(非同期・セマフォ層を迂回し、問題を切り分けやすくするため)。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from irodori_openai_tts import cuda_graph  # noqa: E402
from irodori_openai_tts.config import get_settings  # noqa: E402
from irodori_openai_tts.runtime import RuntimeManager  # noqa: E402
from irodori_tts.inference_runtime import SamplingRequest  # noqa: E402

FIXED_TEXT = "これはCUDA Graphの検証用の固定文章です。品質が変わっていないか確認します。"
SEED = 1234

# (num_candidates, seconds, cfg_guidance_mode)の組み合わせ。seedが同じでも
# 1リクエスト内でforward_with_encoded_conditionsは複数の異なるshapeで呼ばれる
# (CFG有無でバッチが変わる等)ため、代表的な組み合わせを総当たりする。
COMBOS = [
    {"num_candidates": 1, "seconds": 2.0, "cfg_guidance_mode": "independent"},
    {"num_candidates": 2, "seconds": 2.0, "cfg_guidance_mode": "independent"},
    {"num_candidates": 1, "seconds": 10.0, "cfg_guidance_mode": "independent"},
    {"num_candidates": 1, "seconds": 2.0, "cfg_guidance_mode": "joint"},
    {"num_candidates": 1, "seconds": 2.0, "cfg_guidance_mode": "alternating"},
]


def _find_sample_ref_wav() -> str | None:
    voices_dir = Path(__file__).resolve().parent.parent / "voices"
    wavs = sorted(voices_dir.glob("*.wav"))
    return str(wavs[0]) if wavs else None


def _run_once(runtime, **kwargs) -> torch.Tensor:
    req = SamplingRequest(text=FIXED_TEXT, seed=SEED, ref_wav=_find_sample_ref_wav(), **kwargs)
    result = runtime.synthesize(req)
    return result.audio


def main() -> None:
    if not torch.cuda.is_available():
        print("FAIL: CUDA GPUが見つかりません。GPU環境で実行してください。")
        sys.exit(1)

    settings = get_settings()
    settings.use_cuda_graph = False
    manager = RuntimeManager(settings)

    print("=== モデルをロード ===")
    runtime = manager.get()

    print("=== 各comboのベースライン(CUDA Graph無効)を先に取得 ===")
    baselines: dict[int, torch.Tensor] = {}
    for idx, combo in enumerate(COMBOS):
        baselines[idx] = _run_once(runtime, **combo)
        print(f"  combo {idx} ({combo}) baseline取得完了")

    print("=== CUDA Graphを有効化 ===")
    settings.use_cuda_graph = True
    cuda_graph.maybe_install(runtime, settings)
    if not getattr(runtime, "_cuda_graph_installed", False):
        print("FAIL: CUDA Graphが有効化されませんでした(ログを確認してください)。")
        sys.exit(1)
    cg_manager = runtime._cuda_graph_manager

    all_ok = True

    print("=== 同一リクエスト(combo 0)を複数回実行し、baselineと比較 ===")
    repeat_count = cg_manager._warmup_steps + 3
    for i in range(repeat_count):
        out = _run_once(runtime, **COMBOS[0])
        match = bool(torch.allclose(out, baselines[0], atol=1e-4, rtol=1e-4))
        all_ok = all_ok and match
        print(f"  call {i}: match={match} stats={cg_manager.stats()}")

    print("=== 各comboを再実行し、対応するbaselineと比較 ===")
    for idx, combo in enumerate(COMBOS):
        out = None
        for _ in range(cg_manager._warmup_steps + 2):
            out = _run_once(runtime, **combo)
        match = bool(torch.allclose(out, baselines[idx], atol=1e-4, rtol=1e-4))
        all_ok = all_ok and match
        print(f"  combo {idx} ({combo}): match={match}")

    print("=== LoRAキルスイッチの検証(存在しないLoRAパスを指定) ===")
    was_disabled_before = cg_manager.disabled
    try:
        _run_once(runtime, lora_adapter="/nonexistent/lora/adapter")
    except Exception as exc:  # noqa: BLE001 - 想定内(パス不正によるエラー)
        print(f"  (synthesize側のエラーは想定内: {exc})")
    disabled_after = cg_manager.disabled
    print(f"  disabled before={was_disabled_before} after={disabled_after} reason={cg_manager.disabled_reason}")
    all_ok = all_ok and (not was_disabled_before) and disabled_after

    print()
    print(f"最終stats: {cg_manager.stats()}")
    print("PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
