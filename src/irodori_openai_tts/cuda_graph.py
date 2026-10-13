"""Aratako/Irodori-TTS本体(irodori_ttsパッケージ、pip依存)には一切変更を加えず、
呼び出し側(このリポジトリ)だけでforward_with_encoded_conditions呼び出しをCUDA Graph
でキャプチャ・リプレイすることで高速化する。

## 前提と安全策
- デフォルトでは無効(settings.use_cuda_graph=False)。
- FlashAttention3(flash_attn_interface、通常は未インストールのoptional依存)が
  使える環境では、attention.build_context_attention_plan内でtorch.nonzeroによる
  動的shape計算(device syncを要する)が走り、CUDA Graphキャプチャ中にエラーまたは
  未検証の動作を起こす可能性があるため、検出時は常に無効化する。
- compile_model(torch.compile)と同時には使わない(二重適用で不定動作になりうる
  ため、compile_model側を優先してCUDA Graphは無効化する)。
- 動的LoRA適用(InferenceRuntime._prepare_lora_for_request)はself.modelを新しい
  PeftModelに差し替えるが、委譲チェーン経由で元のインスタンス(モンキーパッチ済み)
  に到達する可能性があるため、LoRA使用を検出した時点でCUDA Graphを永久停止する
  (プロセス再起動まで、全リクエストがeager実行にフォールバックする)。
- irodori_ttsはgitの浮動参照(pyproject.toml)でインストールされ、将来
  forward_with_encoded_conditionsの引数が変わる可能性がある(例: MeanFlow対応版は
  delta_t引数を持つ)。想定外の引数が増えているとTypeErrorでクラッシュするため、
  起動時にシグネチャを検査し、既知の契約と一致しない場合は無効化する
  (_forward_signature_matches_known_contract参照)。
- model.pyのget_timestep_embeddingは、timestepの値に依存しない定数(freqs)を
  毎回torch.tensor(10000.0, device=...)で新規作成しており、これがCUDA Graph
  キャプチャ中に"operation not permitted when stream is capturing"エラーを
  起こすことを実機で確認した。freqsをdevice/dimごとに1回だけ計算してキャッシュ
  する版にモンキーパッチする(_patch_timestep_embedding_for_capture参照。
  model.pyファイル自体は変更しない)。
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from irodori_tts.inference_runtime import InferenceRuntime

    from .config import Settings

logger = logging.getLogger(__name__)

# このラッパーが前提にしているforward_with_encoded_conditionsの引数契約。
# インストールされているirodori_ttsのバージョンによって引数が増えている場合
# (例: MeanFlow対応版のdelta_t)、ここに無い引数は未対応のため無効化する。
# runtime.model.forward_with_encoded_conditionsはインスタンス経由のバインド
# メソッドとして渡ってくるため、selfはシグネチャに含まれない(実機で確認済み:
# TextToLatentRFDiTのインスタンスメソッドをinspect.signatureしてもselfは出ない)。
_SUPPORTED_PARAM_NAMES = frozenset(
    {
        "x_t",
        "t",
        "text_state",
        "text_mask",
        "speaker_state",
        "speaker_mask",
        "caption_state",
        "caption_mask",
        "latent_mask",
        "context_kv_cache",
    }
)


def _forward_signature_matches_known_contract(original_forward: Any) -> bool:
    """実際のforward_with_encoded_conditionsの引数名が、このラッパーが前提にしている
    集合と完全に一致するかを確認する。部分集合(<=)だけの判定では、将来upstream側で
    既存の引数(例: caption_state)が削除・改名されても「余分な引数が無い」ため
    合格してしまい、その後の実呼び出しで欠落した引数によりTypeErrorになる
    (このラッパーは常に全既知引数をキーワードで渡すため)。完全一致(==)にすることで、
    引数が増えた場合・減った場合の両方を未対応として検出する。"""
    try:
        sig = inspect.signature(original_forward)
    except (TypeError, ValueError):
        return False
    return set(sig.parameters) == _SUPPORTED_PARAM_NAMES

# forward_with_encoded_conditionsが受け取る引数のうち、常にTensor|Noneの
# フラットな値であるもの(context_kv_cacheだけ別扱いのネスト構造)。
_TENSOR_ARG_NAMES = (
    "x_t",
    "t",
    "text_state",
    "text_mask",
    "speaker_state",
    "speaker_mask",
    "caption_state",
    "caption_mask",
)


def _tensor_sig(value: torch.Tensor | None) -> tuple | None:
    if value is None:
        return None
    return (tuple(value.shape), str(value.dtype))


def _kv_cache_sig(cache: list[tuple[torch.Tensor, ...]] | None) -> tuple | None:
    if cache is None:
        return None
    return tuple(tuple(_tensor_sig(t) for t in block) for block in cache)


def _make_signature(
    tensor_args: dict[str, torch.Tensor | None],
    context_kv_cache: list[tuple[torch.Tensor, ...]] | None,
) -> tuple:
    return (
        tuple(_tensor_sig(tensor_args[name]) for name in _TENSOR_ARG_NAMES),
        _kv_cache_sig(context_kv_cache),
    )


def _clone_tensor_or_none(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    buf = torch.empty_like(value)
    buf.copy_(value)
    return buf


def _clone_kv_cache(
    cache: list[tuple[torch.Tensor, ...]] | None,
) -> list[tuple[torch.Tensor, ...]] | None:
    if cache is None:
        return None
    return [tuple(_clone_tensor_or_none(t) for t in block) for block in cache]


def _copy_tensor_or_none(static: torch.Tensor | None, new: torch.Tensor | None) -> None:
    if static is None or new is None:
        if static is not None or new is not None:
            raise AssertionError(
                "shape signatureが一致したはずなのにNoneの有無が食い違っています"
            )
        return
    static.copy_(new)


def _copy_kv_cache(
    static: list[tuple[torch.Tensor, ...]] | None,
    new: list[tuple[torch.Tensor, ...]] | None,
) -> None:
    if static is None or new is None:
        if static is not None or new is not None:
            raise AssertionError(
                "shape signatureが一致したはずなのにcontext_kv_cacheの有無が食い違っています"
            )
        return
    for static_block, new_block in zip(static, new, strict=True):
        for static_t, new_t in zip(static_block, new_block, strict=True):
            static_t.copy_(new_t)


class _GraphEntry:
    __slots__ = ("graph", "static_inputs", "static_kv_cache", "static_output")

    def __init__(
        self,
        *,
        graph: torch.cuda.CUDAGraph,
        static_inputs: dict[str, torch.Tensor | None],
        static_kv_cache: list[tuple[torch.Tensor, ...]] | None,
        static_output: torch.Tensor,
    ) -> None:
        self.graph = graph
        self.static_inputs = static_inputs
        self.static_kv_cache = static_kv_cache
        self.static_output = static_output


class CudaGraphForwardManager:
    """model.forward_with_encoded_conditionsをCUDA Graphのキャプチャ・リプレイで
    置き換えるためのラッパー。shape signature(引数テンソルのshape/dtypeの組)
    ごとに独立したグラフをキャッシュする(LRU)。

    InferenceRuntime._infer_lock により呼び出しはプロセス内で常に直列化されている
    前提だが、防御的に自前のロックも持つ。
    """

    def __init__(
        self,
        *,
        original_forward: Any,
        device: torch.device,
        warmup_steps: int,
        max_entries: int,
    ) -> None:
        self._original_forward = original_forward
        self._device = device
        self._warmup_steps = max(1, int(warmup_steps))
        self._max_entries = max(1, int(max_entries))
        self._pool = torch.cuda.graph_pool_handle()
        self._graphs: OrderedDict[tuple, _GraphEntry] = OrderedDict()
        # ウォームアップ中(まだキャプチャに至っていない)shapeのカウンタ。
        # エントリはcuda_graph_warmup_steps回まで同じshapeが来たときにしか
        # 消費されない(_capture後にpopされる)ため、_graphsと同様のLRU上限を
        # 設けないと、入力長が多様な実運用で全く上限に達せず無制限に増え続ける
        # (発話ごとにtext長・推定秒数が変わり、同じshapeが繰り返されにくいため)。
        self._pending_warmup: OrderedDict[tuple, int] = OrderedDict()
        self._lock = threading.Lock()
        self.disabled = False
        self.disabled_reason: str | None = None
        self.hits = 0
        self.misses = 0
        self.captures = 0
        self.evictions = 0

    def disable_permanently(self, reason: str) -> None:
        with self._lock:
            if self.disabled:
                return
            self.disabled = True
            self.disabled_reason = reason
            self._graphs.clear()
            self._pending_warmup.clear()
        logger.warning(
            "CUDA Graphを永久停止しました(理由: %s)。以降はeager実行にフォールバックします。",
            reason,
        )

    def stats(self) -> dict[str, Any]:
        # 全グラフが同じメモリプール(self._pool)を共有するため、evictionが
        # Python側の_GraphEntryを捨てても、GPU側のメモリ確保が実際に減るかは
        # PyTorchのアロケータ次第で保証されない(README/計画書に既知のリスクと
        # して記載)。gpu_memory_*を公開し、長時間稼働でメモリが高止まり・増大
        # し続けていないかを検証者が/healthだけで確認できるようにする。
        return {
            "active": not self.disabled,
            "disabled_reason": self.disabled_reason,
            "cache_entries": len(self._graphs),
            "pending_warmup_entries": len(self._pending_warmup),
            "hits": self.hits,
            "misses": self.misses,
            "captures": self.captures,
            "evictions": self.evictions,
            "gpu_memory_allocated_bytes": torch.cuda.memory_allocated(self._device),
            "gpu_memory_reserved_bytes": torch.cuda.memory_reserved(self._device),
        }

    def __call__(
        self,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_state: torch.Tensor,
        text_mask: torch.Tensor,
        speaker_state: torch.Tensor | None = None,
        speaker_mask: torch.Tensor | None = None,
        caption_state: torch.Tensor | None = None,
        caption_mask: torch.Tensor | None = None,
        latent_mask: torch.Tensor | None = None,
        context_kv_cache: list[tuple[torch.Tensor, ...]] | None = None,
    ) -> torch.Tensor:
        tensor_args = {
            "x_t": x_t,
            "t": t,
            "text_state": text_state,
            "text_mask": text_mask,
            "speaker_state": speaker_state,
            "speaker_mask": speaker_mask,
            "caption_state": caption_state,
            "caption_mask": caption_mask,
        }
        # 未知の呼び出しパターン(latent_maskが指定される等)には関与せず、
        # 常に安全なeager実行に委ねる。
        if self.disabled or latent_mask is not None:
            return self._call_original(tensor_args, latent_mask=latent_mask, context_kv_cache=context_kv_cache)

        sig = _make_signature(tensor_args, context_kv_cache)

        with self._lock:
            entry = self._graphs.get(sig)
            if entry is not None:
                self._graphs.move_to_end(sig)
                self.hits += 1
                for name in _TENSOR_ARG_NAMES:
                    _copy_tensor_or_none(entry.static_inputs[name], tensor_args[name])
                _copy_kv_cache(entry.static_kv_cache, context_kv_cache)
                entry.graph.replay()
                return entry.static_output.clone()

            self.misses += 1
            warmup_count = self._pending_warmup.get(sig, 0)
            run_capture = warmup_count >= self._warmup_steps
            if not run_capture:
                self._pending_warmup[sig] = warmup_count + 1
                self._pending_warmup.move_to_end(sig)
                self._evict_pending_warmup_if_needed()

        if not run_capture:
            # ウォームアップ中は普通にeager実行する(cuBLAS/cuDNNのアルゴリズム
            # 選択・メモリ確保を安定させ、キャプチャ対象から除外するため)。
            return self._call_original(tensor_args, latent_mask=None, context_kv_cache=context_kv_cache)

        entry = self._capture(tensor_args, context_kv_cache)
        with self._lock:
            self._pending_warmup.pop(sig, None)
            self._evict_if_needed()
            self._graphs[sig] = entry
            self.captures += 1
        return entry.static_output.clone()

    def _call_original(
        self,
        tensor_args: dict[str, torch.Tensor | None],
        *,
        latent_mask: torch.Tensor | None,
        context_kv_cache: list[tuple[torch.Tensor, ...]] | None,
    ) -> torch.Tensor:
        """eager実行(元のforward_with_encoded_conditions呼び出し)の一本化。
        呼び出し箇所を1つに絞ることで、引数名の変更漏れ(一部の呼び出しだけ
        更新し忘れてTypeErrorになる)を防ぐ。"""
        return self._original_forward(
            **tensor_args, latent_mask=latent_mask, context_kv_cache=context_kv_cache
        )

    def _capture(
        self,
        tensor_args: dict[str, torch.Tensor | None],
        context_kv_cache: list[tuple[torch.Tensor, ...]] | None,
    ) -> _GraphEntry:
        static_inputs = {name: _clone_tensor_or_none(value) for name, value in tensor_args.items()}
        static_kv_cache = _clone_kv_cache(context_kv_cache)

        # PyTorch公式のCUDA Graph利用パターン: 専用のサイドストリームで
        # ウォームアップ実行してから、メインストリームに同期を戻してキャプチャする。
        side_stream = torch.cuda.Stream(device=self._device)
        side_stream.wait_stream(torch.cuda.current_stream(self._device))
        with torch.cuda.stream(side_stream):
            for _ in range(2):
                self._call_original(static_inputs, latent_mask=None, context_kv_cache=static_kv_cache)
        torch.cuda.current_stream(self._device).wait_stream(side_stream)
        torch.cuda.synchronize(self._device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            static_output = self._call_original(
                static_inputs, latent_mask=None, context_kv_cache=static_kv_cache
            )
        # キャプチャ直後のstatic_outputはまだ正しい値になっていない(PyTorchの
        # CUDA Graph APIの既知の挙動。キャプチャはストリームへの記録のみで、
        # 実際の計算はreplay()を呼ぶまで確定しない)。ここで1回replayして
        # static_outputを確定させてから返す。
        graph.replay()
        return _GraphEntry(
            graph=graph,
            static_inputs=static_inputs,
            static_kv_cache=static_kv_cache,
            static_output=static_output,
        )

    def _evict_if_needed(self) -> None:
        while len(self._graphs) >= self._max_entries:
            self._graphs.popitem(last=False)
            self.evictions += 1

    def _evict_pending_warmup_if_needed(self) -> None:
        # _graphsと同じ上限を使う。古いshapeを捨ててもウォームアップの
        # カウントが0から再開されるだけで安全(キャプチャ済みグラフと違い、
        # 失っても壊れた状態にはならない)。
        while len(self._pending_warmup) > self._max_entries:
            self._pending_warmup.popitem(last=False)


# InferenceRuntime._resolve_lora_adapter_path(irodori_tts/inference_runtime.py)が
# "LoRA無し"として扱う文字列と同じ正規化ルール。ここでの判定は検出のみが目的
# (実際のパス存在確認・adapter_config.json検証はInferenceRuntime.synthesize側の
# _prepare_lora_for_requestに委ね、ここでファイルI/Oを重複させない)。
_LORA_ADAPTER_OFF_VALUES = frozenset({"", "none", "null", "off", "disable", "disabled", "base"})


def _lora_requested(req: Any) -> bool:
    """reqがLoRAアダプタを要求しているかどうかを、ファイルI/Oを伴わない
    軽量な文字列正規化だけで判定する。"""
    raw = getattr(req, "lora_adapter", None)
    if raw is None:
        return False
    return str(raw).strip().lower() not in _LORA_ADAPTER_OFF_VALUES


def _install_lora_kill_switch(runtime: InferenceRuntime, manager: CudaGraphForwardManager) -> None:
    original_synthesize = runtime.synthesize

    def guarded_synthesize(req: Any, **kwargs: Any) -> Any:
        if not manager.disabled and _lora_requested(req):
            manager.disable_permanently(
                reason=f"dynamic lora_adapter={getattr(req, 'lora_adapter', None)!r} requested"
            )
        return original_synthesize(req, **kwargs)

    runtime.synthesize = guarded_synthesize


def _patch_timestep_embedding_for_capture() -> None:
    """irodori_tts.model.get_timestep_embeddingは、timestepの値に依存しない
    定数(freqs)を毎回torch.tensor(10000.0, device=...)で新規作成している。
    Pythonスカラーから新規にホスト→デバイスのテンソルを作る操作はCUDA Graph
    キャプチャ中には許可されず、"operation not permitted when stream is
    capturing"エラーになる(実機で確認)。freqsはdim(モデル設定で固定)が同じ
    なら常に同じ値になるため、device/dimごとに1回だけ計算してキャッシュする
    版にモジュール関数自体を差し替える(model.pyファイルは変更しない)。

    CUDA Graph無効時に適用しても元の実装と同じ結果を返す(むしろ無駄な
    再計算が減る)ため、無条件に適用してよい。冪等(二重適用を防ぐ)。"""
    import irodori_tts.model as irodori_model

    if getattr(irodori_model.get_timestep_embedding, "_cuda_graph_patched", False):
        return

    freqs_cache: dict[tuple[torch.device, int], torch.Tensor] = {}

    def patched_get_timestep_embedding(timestep: torch.Tensor, dim: int) -> torch.Tensor:
        assert dim % 2 == 0
        half = dim // 2
        key = (timestep.device, dim)
        freqs = freqs_cache.get(key)
        if freqs is None:
            freqs = 1000.0 * torch.exp(
                -torch.log(torch.tensor(10000.0, device=timestep.device, dtype=torch.float32))
                * torch.arange(half, device=timestep.device, dtype=torch.float32)
                / half
            )
            freqs_cache[key] = freqs
        args = timestep[:, None].float() * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(timestep.dtype)

    patched_get_timestep_embedding._cuda_graph_patched = True
    irodori_model.get_timestep_embedding = patched_get_timestep_embedding
    logger.info("get_timestep_embeddingをCUDA Graph対応版(freqsキャッシュ)に差し替えました。")


def maybe_install(runtime: InferenceRuntime, settings: Settings) -> None:
    """settings.use_cuda_graphが有効で、既知の非互換条件に当たらない場合のみ、
    runtime.model.forward_with_encoded_conditionsをCUDA Graph対応のラッパーに
    差し替える。irodori_tts本体のコードは一切変更しない(モンキーパッチのみ)。"""
    if getattr(runtime, "_cuda_graph_installed", False):
        return
    if not settings.use_cuda_graph:
        return
    if runtime.model_device.type != "cuda":
        logger.info(
            "CUDA Graphは無効です(model_deviceがcudaではありません: %s)", runtime.model_device
        )
        return
    if importlib.util.find_spec("flash_attn_interface") is not None:
        # パッケージが同じvenvにインストールされているだけで無効化する(実際に
        # このコードパスで使われているかまでは見ない)。意図的に過剰検出(false
        # positive)側に倒した判断: FA3が使われている場合、attention.py内の
        # pack_index計算(torch.nonzero、device syncを要する)がCUDA Graph
        # キャプチャ下で未検証のため、無関係な理由でインストールされていただけの
        # 場合でも安全側に無効化する。本当に使われているかどうかの判定は複雑に
        # なるため採用していない。
        logger.warning(
            "CUDA Graphは無効です(flash_attn_interfaceが検出されました。"
            "FlashAttention3経路の動的shape処理はCUDA Graphキャプチャ下で未検証のため)。"
        )
        return
    if runtime.key.compile_model:
        logger.warning(
            "CUDA Graphは無効です(compile_model=trueがforward_with_encoded_conditionsを"
            "既にtorch.compile済みのため、両方の同時適用は非対応です)。"
        )
        return
    if not _forward_signature_matches_known_contract(runtime.model.forward_with_encoded_conditions):
        logger.warning(
            "CUDA Graphは無効です(forward_with_encoded_conditionsの引数がこの実装の"
            "前提と一致しません。irodori_ttsのバージョンが更新され、未対応の引数"
            "(例: MeanFlow対応のdelta_t)が追加された可能性があります)。"
        )
        return

    _patch_timestep_embedding_for_capture()

    manager = CudaGraphForwardManager(
        original_forward=runtime.model.forward_with_encoded_conditions,
        device=runtime.model_device,
        warmup_steps=settings.cuda_graph_warmup_steps,
        max_entries=settings.cuda_graph_max_cache_entries,
    )
    runtime.model.forward_with_encoded_conditions = manager
    _install_lora_kill_switch(runtime, manager)
    runtime._cuda_graph_installed = True
    runtime._cuda_graph_manager = manager
    logger.info(
        "CUDA Graphを有効化しました(warmup_steps=%d, max_cache_entries=%d)",
        settings.cuda_graph_warmup_steps,
        settings.cuda_graph_max_cache_entries,
    )
