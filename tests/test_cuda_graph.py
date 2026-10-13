from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from irodori_openai_tts import cuda_graph as cg

_CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA GPUが無い環境ではCUDA Graph自体を検証できない")


def _known_contract_forward(
    *,
    x_t=None,
    t=None,
    text_state=None,
    text_mask=None,
    speaker_state=None,
    speaker_mask=None,
    caption_state=None,
    caption_mask=None,
    latent_mask=None,
    context_kv_cache=None,
):
    """本物のforward_with_encoded_conditionsと同じ引数契約を持つダミー。"""
    return None


def _unknown_contract_forward(*, x_t=None, t=None, text_state=None, text_mask=None, delta_t=None):
    """MeanFlow対応版等、未知の引数(delta_t)を持つ将来のバージョンを模すダミー。"""
    return None


def _make_runtime(
    *, device_type: str = "cuda", compile_model: bool = False, forward_fn=_known_contract_forward
) -> SimpleNamespace:
    model = SimpleNamespace(forward_with_encoded_conditions=forward_fn)
    return SimpleNamespace(
        model=model,
        model_device=torch.device(device_type),
        key=SimpleNamespace(compile_model=compile_model),
        synthesize=lambda req, **kw: None,
    )


def _make_settings(**overrides) -> SimpleNamespace:
    base = {"use_cuda_graph": True, "cuda_graph_warmup_steps": 3, "cuda_graph_max_cache_entries": 8}
    base.update(overrides)
    return SimpleNamespace(**base)


class TestMaybeInstallGuards:
    def test_disabled_by_default(self):
        runtime = _make_runtime()
        settings = _make_settings(use_cuda_graph=False)
        cg.maybe_install(runtime, settings)
        assert getattr(runtime, "_cuda_graph_installed", False) is False

    def test_disabled_on_non_cuda_device(self):
        runtime = _make_runtime(device_type="cpu")
        settings = _make_settings()
        cg.maybe_install(runtime, settings)
        assert getattr(runtime, "_cuda_graph_installed", False) is False

    def test_disabled_when_compile_model_enabled(self):
        runtime = _make_runtime(compile_model=True)
        settings = _make_settings()
        cg.maybe_install(runtime, settings)
        assert getattr(runtime, "_cuda_graph_installed", False) is False

    @requires_cuda
    def test_disabled_when_forward_signature_has_unknown_params(self):
        """irodori_ttsが更新され、未対応の引数(例: delta_t)を持つ
        forward_with_encoded_conditionsに変わった場合、クラッシュせず
        無効化されることを確認する(過去にdelta_tをダミー側だけが知らずに
        受け取ってしまい本物では全リクエストがTypeErrorで落ちたバグの再発防止)。
        デバイスチェックより後段のガードを検証するためCUDA環境が必要。"""
        runtime = _make_runtime(forward_fn=_unknown_contract_forward)
        settings = _make_settings()
        cg.maybe_install(runtime, settings)
        assert getattr(runtime, "_cuda_graph_installed", False) is False

    @requires_cuda
    def test_installs_on_cuda_without_conflicts(self):
        runtime = _make_runtime()
        settings = _make_settings()
        cg.maybe_install(runtime, settings)
        assert runtime._cuda_graph_installed is True
        assert isinstance(runtime.model.forward_with_encoded_conditions, cg.CudaGraphForwardManager)

    @requires_cuda
    def test_idempotent_on_second_call(self):
        runtime = _make_runtime()
        settings = _make_settings()
        cg.maybe_install(runtime, settings)
        first_manager = runtime.model.forward_with_encoded_conditions
        cg.maybe_install(runtime, settings)
        assert runtime.model.forward_with_encoded_conditions is first_manager


class TestLoraKillSwitch:
    @staticmethod
    def _runtime_stub():
        return SimpleNamespace(synthesize=lambda req, **kw: "ok")

    def test_no_lora_does_not_disable(self):
        runtime = self._runtime_stub()
        manager = SimpleNamespace(disabled=False, disabled_reason=None, disable_permanently=lambda reason: None)
        cg._install_lora_kill_switch(runtime, manager)
        runtime.synthesize(SimpleNamespace(lora_adapter=None))
        assert manager.disabled is False

    def test_none_like_string_does_not_disable(self):
        runtime = self._runtime_stub()
        calls = []
        manager = SimpleNamespace(disabled=False, disabled_reason=None, disable_permanently=lambda reason: calls.append(reason))
        cg._install_lora_kill_switch(runtime, manager)
        runtime.synthesize(SimpleNamespace(lora_adapter="none"))
        assert calls == []

    def test_real_adapter_disables(self):
        runtime = self._runtime_stub()
        calls = []
        manager = SimpleNamespace(disabled=False, disabled_reason=None, disable_permanently=lambda reason: calls.append(reason))
        cg._install_lora_kill_switch(runtime, manager)
        runtime.synthesize(SimpleNamespace(lora_adapter="/some/real/adapter"))
        assert len(calls) == 1
        assert "/some/real/adapter" in calls[0]

    def test_unrecognized_value_disables_on_safe_side(self):
        """"off"系の既知の文字列のいずれでもない値は、常にLoRA使用とみなす
        (ファイルI/Oを伴う厳密なパス検証はInferenceRuntime.synthesize側に委ねる)。"""
        runtime = self._runtime_stub()
        calls = []
        manager = SimpleNamespace(disabled=False, disabled_reason=None, disable_permanently=lambda reason: calls.append(reason))
        cg._install_lora_kill_switch(runtime, manager)
        runtime.synthesize(SimpleNamespace(lora_adapter="not-a-real-path-but-not-off-either"))
        assert len(calls) == 1

    def test_does_not_call_resolve_lora_adapter_path(self):
        """判定にファイルI/Oを伴うruntime._resolve_lora_adapter_pathを使わない
        (synthesize側の_prepare_lora_for_requestと重複I/Oしないことの確認)。"""
        runtime = self._runtime_stub()

        def boom(_path):
            raise AssertionError("_resolve_lora_adapter_pathを呼んではいけない")

        runtime._resolve_lora_adapter_path = boom
        manager = SimpleNamespace(disabled=False, disabled_reason=None, disable_permanently=lambda reason: None)
        cg._install_lora_kill_switch(runtime, manager)
        runtime.synthesize(SimpleNamespace(lora_adapter="/some/real/adapter"))


@requires_cuda
class TestCudaGraphForwardManager:
    """ダミーのforward_with_encoded_conditions相当の関数でキャプチャ・リプレイの
    正しさを検証する。実モデル(Aratako/Irodori-TTS)との統合は別PCでの
    scripts/verify_cuda_graph_parity.py実行で確認する。"""

    @staticmethod
    def _dummy_forward(*, x_t, t, text_state, text_mask, speaker_state=None, speaker_mask=None, caption_state=None, caption_mask=None, latent_mask=None, context_kv_cache=None):
        return x_t * 2.0 + text_state.sum(dim=1, keepdim=True) * t.mean()

    def test_dummy_forward_matches_known_contract(self):
        """このダミーの引数契約が、本物のforward_with_encoded_conditionsを前提にした
        _SUPPORTED_PARAM_NAMESと食い違っていないことを保証する(過去に実際のシグネチャに
        存在しないdelta_t引数をダミー側だけが受け取ってしまい、全リクエストを
        クラッシュさせるバグをテストが検出できなかった事例があるため)。"""
        assert cg._forward_signature_matches_known_contract(self._dummy_forward)

    @staticmethod
    def _make_inputs(device, n, seed):
        g = torch.Generator(device=device).manual_seed(seed)
        x_t = torch.randn(1, n, device=device, generator=g)
        t = torch.full((1,), 0.5, device=device)
        text_state = torch.randn(1, n, device=device, generator=g)
        text_mask = torch.ones(1, n, dtype=torch.bool, device=device)
        return x_t, t, text_state, text_mask

    def test_replay_matches_eager_baseline(self):
        device = torch.device("cuda")
        manager = cg.CudaGraphForwardManager(
            original_forward=self._dummy_forward, device=device, warmup_steps=2, max_entries=4
        )
        x_t, t, text_state, text_mask = self._make_inputs(device, 4, seed=42)
        baseline = self._dummy_forward(x_t=x_t, t=t, text_state=text_state, text_mask=text_mask)

        for _ in range(6):
            out = manager(x_t=x_t, t=t, text_state=text_state, text_mask=text_mask)
            assert torch.allclose(out, baseline)
        assert manager.stats()["captures"] == 1

    def test_distinct_shapes_get_distinct_graphs_without_cross_contamination(self):
        device = torch.device("cuda")
        manager = cg.CudaGraphForwardManager(
            original_forward=self._dummy_forward, device=device, warmup_steps=1, max_entries=4
        )
        x_t_a, t_a, text_a, mask_a = self._make_inputs(device, 4, seed=1)
        x_t_b, t_b, text_b, mask_b = self._make_inputs(device, 6, seed=2)
        baseline_a = self._dummy_forward(x_t=x_t_a, t=t_a, text_state=text_a, text_mask=mask_a)
        baseline_b = self._dummy_forward(x_t=x_t_b, t=t_b, text_state=text_b, text_mask=mask_b)

        for _ in range(3):
            assert torch.allclose(manager(x_t=x_t_a, t=t_a, text_state=text_a, text_mask=mask_a), baseline_a)
            assert torch.allclose(manager(x_t=x_t_b, t=t_b, text_state=text_b, text_mask=mask_b), baseline_b)
        # 両方キャプチャ済みのはずなのに、互いの出力が混ざっていないことを再確認。
        assert torch.allclose(manager(x_t=x_t_a, t=t_a, text_state=text_a, text_mask=mask_a), baseline_a)

    def test_lru_eviction_respects_max_entries(self):
        device = torch.device("cuda")
        manager = cg.CudaGraphForwardManager(
            original_forward=self._dummy_forward, device=device, warmup_steps=1, max_entries=3
        )
        for n in range(2, 7):
            x_t, t, text_state, text_mask = self._make_inputs(device, n, seed=n)
            for _ in range(2):
                manager(x_t=x_t, t=t, text_state=text_state, text_mask=text_mask)
            assert manager.stats()["cache_entries"] <= 3
        assert manager.stats()["evictions"] > 0

    def test_pending_warmup_does_not_grow_without_bound(self):
        """入力長が毎回異なり同じshapeが繰り返されない(=warmup_stepsに到達せず
        captureされない)実運用を模す。_pending_warmupにもLRU上限が無いと、
        shapeごとに1エントリが永久に残り続けてしまう。"""
        device = torch.device("cuda")
        manager = cg.CudaGraphForwardManager(
            original_forward=self._dummy_forward, device=device, warmup_steps=5, max_entries=3
        )
        for n in range(2, 20):
            x_t, t, text_state, text_mask = self._make_inputs(device, n, seed=n)
            manager(x_t=x_t, t=t, text_state=text_state, text_mask=text_mask)
            assert manager.stats()["pending_warmup_entries"] <= 3
        assert manager.stats()["captures"] == 0

    def test_disabled_manager_always_falls_back_to_eager(self):
        device = torch.device("cuda")
        manager = cg.CudaGraphForwardManager(
            original_forward=self._dummy_forward, device=device, warmup_steps=1, max_entries=4
        )
        x_t, t, text_state, text_mask = self._make_inputs(device, 4, seed=7)
        baseline = self._dummy_forward(x_t=x_t, t=t, text_state=text_state, text_mask=text_mask)
        manager(x_t=x_t, t=t, text_state=text_state, text_mask=text_mask)
        manager(x_t=x_t, t=t, text_state=text_state, text_mask=text_mask)
        assert manager.stats()["captures"] == 1

        manager.disable_permanently(reason="test")
        out = manager(x_t=x_t, t=t, text_state=text_state, text_mask=text_mask)
        assert torch.allclose(out, baseline)
        assert manager.stats()["cache_entries"] == 0


@requires_cuda
class TestTimestepEmbeddingPatch:
    """irodori_tts.model.get_timestep_embeddingは、timestepの値に依存しない
    定数(freqs)を毎回torch.tensor(10000.0, device=...)で新規作成しており、
    これが実機でCUDA Graphキャプチャ中に"operation not permitted when stream
    is capturing"エラーを起こすことを確認済み(別PCでの検証で発生)。
    _patch_timestep_embedding_for_captureがこれを解決することを検証する。"""

    def test_patched_output_matches_original(self):
        import irodori_tts.model as irodori_model

        device = torch.device("cuda")
        original_fn = irodori_model.get_timestep_embedding
        t = torch.tensor([0.3, 0.7], device=device)
        baseline = original_fn(t, 128)

        cg._patch_timestep_embedding_for_capture()
        patched_fn = irodori_model.get_timestep_embedding
        out = patched_fn(t, 128)

        assert torch.equal(out, baseline)

    def test_idempotent(self):
        import irodori_tts.model as irodori_model

        cg._patch_timestep_embedding_for_capture()
        patched_fn = irodori_model.get_timestep_embedding
        cg._patch_timestep_embedding_for_capture()

        assert irodori_model.get_timestep_embedding is patched_fn

    def test_patched_function_is_capturable(self):
        """修正前はここでCUDA Graphキャプチャ自体が例外になっていた
        (operation not permitted when stream is capturing)。"""
        import irodori_tts.model as irodori_model

        device = torch.device("cuda")
        cg._patch_timestep_embedding_for_capture()
        patched_fn = irodori_model.get_timestep_embedding
        t = torch.tensor([0.3, 0.7], device=device)
        baseline = patched_fn(t, 128)

        side_stream = torch.cuda.Stream(device=device)
        side_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side_stream):
            for _ in range(2):
                patched_fn(t, 128)
        torch.cuda.current_stream(device).wait_stream(side_stream)
        torch.cuda.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_out = patched_fn(t, 128)
        graph.replay()

        assert torch.equal(captured_out, baseline)
