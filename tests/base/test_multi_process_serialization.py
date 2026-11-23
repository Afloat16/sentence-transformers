from __future__ import annotations

import pickle
import queue
import threading
from multiprocessing.reduction import ForkingPickler
from types import SimpleNamespace

import pytest
import torch.multiprocessing as mp

from sentence_transformers import CrossEncoder, MultiVectorEncoder, SentenceTransformer, SparseEncoder
from sentence_transformers.multi_vector_encoder.modules import LambdaTokenPooling


class _EchoInference:
    def _inference(self, inputs, **kwargs):
        return inputs


def _fail_reconstruction():
    raise ValueError("argument reconstruction failed")


class _FailsReconstruction:
    def __reduce__(self):
        return _fail_reconstruction, ()


class _RequiresRegisteredReducer:
    def __reduce__(self):
        raise TypeError("requires the multiprocessing reducer")


def _registered_reducer(value):
    return str, ("restored through registered reducer",)


@pytest.mark.parametrize("model_class", [SentenceTransformer, SparseEncoder, CrossEncoder, MultiVectorEncoder])
def test_invalid_inference_argument_is_rejected_before_enqueuing(model_class):
    class UnreadableResults:
        def get(self, **kwargs):
            pytest.fail("No chunk was submitted, so no result should be awaited")

    pool = {"input": queue.Queue(), "output": UnreadableResults(), "processes": [None]}
    model = model_class.__new__(model_class)
    with pytest.raises(ValueError, match="serialize multi-process input chunk 0") as error:
        model._multi_process(["text"], pool=pool, token_pooling=LambdaTokenPooling(lambda value: value))
    assert isinstance(error.value.__cause__, (pickle.PicklingError, AttributeError))
    assert pool["input"].empty()


def test_input_serialization_uses_registered_multiprocessing_reducers(monkeypatch):
    monkeypatch.setitem(ForkingPickler._extra_reducers, _RequiresRegisteredReducer, _registered_reducer)
    pool = {"input": queue.Queue(), "output": queue.Queue(), "processes": [None]}
    pool["output"].put([0, ["result"]])
    model = SentenceTransformer.__new__(SentenceTransformer)
    assert model._multi_process(["text"], pool=pool, show_progress_bar=False, option=_RequiresRegisteredReducer()) == [
        "result"
    ]
    chunk_id, payload = pool["input"].get_nowait()
    inputs, options = pickle.loads(payload)
    assert chunk_id == 0 and inputs == ["text"]
    assert options["option"] == "restored through registered reducer"


def test_full_input_queue_reports_dead_workers():
    inputs = queue.Queue(maxsize=1)
    inputs.put("occupied")
    worker = SimpleNamespace(is_alive=lambda: False, pid=4321, exitcode=-9)
    pool = {"input": inputs, "output": queue.Queue(), "processes": [worker]}
    model = SentenceTransformer.__new__(SentenceTransformer)
    with pytest.raises(RuntimeError, match=r"died unexpectedly: pid 4321 \(exit code -9\)"):
        model._multi_process(["text"], pool=pool, show_progress_bar=False)


@pytest.mark.parametrize("failure", ["argument", "later_input", "reconstruction"])
def test_serialization_failure_keeps_spawned_pool_reusable(failure):
    context = mp.get_context("spawn")
    inputs, outputs = context.Queue(maxsize=2), context.Queue()
    worker = context.Process(
        target=SentenceTransformer._multi_process_worker,
        args=("cpu", _EchoInference(), inputs, outputs),
        daemon=True,
    )
    worker.start()
    pool = {"input": inputs, "output": outputs, "processes": [worker]}
    model = SentenceTransformer.__new__(SentenceTransformer)
    guard = threading.Timer(30, worker.terminate)
    guard.start()
    try:
        assert model._multi_process(["initial"], pool=pool, show_progress_bar=False) == ["initial"]
        values, kwargs = ["first", "second"], {}
        message = "serialize multi-process input chunk"
        if failure == "argument":
            kwargs["token_pooling"] = LambdaTokenPooling(lambda value: value)
        elif failure == "later_input":
            values[1] = lambda: None
        else:
            kwargs["option"] = _FailsReconstruction()
            message = "argument reconstruction failed"
        with pytest.raises(ValueError, match=message):
            model._multi_process(values, pool=pool, chunk_size=1, show_progress_bar=False, **kwargs)
        # A later serialization error must drain successful chunks from the failed call.
        assert model._multi_process(["new", "request"], pool=pool, chunk_size=1, show_progress_bar=False) == [
            "new",
            "request",
        ]
        assert worker.is_alive()
    finally:
        guard.cancel()
        guard.join()
        model.stop_multi_process_pool(pool)
