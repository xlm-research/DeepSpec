"""Exercise real Mooncake Store with registered CPU buffers and owned services."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from deepspec.pipeline.store import FIELDS, TensorStore, describe_tensors, free_port


def test_cpu_contract_remove_requires_confirmed_absence():
    import threading
    from types import SimpleNamespace

    store = TensorStore.__new__(TensorStore)
    store._closed = False
    store._io_lock = threading.Lock()
    store.client = SimpleNamespace(
        batch_remove=lambda keys, **kw: [0] * len(keys),
        batch_is_exist=lambda keys: [1] * len(keys),
    )
    fields = {"x": {"chunks": [{"key": "r/sample/x"}]}}
    with pytest.raises(RuntimeError, match="visible|exist"):
        store.remove(fields)
    store.client.batch_is_exist = lambda keys: [-500] * len(keys)
    with pytest.raises(RuntimeError, match="existence"):
        store.remove(fields)
    store.client.batch_is_exist = lambda keys: [0] * len(keys)
    store.remove(fields)


def test_cpu_contract_same_bytes_wrong_shape_or_dtype_cannot_write():
    store = TensorStore.__new__(TensorStore)
    tensor = torch.zeros(2, 4, dtype=torch.float32)
    fields = {
        "x": {
            "shape": [2, 4],
            "dtype": "float32",
            "nbytes": 32,
            "chunks": [{"key": "fixture/x", "offset": 0, "nbytes": 32}],
        }
    }
    fields["x"]["shape"] = [4, 2]
    with pytest.raises(ValueError, match="shape|dtype"):
        store._plan_put(fields, {"x": tensor})


def test_cpu_contract_prefetch_timeout_keeps_inflight_bytes(monkeypatch):
    import threading
    from types import SimpleNamespace

    from deepspec.pipeline.prefetch import FeaturePrefetch

    release = threading.Event()
    reads = []

    def get(fields, names, **kwargs):
        reads.append(1)
        release.wait(2)
        return {name: torch.zeros(1) for name in names}

    monkeypatch.setattr(
        "deepspec.pipeline.prefetch.TensorStore",
        lambda *a, **kw: SimpleNamespace(get=get, close=lambda **kw: None),
    )
    prefetch = FeaturePrefetch({}, depth=2, max_bytes=8, timeout=0.01)
    descriptor = {
        "position": 0,
        "fields": {
            name: {"nbytes": 4}
            for name in ("target_hidden_states", "target_last_hidden_states")
        },
    }
    try:
        prefetch.submit(descriptor)
        with pytest.raises(TimeoutError):
            prefetch.take(0)
        assert prefetch.pending_bytes == 8
        with pytest.raises(RuntimeError, match="byte budget"):
            prefetch.submit({**descriptor, "position": 1})
        assert len(reads) == 1
        with pytest.raises(TimeoutError):
            prefetch.close(timeout=0.01)
        assert prefetch.pending_bytes == 8
        release.set()
        prefetch.take(0)
        assert prefetch.pending_bytes == 0
    finally:
        release.set()
        prefetch.close()


@pytest.mark.parametrize("native_failure", [False, True])
def test_cpu_contract_store_close_timeout_does_not_claim_native_cleanup(native_failure):
    import threading
    from types import SimpleNamespace

    entered, release = threading.Event(), threading.Event()

    def native_close():
        entered.set()
        assert release.wait(2)
        if native_failure:
            raise RuntimeError("native close failed")
        return 0

    store = TensorStore.__new__(TensorStore)
    store._closed = False
    store._put_manager = store._host_buffer_pool = store._get_executor = None
    store._io_lock = threading.RLock()
    store._close_lock = threading.Lock()
    store._close_finished = threading.Event()
    store._close_thread, store._close_error = None, None
    store.quarantined = ["native-buffer"]
    store._registered_buffers = {1: "retained"}
    store.client = SimpleNamespace(close=native_close)
    try:
        with pytest.raises(TimeoutError, match="close"):
            store.close(timeout=0.01)
        assert entered.is_set()
        assert store._registered_buffers and store.quarantined
        with pytest.raises(TimeoutError, match="close"):
            store.close(timeout=0.01)
        release.set()
        if native_failure:
            with pytest.raises(RuntimeError, match="native close failed"):
                store.close(timeout=1)
            assert store._registered_buffers and store.quarantined
        else:
            store.close(timeout=1)
            assert not store._registered_buffers and not store.quarantined
    finally:
        release.set()
        if native_failure:
            with pytest.raises(RuntimeError, match="native close failed"):
                store.close(timeout=2)
        else:
            store.close(timeout=2)


@pytest.fixture
def store_config(tmp_path):
    import mooncake

    port, metrics = free_port(), free_port()
    log = (tmp_path / "master.log").open("w")
    binary = Path(mooncake.__file__).parent / "mooncake_master"
    process = subprocess.Popen(
        [
            str(binary),
            f"--rpc_port={port}",
            f"--metrics_port={metrics}",
            "--enable_offload=false",
            "--enable_disk_eviction=false",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 20
        while True:
            if process.poll() is not None:
                raise RuntimeError((tmp_path / "master.log").read_text())
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError("Mooncake master did not start")
                time.sleep(0.1)
        yield {"host": "127.0.0.1", "master": f"127.0.0.1:{port}", "protocol": "tcp"}
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()


def test_chunked_tensors_round_trip_and_explicit_deletion(store_config):
    owner = TensorStore(store_config, pool_bytes=64 * 1024**2)
    writer = TensorStore(store_config)
    reader = TensorStore(store_config)
    tensors = {
        "input_ids": torch.arange(4096).reshape(1, -1),
        "loss_mask": torch.ones(1, 4096, dtype=torch.bool),
        "seq_len": torch.tensor([4096]),
        "context_chunk_len": torch.tensor([4096]),
        "target_hidden_states": torch.randn(1, 4096, 1280, dtype=torch.bfloat16),
        "target_last_hidden_states": torch.randn(1, 4096, 256, dtype=torch.bfloat16),
    }
    fields = describe_tensors("test/sample-0", tensors)
    assert len(fields["target_hidden_states"]["chunks"]) > 1
    try:
        writer.put(fields, tensors)
        result = reader.get(fields, FIELDS)
        for name in FIELDS:
            assert torch.equal(result[name], tensors[name])
        owner.remove(fields)
        key = fields["input_ids"]["chunks"][0]["key"]
        assert owner.client.is_exist(key) == 0
    finally:
        reader.close()
        writer.close()
        owner.close()


def test_consumer_pool_with_distinct_writer_and_reader_clients(store_config):
    from deepspec.pipeline.cluster import StoreProbe

    owner = TensorStore(store_config, pool_bytes=64 * 1024**2)
    writer = StoreProbe(store_config)
    reader = StoreProbe(store_config)
    try:
        assert owner.endpoint.startswith("127.0.0.1:")
        fields = writer.write("placement-probe")
        assert len(fields["target_hidden_states"]["chunks"]) > 1
        result = reader.read(fields)
        assert result["nbytes"] == sum(f["nbytes"] for f in fields.values())
        assert reader.store.last_read["verified"]
        assert writer.store.last_write["nbytes"] == result["nbytes"]
        reader.remove(fields)
    finally:
        reader.close()
        writer.close()
        owner.close()


def test_hard_pinned_features_survive_store_allocation_pressure(store_config):
    owner = TensorStore(store_config, pool_bytes=64 * 1024**2)
    client = TensorStore(store_config)
    payload = torch.ones(8 * 1024**2, dtype=torch.uint8)
    status = client.client.register_buffer(payload.data_ptr(), payload.numel())
    assert status == 0
    try:
        assert client.client.batch_put_from(
            ["protected"], [payload.data_ptr()], [payload.numel()], client.config
        ) == [0]
        failed = False
        for index in range(20):
            result = client.client.batch_put_from(
                [f"pressure-{index}"],
                [payload.data_ptr()],
                [payload.numel()],
                client.config,
            )
            if result != [0]:
                failed = True
                break
        assert failed, "The pressure probe must exhaust this bounded pool"
        time.sleep(1)  # Allow the master eviction loop to run after pressure.
        assert client.client.is_exist("protected") == 1
        payload.zero_()
        assert client.client.batch_get_into(
            ["protected"], [payload.data_ptr()], [payload.numel()]
        ) == [payload.numel()]
        assert bool(payload.eq(1).all())
    finally:
        client.client.unregister_buffer(payload.data_ptr())
        client.close()
        owner.close()


def test_ray_buffer_waits_for_all_readers_and_drains(store_config, tmp_path):
    import asyncio

    import ray

    from deepspec.pipeline.actors import Producer
    from deepspec.pipeline.buffer import FeatureBuffer

    class ProducerLoopProbe(Producer):
        def __init__(self):
            pass

        def run(self):
            return asyncio.run(asyncio.sleep(0, result="native-loop-entered"))

    tensors = {
        "input_ids": torch.arange(16).reshape(1, -1),
        "loss_mask": torch.ones(1, 16, dtype=torch.int64),
        "seq_len": torch.tensor([16]),
        "context_chunk_len": torch.tensor([16]),
        "target_hidden_states": torch.ones(1, 16, 40, dtype=torch.bfloat16),
        "target_last_hidden_states": torch.ones(1, 16, 8, dtype=torch.bfloat16),
    }
    size = sum(t.numel() * t.element_size() for t in tensors.values())
    samples = [
        {
            "position": i,
            "sample_id": str(i),
            "input_identity": str(i),
            "length": 16,
            "nbytes": size,
        }
        for i in range(4)
    ]
    config = {
        "samples": samples,
        "capacity_bytes": size * 4,
        "window": 4,
        "consumer_world_size": 4,
        "samples_per_update": 4,
        "store": store_config,
        "pool_bytes": 64 * 1024**2,
        "events_path": str(tmp_path / "events.jsonl"),
        "feature_memory_budget": 256 * 1024**2,
        "timeout_seconds": 10,
        "memory_reserve_bytes": 0,
        "scratch_bound_bytes": 0,
    }
    ray.init(
        address="local",
        num_cpus=2,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=128 * 1024**2,
        log_to_driver=False,
    )
    buffer = ray.remote(FeatureBuffer).remote(config)
    frontend = ray.remote(ProducerLoopProbe).remote()
    client = TensorStore(store_config)
    try:
        assert ray.get(frontend.run.remote(), timeout=30) == "native-loop-entered"
        ray.get(buffer.summary.remote(), timeout=30)
        for i, sample in enumerate(samples):
            ray.get(buffer.reserve.remote(i))
            ray.get(buffer.begin_write.remote(i))
            fields = describe_tensors(f"ray-test/{i}", tensors)
            client.put(fields, tensors)
            ray.get(buffer.publish.remote(i, {**sample, "fields": fields}))
            for reader in range(4):
                descriptor = ray.get(buffer.claim.remote(i, reader))
                actual = client.get(descriptor["fields"], FIELDS)
                assert torch.equal(
                    actual["target_hidden_states"], tensors["target_hidden_states"]
                )
                ray.get(buffer.acknowledge.remote(i, reader))
                assert client.client.is_exist(
                    fields["input_ids"]["chunks"][0]["key"]
                ) == (reader < 3)
        ray.get(buffer.finish_production.remote())
        summary = ray.get(buffer.summary.remote())
        assert summary["released"] == 4 and summary["remaining"] == 0
        ray.get(buffer.close.remote())
    finally:
        ray.kill(frontend)
        ray.kill(buffer)
        ray.shutdown()
        client.close()


def test_prefetch_bounds_storage_and_preserves_read_order(store_config):
    from deepspec.pipeline.prefetch import FeaturePrefetch

    owner = TensorStore(store_config, pool_bytes=64 * 1024**2)
    prefetch = FeaturePrefetch(store_config, depth=2, timeout=10)
    descriptors = []
    try:
        for position in range(3):
            tensors = {
                "input_ids": torch.arange(16).reshape(1, -1),
                "loss_mask": torch.ones(1, 16, dtype=torch.int64),
                "seq_len": torch.tensor([16]),
                "context_chunk_len": torch.tensor([16]),
                "target_hidden_states": torch.full(
                    (1, 16, 40), position, dtype=torch.bfloat16
                ),
                "target_last_hidden_states": torch.full(
                    (1, 16, 8), position, dtype=torch.bfloat16
                ),
            }
            fields = describe_tensors(f"prefetch/{position}", tensors)
            owner.put(fields, tensors)
            descriptors.append({"position": position, "fields": fields})
        prefetch.submit(descriptors[0])
        prefetch.submit(descriptors[1])
        with pytest.raises(RuntimeError, match="window is full"):
            prefetch.submit(descriptors[2])
        first = prefetch.take(0)
        owner.remove(descriptors[0]["fields"])
        prefetch.submit(descriptors[2])
        # Deleting the source does not invalidate a completed independent read.
        assert bool(first["target_hidden_states"].eq(0).all())
        for position in (1, 2):
            result = prefetch.take(position)
            assert bool(result["target_hidden_states"].eq(position).all())
            owner.remove(descriptors[position]["fields"])
        assert prefetch.peak_pending == 2
        assert not prefetch.pending
    finally:
        prefetch.close()
        owner.close()


@pytest.mark.parametrize("consumer_dp,producer_dp", [(1, 1), (2, 1), (2, 2)])
def test_native_loader_ranks_consume_two_complete_updates(
    store_config, tmp_path, consumer_dp, producer_dp
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    import ray
    from torchtitan.models.dspark_draft.planning import input_identity

    from deepspec.pipeline.buffer import FeatureBuffer

    teacher = {
        "target_layer_ids": [1, 3],
        "hidden_size": 64,
        "activation_dtype": "bfloat16",
        "target_final_hidden_source": "full_model_final_norm_output",
    }
    batches, samples = [], []
    for position in range(8):
        tensors = {
            "input_ids": torch.arange(16).reshape(1, -1),
            "loss_mask": torch.ones(1, 16, dtype=torch.int64),
            "seq_len": torch.tensor([16]),
            "context_chunk_len": torch.tensor([16]),
            "target_hidden_states": torch.full(
                (1, 16, 128), position + 1, dtype=torch.bfloat16
            ),
            "target_last_hidden_states": torch.full(
                (1, 16, 64), -position, dtype=torch.bfloat16
            ),
        }
        batches.append(tensors)
        samples.append(
            {
                "id": f"batch-{position}",
                "position": position,
                "sample_id": str(position),
                "input_identity": input_identity(tensors),
                "length": 16,
                "nbytes": sum(t.numel() * t.element_size() for t in tensors.values()),
            }
        )
    plan_path, manifest_path = tmp_path / "plan.json", tmp_path / "manifest.json"
    plan_path.write_text(
        json.dumps(
            {
                "version": 1,
                "run_id": "rank-probe",
                "batches": samples,
                "producer_requirements": teacher,
            }
        )
    )
    manifest_path.write_text(
        json.dumps({"batches": [{"id": s["id"]} for s in samples]})
    )
    config = {
        "run_id": "rank-probe",
        "namespace": "rank-probe",
        "buffer_name": "features",
        "samples": samples,
        "teacher": teacher,
        "capacity_bytes": samples[0]["nbytes"] * 4,
        "window": 4,
        "consumer_world_size": 4 * consumer_dp,
        "consumer_dp": consumer_dp,
        "producer_dp": producer_dp,
        "samples_per_update": 4,
        "store": store_config,
        "pool_bytes": 64 * 1024**2,
        "events_path": str(tmp_path / "events.jsonl"),
        "feature_memory_budget": 256 * 1024**2,
        "memory_reserve_bytes": 0,
        "scratch_bound_bytes": 0,
        "timeout_seconds": 120,
        "plan_path": str(plan_path),
        "manifest_path": str(manifest_path),
        "receive_device": "cpu",
        "result_prefix": str(tmp_path / "rank-result"),
    }
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(json.dumps(config))
    context = ray.init(
        address="local",
        namespace="rank-probe",
        num_cpus=2,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=128 * 1024**2,
        log_to_driver=False,
    )
    buffer = ray.remote(FeatureBuffer).options(name="features").remote(config)
    client = TensorStore(store_config)
    second_client = TensorStore(store_config) if producer_dp == 2 else None
    process = None
    try:
        ray.get(buffer.summary.remote(), timeout=30)
        with (tmp_path / "ranks.log").open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "deepspec.orchestration.process",
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc-per-node={4 * consumer_dp}",
                    "-m",
                    "tests.pipeline_rank_probe",
                ],
                env=dict(
                    os.environ,
                    RAY_ADDRESS=context.address_info["gcs_address"],
                    DEEPSPEC_PIPELINE_CONFIG=str(config_path),
                    CUDA_VISIBLE_DEVICES="",
                    DEEPSPEC_ORCHESTRATOR_PID=str(os.getpid()),
                    OMP_NUM_THREADS="1",
                ),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            odd_ready = [Event() for _ in range(4)]

            def write(position):
                sample, tensors = samples[position], batches[position]
                rank = position % producer_dp
                ray.get(buffer.begin_write.remote(position, rank))
                fields = describe_tensors(f"rank-probe/{position}", tensors)
                writer = client if rank == 0 else second_client
                writer.put(fields, tensors)
                if producer_dp == 2 and rank == 0:
                    assert odd_ready[position // 2].wait(30)
                ray.get(
                    buffer.publish.remote(position, {**sample, "fields": fields}, rank)
                )
                if rank == 1:
                    odd_ready[position // 2].set()

            # Each writer owns one client and a serial queue; only admission
            # remains global. DP1 is the existing synchronous reference.
            with ThreadPoolExecutor(1) as even, ThreadPoolExecutor(1) as odd:
                writes = []
                for position in range(len(samples)):
                    ray.get(buffer.reserve.remote(position), timeout=125)
                    if producer_dp == 1:
                        write(position)
                    else:
                        writes.append(
                            (even if position % 2 == 0 else odd).submit(write, position)
                        )
                for future in writes:
                    future.result(timeout=125)
            ray.get(buffer.finish_production.remote())
            assert process.wait(timeout=90) == 0, (tmp_path / "ranks.log").read_text()
        for rank in range(4 * consumer_dp):
            report = json.loads((tmp_path / f"rank-result-{rank}.json").read_text())
            assert report["positions"] == list(range(rank // 4, 8, consumer_dp))
            assert report["cursor"] == report["native_cursor"] == 8 // consumer_dp
            assert report["sample_cursor"] == 8 and report["steps"] == 2
            assert report["gas"] == 4 // consumer_dp and report["rank"] == rank
            assert report["run_id"] == "rank-probe" and report["node_id"]
        summary = ray.get(buffer.summary.remote())
        assert summary["remaining"] == 0 and summary["released"] == 8
        assert "backpressure" in (tmp_path / "events.jsonl").read_text()
        if producer_dp == 2:
            events = [
                json.loads(line)
                for line in (tmp_path / "events.jsonl").read_text().splitlines()
            ]
            ready = [e["position"] for e in events if e["event"] == "ready"]
            assert all(ready.index(p + 1) < ready.index(p) for p in range(0, 8, 2))
            assert all(
                e["producer_rank"] == e["position"] % 2
                for e in events
                if e["event"] == "ready"
            )
        ray.get(buffer.close.remote())
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        ray.kill(buffer)
        ray.shutdown()
        client.close()
        if second_client is not None:
            second_client.close()
