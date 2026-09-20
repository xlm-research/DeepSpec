"""Bounded admission and all-reader acknowledgements for feature objects."""

import asyncio
import json
import time
from pathlib import Path

from .mooncake import DeleteManager


class BufferLedger:
    def __init__(
        self,
        samples,
        *,
        capacity,
        window,
        readers,
        samples_per_update,
        readers_by_position=None,
        producer_dp=1,
        approved_producer_dp=None,
    ):
        if window < samples_per_update:
            raise ValueError("Buffer window must hold one complete optimizer update")
        if any(s["nbytes"] > capacity for s in samples):
            raise ValueError("One sample exceeds the feature capacity")
        for start in range(0, len(samples), samples_per_update):
            if (
                sum(s["nbytes"] for s in samples[start : start + samples_per_update])
                > capacity
            ):
                raise ValueError("Capacity cannot complete one optimizer update")
        self.samples = samples
        self.capacity = capacity
        self.window = window
        self.readers = set(readers)
        self.readers_by_position = (
            [self.readers] * len(samples)
            if readers_by_position is None
            else [set(group) for group in readers_by_position]
        )
        if len(self.readers_by_position) != len(samples) or any(
            not group or not group.issubset(self.readers)
            for group in self.readers_by_position
        ):
            raise ValueError("Every sample requires a valid group of feature readers")
        self.samples_per_update = samples_per_update
        if (
            type(producer_dp) is not int
            or producer_dp < 1
            or (producer_dp not in (1, 2) and approved_producer_dp != producer_dp)
        ):
            raise ValueError(
                "Producer DP must be positive and approved by the topology plan"
            )
        self.producer_dp = producer_dp
        self.records = {}
        self.history = {}
        self.reserved_bytes = 0
        self.peak_bytes = 0
        self.resident_bytes = 0
        self.peak_resident_bytes = 0
        self.next_position = 0
        self.released = 0
        self.high_water = False

    def reserve(self, position):
        if position != self.next_position:
            raise ValueError("Production must be admitted in input-plan order")
        sample = self.samples[position]
        if self.high_water and position % self.samples_per_update == 0:
            if self.reserved_bytes > self.capacity * 0.6:
                return False
            self.high_water = False
        if (
            self.reserved_bytes + sample["nbytes"] > self.capacity
            or len(self.records) >= self.window
        ):
            self.high_water = True
            return False
        self.records[position] = {"state": "reserved", "claimed": set(), "acked": set()}
        self.reserved_bytes += sample["nbytes"]
        self.peak_bytes = max(self.peak_bytes, self.reserved_bytes)
        self.next_position += 1
        return True

    def start_write(self, position, producer_rank=0):
        if producer_rank != position % self.producer_dp:
            raise ValueError("Unexpected feature producer")
        record = self.records.get(position)
        if record is None or record["state"] != "reserved":
            raise ValueError("Duplicate or unreserved feature write")
        record.update(state="writing", producer_rank=producer_rank)

    def ready(self, position, descriptor, producer_rank=0):
        if producer_rank != position % self.producer_dp:
            raise ValueError("Unexpected feature producer")
        record = self.records[position]
        expected = self.samples[position]
        if record["state"] != "writing":
            raise ValueError("Duplicate or unreserved feature publication")
        for key in ("position", "sample_id", "input_identity", "length"):
            if descriptor[key] != expected[key]:
                raise ValueError(f"Published {key} differs from the input plan")
        actual = sum(spec["nbytes"] for spec in descriptor["fields"].values())
        if actual != expected["nbytes"]:
            raise ValueError("Feature bytes differ from the production reservation")
        if "fields" in expected:
            if set(descriptor["fields"]) != set(expected["fields"]):
                raise ValueError("Feature fields differ from the input plan")
            for name, field in expected["fields"].items():
                observed = descriptor["fields"][name]
                if any(observed.get(key) != field[key] for key in ("shape", "dtype")):
                    raise ValueError(
                        f"Published {name} shape/dtype differs from the input plan"
                    )
        record.update(state="ready", descriptor=descriptor)
        self.resident_bytes += actual
        self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes)

    def claim(self, position, reader):
        if reader not in self.readers_by_position[position]:
            raise ValueError("Unexpected feature reader")
        record = self.records.get(position)
        if record is None or record["state"] in ("reserved", "writing"):
            return None
        if reader in record["claimed"]:
            raise ValueError("Duplicate feature claim")
        record["claimed"].add(reader)
        return record["descriptor"]

    def acknowledge(self, position, reader):
        record = self.records[position]
        if reader not in record["claimed"] or reader in record["acked"]:
            raise ValueError("Duplicate or unclaimed feature acknowledgement")
        record["acked"].add(reader)
        if record["acked"] == self.readers_by_position[position]:
            record["state"] = "deleting"
            return record["descriptor"]["fields"]
        return None

    def deleted(self, position):
        if self.records[position]["state"] != "deleting":
            raise ValueError("Cannot release capacity before all readers complete")
        self.history[position] = {**self.records.pop(position), "state": "released"}
        self.reserved_bytes -= self.samples[position]["nbytes"]
        self.resident_bytes -= self.samples[position]["nbytes"]
        self.released += 1


class FeatureBuffer:
    """Ray actor; only descriptors enter Ray, while Store owns feature memory."""

    def __init__(self, config, node_monitors=None, *, defer_store=False):
        from .schema import normalize_pipeline_config
        from .topology import producer_dp, sample_readers

        normalize_pipeline_config(config)
        self.config = config
        self.node_monitors = node_monitors or []
        self.runtime_budget = None
        if "node_budgets" in config:
            from .memory import BudgetAdmission

            policy = config["timeouts_seconds"]
            self.runtime_budget = BudgetAdmission(
                config["node_budgets"],
                run_id=config["run_id"],
                plan_hash=config["plan_hash"],
                freshness=policy["budget_snapshot"],
                transfer_timeout=policy["transfer"],
                run_deadline=time.monotonic()
                + config.get("remaining_run_seconds", policy["run"]),
            )
        self.ledger = BufferLedger(
            config["samples"],
            capacity=config["capacity_bytes"],
            window=config["window"],
            readers=range(config["consumer_world_size"]),
            samples_per_update=config["samples_per_update"],
            readers_by_position=[
                sample_readers(config, sample["position"])
                for sample in config["samples"]
            ],
            producer_dp=producer_dp(config),
            approved_producer_dp=config.get("approved_inference_dp"),
        )
        self.store = None
        self._initialization = None
        self._closing = self._closed = False
        self.delete_manager = DeleteManager(
            lambda fields: self.store.remove(fields),
            max_workers=int(config.get("delete_workers", 1)),
            max_attempts=int(config.get("delete_max_attempts", 5)),
            retry_delay=float(config.get("delete_retry_delay", 0.1)),
            timeout=config["timeout_seconds"],
        )
        self.changed = asyncio.Condition()
        self.error = None
        self.producer_finished = False
        self.consumer_ready = False
        self.reader_copies = {}
        self.retention_released = not config.get("retain_until_bytes", 0)
        self.events = Path(config["events_path"]).open("a", buffering=1)  # noqa: SIM115 -- actor lifetime; closed in close()
        self.structured_events = None
        self.production_started = None
        if config.get("topology_plan_path"):
            from .runtime import EventWriter

            self.structured_events = EventWriter(
                Path(config["output_dir"]) / "events/feature-buffer.jsonl",
                run_id=config["run_id"],
                plan_hash=config["plan_hash"],
                sender_identity={"component": "feature_buffer"},
            )
        if not defer_store:
            self._open_store()

    def _open_store(self):
        from .store import TensorStore

        config = self.config
        self.store = TensorStore(config["store"], pool_bytes=config["pool_bytes"])
        self._event(
            "buffer_started",
            pool_bytes=config["pool_bytes"],
            feature_memory_budget=config.get("feature_memory_budget"),
            node_budgets=config.get("node_budgets"),
            store_endpoint=self.store.endpoint,
            store_node_id=config.get("consumer_node_id"),
        )

    async def start(self):
        if self._closing:
            raise RuntimeError("Feature buffer is closed")
        if self.store is None and self._initialization is None:
            self._initialization = asyncio.create_task(
                asyncio.to_thread(self._open_store)
            )
        return {"started": True}

    def identity(self):
        from .runtime import actor_identity

        return actor_identity(self.config)

    def ready(self):
        return self.store is not None and not self._closing

    def service_status(self):
        if self._initialization is not None and self._initialization.done():
            self._initialization.result()
        return {
            "ready": self.ready(),
            "endpoint": None if self.store is None else self.store.endpoint,
            "pool_bytes": self.config["pool_bytes"],
            "closing": self._closing,
        }

    def _event(self, event, **fields):
        self.events.write(
            json.dumps(
                {
                    "event": event,
                    "time": time.time(),
                    "monotonic": time.monotonic(),
                    **fields,
                }
            )
            + "\n"
        )

    def _check(self):
        if self.error:
            raise RuntimeError(self.error)

    async def event(self, event, **fields):
        if event == "inference_start" and self.production_started is None:
            self.production_started = time.monotonic()
        self._event(event, **fields)

    async def fail(self, message):
        async with self.changed:
            self.error = self.error or message
            self._event("failed", message=message)
            self.changed.notify_all()

    async def wait_for_failure(self):
        async with self.changed:
            await asyncio.wait_for(
                self.changed.wait_for(lambda: self.error is not None),
                self.config.get("timeouts_seconds", {}).get(
                    "run", self.config["timeout_seconds"]
                ),
            )
            self._check()

    async def consumer_initialized(self):
        async with self.changed:
            self.consumer_ready = True
            self._event("consumer_initialized")
            self.changed.notify_all()

    async def wait_for_consumer(self):
        async with self.changed:
            await asyncio.wait_for(
                self.changed.wait_for(lambda: self.consumer_ready or self.error),
                self.config["timeout_seconds"],
            )
            self._check()

    async def reserve(self, position):
        await self.reserve_batch(position, 1)

    async def _sample_admission(self, position, timeout):
        from .memory import node_memory

        if self.runtime_budget is not None:
            update_index = position // self.config["samples_per_update"]
            request = self.runtime_budget.request(update_index)
            responses = await asyncio.wait_for(
                asyncio.gather(
                    *(m.sample_budget.remote(request) for m in self.node_monitors)
                ),
                timeout=timeout,
            )
            return {"responses": responses, "update_index": update_index}
        memory = node_memory()
        pressure = memory["headroom_bytes"] < (
            self.config["memory_reserve_bytes"]
            + self.config["scratch_bound_bytes"]
            + self.config.get("transport_staging_bound_bytes", 0)
        )
        nodes = []
        if self.node_monitors:
            nodes = await asyncio.wait_for(
                asyncio.gather(*(m.memory.remote() for m in self.node_monitors)),
                timeout=timeout,
            )
            pressure = any(item["pressure"] for item in nodes)
        return {
            "pressure": pressure,
            "headroom_bytes": memory["headroom_bytes"],
            "nodes": nodes,
        }

    async def reserve_batch(self, position, count):
        if type(count) is not int or count < 1:
            raise ValueError("Reservation count must be a positive integer")
        started = time.monotonic()
        deadline = started + self.config["timeout_seconds"]
        admitted = []
        try:
            while position + len(admitted) < min(
                position + count, len(self.ledger.samples)
            ):
                following = position + len(admitted)
                self._check()
                boundary = following % self.config["samples_per_update"] == 0
                sample = None
                if boundary:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Production admission timed out")
                    # Remote sampling never holds the condition lock: fail,
                    # ACK and stop remain responsive even if a node stalls.
                    sample = await self._sample_admission(following, min(remaining, 30))
                async with self.changed:
                    self._check()
                    pressure = False
                    if sample is not None:
                        if self.runtime_budget is not None:
                            pressure = not self.runtime_budget.check(
                                sample["update_index"], sample["responses"]
                            )
                            if self.structured_events is not None:
                                for response in sample["responses"]:
                                    memory = response["memory"]
                                    node = response["node_id"]
                                    self.structured_events.emit(
                                        "budget_sample",
                                        {
                                            "node_id": node,
                                            "approved_bytes": self.config[
                                                "node_budgets"
                                            ][node]["feature_bound"],
                                            "observed_bytes": max(
                                                0,
                                                memory["limit_bytes"]
                                                - memory["headroom_bytes"],
                                            ),
                                            "observed_scope": "node_memory_usage_including_other_processes",
                                            "headroom_bytes": memory["headroom_bytes"],
                                            "request_id": response["request_id"],
                                            "update_index": sample["update_index"],
                                        },
                                        basis="observed",
                                    )
                        else:
                            pressure = sample["pressure"]
                    if not pressure and self.ledger.reserve(following):
                        if boundary and self.runtime_budget is not None:
                            self.runtime_budget.commit(sample["update_index"])
                        admitted.append(following)
                        self._event(
                            "reserved",
                            position=following,
                            reserved_bytes=self.ledger.reserved_bytes,
                            waited_seconds=time.monotonic() - started
                            if len(admitted) == 1
                            else 0,
                        )
                        continue
                    self._event(
                        "backpressure",
                        position=following,
                        reserved_bytes=self.ledger.reserved_bytes,
                        memory_pressure=pressure,
                        reason=self.runtime_budget.reason
                        if self.runtime_budget
                        else "legacy_capacity",
                        budget=self.runtime_budget.last_observations
                        if self.runtime_budget
                        else sample,
                    )
                    # Already reserved samples must be returned to the writer;
                    # waiting for the rest would withhold their production.
                    if admitted:
                        return admitted
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Production admission timed out")
                    try:
                        waited = time.monotonic()
                        await asyncio.wait_for(self.changed.wait(), min(1, remaining))
                    except TimeoutError:
                        pass
                    finally:
                        if self.structured_events is not None:
                            self.structured_events.emit(
                                "wait",
                                {
                                    "reason": self.runtime_budget.reason
                                    if pressure and self.runtime_budget
                                    else "source_capacity",
                                    "duration_seconds": time.monotonic() - waited,
                                },
                                basis="observed",
                            )
            return admitted
        except Exception as error:
            await self.fail(f"Production admission failed: {error}")
            raise

    async def begin_write(self, position, producer_rank=0, node_id=None):
        async with self.changed:
            self._check()
            nodes = self.config.get("producer_node_ids")
            if nodes is not None and (
                not 0 <= producer_rank < len(nodes) or node_id != nodes[producer_rank]
            ):
                raise ValueError("Feature writer is on the wrong producer node")
            self.ledger.start_write(position, producer_rank)
            self._event("write_started", position=position, producer_rank=producer_rank)

    async def publish(self, position, descriptor, producer_rank=0, write_metrics=None):
        async with self.changed:
            self._check()
            self.ledger.ready(position, descriptor, producer_rank)
            if self.structured_events is not None:
                duration = None if write_metrics is None else write_metrics["seconds"]
                self.structured_events.emit(
                    "feature_produced",
                    {
                        "position": position,
                        "replica_id": producer_rank,
                        "nbytes": self.ledger.samples[position]["nbytes"],
                        "tokens": self.ledger.samples[position]["length"],
                        "duration_seconds": duration,
                        "duration_scope": "writer_conversion_and_store_put",
                    },
                    basis="observed",
                    missing={"duration_seconds": "writer duration unavailable"}
                    if duration is None
                    else None,
                )
            self._event(
                "ready",
                position=position,
                producer_rank=producer_rank,
                nbytes=self.ledger.samples[position]["nbytes"],
                resident_bytes=self.ledger.resident_bytes,
            )
            if (
                not self.retention_released
                and self.ledger.resident_bytes >= self.config["retain_until_bytes"]
            ):
                from .memory import node_memory

                rss = next(
                    line
                    for line in Path("/proc/self/status").read_text().splitlines()
                    if line.startswith("VmRSS:")
                )
                self.retention_released = True
                self._event(
                    "pool_peak_reached",
                    resident_bytes=self.ledger.resident_bytes,
                    pool_bytes=self.config["pool_bytes"],
                    owner_rss_bytes=int(rss.split()[1]) * 1024,
                    memory=node_memory(),
                )
                for retained, record in list(self.ledger.records.items()):
                    if record["state"] == "deleting":
                        await self._delete(retained, record["descriptor"]["fields"])
            if write_metrics is not None:
                self._event(
                    "write_complete",
                    position=position,
                    producer_rank=producer_rank,
                    **write_metrics,
                )
            self.changed.notify_all()

    async def claim(self, position, reader):
        from .runtime import Deadline

        deadline = Deadline.after(self.config["timeout_seconds"])
        async with self.changed:
            while True:
                self._check()
                descriptor = self.ledger.claim(position, reader)
                if descriptor is not None:
                    self.reader_copies[position, reader] = {
                        "state": "prefetching",
                        "nbytes": self.ledger.samples[position]["nbytes"],
                    }
                    self._event("claimed", position=position, reader=reader)
                    return descriptor
                await asyncio.wait_for(self.changed.wait(), deadline.remaining())

    async def acknowledge(
        self, position, reader, *, verified=None, nbytes=None, duration_seconds=None
    ):
        async with self.changed:
            self._check()
            if self.config.get("plan_hash") and (
                verified is not True
                or nbytes != self.ledger.samples[position]["nbytes"]
            ):
                raise ValueError(
                    "Reader ACK requires a fully verified independent copy"
                )
            fields = self.ledger.acknowledge(position, reader)
            if self.structured_events is not None:
                self.structured_events.emit(
                    "feature_read",
                    {
                        "position": position,
                        "reader_rank": reader,
                        "nbytes": nbytes,
                        "verified": verified,
                        "duration_seconds": duration_seconds,
                        "duration_scope": "materialization_wait",
                    },
                    basis="verified",
                    missing={"duration_seconds": "reader duration unavailable"}
                    if duration_seconds is None
                    else None,
                )
            self.reader_copies[position, reader]["state"] = "materialized_and_acked"
            self._event("received", position=position, reader=reader)
            if fields is not None and self.retention_released:
                await self._delete(position, fields)
            self.changed.notify_all()

    async def reader_copy_state(self, position, reader, state):
        async with self.changed:
            self._check()
            copy = self.reader_copies[position, reader]
            expected = {"active": "materialized_and_acked", "retired": "active"}
            if state not in expected or copy["state"] != expected[state]:
                raise ValueError("Invalid reader copy lifetime transition")
            copy["state"] = state
            self._event(
                "reader_copy_" + state,
                position=position,
                reader=reader,
                nbytes=copy["nbytes"],
            )

    async def _delete(self, position, fields):
        future = self.delete_manager.submit(fields)
        try:
            await asyncio.wait_for(
                asyncio.shield(asyncio.wrap_future(future)),
                self.config["timeout_seconds"],
            )
        except BaseException as error:
            self.error = self.error or f"Feature deletion {position} failed: {error}"
            self._event("failed", message=self.error, position=position)
            self.changed.notify_all()
            raise
        self.ledger.deleted(position)
        self._event(
            "released",
            position=position,
            reserved_bytes=self.ledger.reserved_bytes,
            resident_bytes=self.ledger.resident_bytes,
        )

    async def finish_production(self):
        from .runtime import Deadline

        deadline = Deadline.after(self.config["timeout_seconds"])
        async with self.changed:
            self._check()
            if self.ledger.next_position != len(self.ledger.samples):
                raise ValueError(
                    "Production finished before the full plan was admitted"
                )
            while any(
                r["state"] in ("reserved", "writing")
                for r in self.ledger.records.values()
            ):
                self._check()
                await asyncio.wait_for(self.changed.wait(), deadline.remaining())
            self._check()
            if not self.retention_released:
                raise RuntimeError(
                    "Production ended before the requested resident peak"
                )
            self.producer_finished = True
            if (
                self.structured_events is not None
                and self.production_started is not None
            ):
                self.structured_events.emit(
                    "phase_duration",
                    {
                        "phase": "production",
                        "duration_seconds": time.monotonic() - self.production_started,
                    },
                    basis="observed",
                )
            self._event("producer_finished")

    async def summary(self):
        self._check()
        return {
            "produced": self.ledger.next_position,
            "released": self.ledger.released,
            "remaining": len(self.ledger.records),
            "peak_reserved_bytes": self.ledger.peak_bytes,
            "peak_resident_bytes": self.ledger.peak_resident_bytes,
            "resident_bytes": self.ledger.resident_bytes,
            "retention_released": self.retention_released,
            "pool_bytes": self.config["pool_bytes"],
            "producer_finished": self.producer_finished,
            "store_endpoint": self.store.endpoint,
            "store_node_id": self.config.get("consumer_node_id"),
        }

    async def source_snapshot(self):
        from .runtime import message_envelope

        async with self.changed:
            records = {**self.ledger.history, **self.ledger.records}
            return message_envelope(
                self.config["run_id"],
                self.config["plan_hash"],
                {"component": "feature_buffer"},
                resident_bytes=self.ledger.resident_bytes,
                reserved_bytes=self.ledger.reserved_bytes,
                records=[
                    {
                        "position": position,
                        "state": record["state"],
                        "acked": sorted(record["acked"]),
                        "delete_confirmed": position in self.ledger.history,
                    }
                    for position, record in sorted(records.items())
                ],
            )

    async def close(self, *, timeout=None):
        from .runtime import Deadline

        deadline = Deadline.after(
            timeout
            if timeout is not None
            else self.config.get("timeouts_seconds", {}).get(
                "cleanup", self.config["timeout_seconds"]
            )
        )
        self._closing = True
        if self._closed:
            return {"cleanup_complete": True}
        if self._initialization is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._initialization), deadline.remaining()
                )
            except Exception:
                # A finished initializer may have allocated the Store before
                # failing. Preserve its failure, but still close that Store.
                # A pending/cancelled task does not prove the native call ended.
                if not self._initialization.done() or self._initialization.cancelled():
                    raise
        await asyncio.wait_for(
            asyncio.to_thread(self.delete_manager.close, timeout=deadline.remaining()),
            deadline.remaining(),
        )
        if self.store is not None:
            await asyncio.wait_for(
                asyncio.to_thread(self.store.close, timeout=deadline.remaining()),
                deadline.remaining(),
            )
        self.events.close()
        if self.structured_events is not None:
            self.structured_events.close()
        self._closed = True
        return {"cleanup_complete": True}
