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
        if producer_dp not in (1, 2):
            raise ValueError("Producer DP must be one or two")
        self.producer_dp = producer_dp
        self.records = {}
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
        del self.records[position]
        self.reserved_bytes -= self.samples[position]["nbytes"]
        self.resident_bytes -= self.samples[position]["nbytes"]
        self.released += 1


class FeatureBuffer:
    """Ray actor; only descriptors enter Ray, while Store owns feature memory."""

    def __init__(self, config, node_monitors=None):
        from .store import TensorStore
        from .topology import producer_dp, sample_readers
        from .schema import normalize_pipeline_config

        normalize_pipeline_config(config)
        self.config = config
        self.node_monitors = node_monitors or []
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
        )
        self.store = TensorStore(config["store"], pool_bytes=config["pool_bytes"])
        self.delete_manager = DeleteManager(
            lambda fields: self.store.remove(fields),
            max_workers=int(config.get("delete_workers", 1)),
            max_attempts=int(config.get("delete_max_attempts", 5)),
            retry_delay=float(config.get("delete_retry_delay", 0.1)),
        )
        self.changed = asyncio.Condition()
        self.error = None
        self.producer_finished = False
        self.consumer_ready = False
        self.retention_released = not config.get("retain_until_bytes", 0)
        self.events = Path(config["events_path"]).open("a", buffering=1)  # noqa: SIM115 -- actor lifetime; closed in close()
        self._event(
            "buffer_started",
            pool_bytes=config["pool_bytes"],
            feature_memory_budget=config["feature_memory_budget"],
            store_endpoint=self.store.endpoint,
            store_node_id=config.get("consumer_node_id"),
        )

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
        self._event(event, **fields)

    async def fail(self, message):
        async with self.changed:
            self.error = self.error or message
            self._event("failed", message=message)
            self.changed.notify_all()

    async def wait_for_failure(self):
        async with self.changed:
            await self.changed.wait_for(lambda: self.error is not None)
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

    async def reserve_batch(self, position, count):
        from .memory import node_memory

        started = time.monotonic()
        async with self.changed:
            while True:
                self._check()
                memory = node_memory()
                # Finish an admitted optimizer group so ranks can always make
                # progress. The whole group's scratch is covered by the bound.
                pressure = (
                    position % self.config["samples_per_update"] == 0
                    and memory["headroom_bytes"]
                    < self.config["memory_reserve_bytes"]
                    + self.config["scratch_bound_bytes"]
                    + self.config.get("transport_staging_bound_bytes", 0)
                )
                node_memory_status = []
                if (
                    self.node_monitors
                    and position % self.config["samples_per_update"] == 0
                ):
                    node_memory_status = await asyncio.wait_for(
                        asyncio.gather(
                            *(m.memory.remote() for m in self.node_monitors)
                        ),
                        timeout=30,
                    )
                    pressure = any(item["pressure"] for item in node_memory_status)
                if not pressure and self.ledger.reserve(position):
                    break
                self._event(
                    "backpressure",
                    position=position,
                    reserved_bytes=self.ledger.reserved_bytes,
                    memory_pressure=pressure,
                    headroom_bytes=memory["headroom_bytes"],
                    nodes=node_memory_status,
                )
                if time.monotonic() - started >= self.config["timeout_seconds"]:
                    raise TimeoutError("Production admission timed out")
                try:
                    await asyncio.wait_for(self.changed.wait(), 1.0)
                except TimeoutError:
                    pass
            self._event(
                "reserved",
                position=position,
                reserved_bytes=self.ledger.reserved_bytes,
                waited_seconds=time.monotonic() - started,
            )
            admitted = [position]
            # Return any available prefix immediately. Waiting for a whole batch
            # here can deadlock: its first reservations have not been generated.
            for following in range(
                position + 1, min(position + count, len(self.ledger.samples))
            ):
                if not self.ledger.reserve(following):
                    break
                admitted.append(following)
                self._event(
                    "reserved",
                    position=following,
                    reserved_bytes=self.ledger.reserved_bytes,
                    waited_seconds=0,
                )
            return admitted

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
        async with self.changed:
            while True:
                self._check()
                descriptor = self.ledger.claim(position, reader)
                if descriptor is not None:
                    self._event("claimed", position=position, reader=reader)
                    return descriptor
                await asyncio.wait_for(
                    self.changed.wait(), self.config["timeout_seconds"]
                )

    async def acknowledge(self, position, reader):
        async with self.changed:
            self._check()
            fields = self.ledger.acknowledge(position, reader)
            self._event("received", position=position, reader=reader)
            if fields is not None and self.retention_released:
                await self._delete(position, fields)
            self.changed.notify_all()

    async def _delete(self, position, fields):
        future = self.delete_manager.submit(fields)
        try:
            await asyncio.wrap_future(future)
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
                await asyncio.wait_for(
                    self.changed.wait(), self.config["timeout_seconds"]
                )
            self._check()
            if not self.retention_released:
                raise RuntimeError(
                    "Production ended before the requested resident peak"
                )
            self.producer_finished = True
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

    async def close(self):
        await asyncio.to_thread(
            self.delete_manager.close, timeout=self.config.get("timeout_seconds")
        )
        await asyncio.to_thread(self.store.close)
        self.events.close()
