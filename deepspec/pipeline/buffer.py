"""Bounded admission and all-reader acknowledgements for feature objects."""

import asyncio
import json
import time
from pathlib import Path


class BufferLedger:
    def __init__(self, samples, *, capacity, window, readers, samples_per_update):
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
        self.samples_per_update = samples_per_update
        self.records = {}
        self.reserved_bytes = 0
        self.peak_bytes = 0
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

    def ready(self, position, descriptor):
        record = self.records[position]
        expected = self.samples[position]
        if record["state"] != "reserved":
            raise ValueError("Duplicate or unreserved feature publication")
        for key in ("position", "sample_id", "input_identity", "length"):
            if descriptor[key] != expected[key]:
                raise ValueError(f"Published {key} differs from the input plan")
        actual = sum(spec["nbytes"] for spec in descriptor["fields"].values())
        if actual != expected["nbytes"]:
            raise ValueError("Feature bytes differ from the production reservation")
        record.update(state="ready", descriptor=descriptor)

    def claim(self, position, reader):
        if reader not in self.readers:
            raise ValueError("Unexpected feature reader")
        record = self.records.get(position)
        if record is None or record["state"] == "reserved":
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
        if record["acked"] == self.readers:
            record["state"] = "deleting"
            return record["descriptor"]["fields"]
        return None

    def deleted(self, position):
        if self.records[position]["state"] != "deleting":
            raise ValueError("Cannot release capacity before all readers complete")
        del self.records[position]
        self.reserved_bytes -= self.samples[position]["nbytes"]
        self.released += 1


class FeatureBuffer:
    """Ray actor; only descriptors enter Ray, while Store owns feature memory."""

    def __init__(self, config):
        from .store import TensorStore

        self.config = config
        self.ledger = BufferLedger(
            config["samples"],
            capacity=config["capacity_bytes"],
            window=config["window"],
            readers=range(config["consumer_world_size"]),
            samples_per_update=config["samples_per_update"],
        )
        self.store = TensorStore(config["store"], pool_bytes=config["pool_bytes"])
        self.changed = asyncio.Condition()
        self.error = None
        self.producer_finished = False
        self.consumer_ready = False
        self.events = Path(config["events_path"]).open("a", buffering=1)  # noqa: SIM115 -- actor lifetime; closed in close()
        self._event(
            "buffer_started",
            pool_bytes=config["pool_bytes"],
            feature_memory_budget=config["feature_memory_budget"],
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
                )
                if not pressure and self.ledger.reserve(position):
                    break
                self._event(
                    "backpressure",
                    position=position,
                    reserved_bytes=self.ledger.reserved_bytes,
                    memory_pressure=pressure,
                    headroom_bytes=memory["headroom_bytes"],
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

    async def publish(self, position, descriptor):
        async with self.changed:
            self._check()
            self.ledger.ready(position, descriptor)
            self._event(
                "ready",
                position=position,
                nbytes=self.ledger.samples[position]["nbytes"],
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
            if fields is not None:
                self.store.remove(fields)
                self.ledger.deleted(position)
                self._event(
                    "released",
                    position=position,
                    reserved_bytes=self.ledger.reserved_bytes,
                )
                self.changed.notify_all()

    async def finish_production(self):
        async with self.changed:
            while any(r["state"] == "reserved" for r in self.ledger.records.values()):
                self._check()
                await asyncio.wait_for(
                    self.changed.wait(), self.config["timeout_seconds"]
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
            "pool_bytes": self.config["pool_bytes"],
            "producer_finished": self.producer_finished,
        }

    async def close(self):
        self.store.close()
        self.events.close()
