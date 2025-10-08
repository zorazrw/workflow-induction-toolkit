from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Callable

from .models import Observation, init_db
from .observers import Observer
from .schemas import Update

class crec:
    def __init__(
        self,
        user_name: str,
        *observers: Observer,
        data_directory: str = "~/Downloads/records",
        db_name: str = "actions.db",
        max_concurrent_updates: int = 4,
        verbosity: int = logging.INFO,
    ):
        # basic paths
        data_directory = os.path.expanduser(data_directory)
        os.makedirs(data_directory, exist_ok=True)

        # runtime
        self.user_name = user_name
        self.observers: list[Observer] = list(observers)

        # logging
        self.logger = logging.getLogger("crec")
        self.logger.setLevel(verbosity)
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(h)

        self.engine = None
        self.Session = None
        self._db_name        = db_name
        self._data_directory = data_directory

        self._update_sem = asyncio.Semaphore(max_concurrent_updates)
        self._tasks: set[asyncio.Task] = set()
        self._loop_task: asyncio.Task | None = None
        self.update_handlers: list[Callable[[Observer, Update], None]] = []

    def start_update_loop(self):
        if self._loop_task is None:
            self._loop_task = asyncio.create_task(self._update_loop())

    async def stop_update_loop(self):
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

    async def connect_db(self):
        if self.engine is None:
            self.engine, self.Session = await init_db(
                self._db_name, self._data_directory
            )

    async def __aenter__(self):
        await self.connect_db()
        self.start_update_loop()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop_update_loop()

        # wait for any in-flight handlers
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # stop observers
        for obs in self.observers:
            await obs.stop()

    async def _update_loop(self):
        """
        Efficiently wait for *any* observer to produce an Update and
        dispatch it through the semaphore-guarded handler.
        """
        while True:

            gets = {
                asyncio.create_task(obs.update_queue.get()): obs
                for obs in self.observers
            }

            done, _ = await asyncio.wait(
                gets.keys(), return_when=asyncio.FIRST_COMPLETED
            )

            for fut in done:
                upd: Update = fut.result()
                obs = gets[fut]

                t = asyncio.create_task(self._run_with_gate(obs, upd))
                self._tasks.add(t)

    async def _run_with_gate(self, observer: Observer, update: Update):
        """Wrapper that enforces max_concurrent_updates."""
        async with self._update_sem:
            try:
                await self._default_handler(observer, update)
            finally:
                self._tasks.discard(asyncio.current_task())

    async def _handle_audit(self, obs: Observation) -> bool:
        return False

    async def _default_handler(self, observer: Observer, update: Update) -> None:
        self.logger.info(f"Processing update from {observer.name}")
        self.logger.info(f"Content ({update.content_type}): {update.content[:10]}")

        async with self._session() as session:
            observation = Observation(
                observer_name=observer.name,
                content=update.content,
                content_type=update.content_type,
            )

            if await self._handle_audit(observation):
                return

            session.add(observation)
            await session.flush()

    @asynccontextmanager
    async def _session(self):
        async with self.Session() as s:
            async with s.begin():
                yield s

    def add_observer(self, observer: Observer):
        self.observers.append(observer)

    def remove_observer(self, observer: Observer):
        if observer in self.observers:
            self.observers.remove(observer)

    def register_update_handler(self, fn: Callable[[Observer, Update], None]):
        self.update_handlers.append(fn)
