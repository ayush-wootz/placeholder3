"""Tool 4: schedule_task — schedules recurring prompts via APScheduler."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Awaitable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger


@dataclass
class ScheduledTask:
    task_id: str
    label: str
    prompt: str
    cron: str
    scheduled_time: str


class TaskScheduler:
    def __init__(self, on_trigger: Callable[[str, str], Awaitable[None]] | None = None):
        """
        Args:
            on_trigger: async callback(label, prompt) invoked when a job fires.
        """
        self._scheduler = AsyncIOScheduler()
        self._on_trigger = on_trigger
        self._tasks: dict[str, ScheduledTask] = {}

    def start(self) -> None:
        if not self._scheduler.running:
            self._scheduler.start()

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    def schedule(self, label: str, prompt: str, cron_schedule: str) -> ScheduledTask:
        task_id = uuid.uuid4().hex[:12]
        trigger = CronTrigger.from_crontab(cron_schedule)

        async def _job():
            if self._on_trigger:
                await self._on_trigger(label, prompt)

        self._scheduler.add_job(_job, trigger, id=task_id)

        task = ScheduledTask(
            task_id=task_id,
            label=label,
            prompt=prompt,
            cron=cron_schedule,
            scheduled_time=datetime.now().isoformat(),
        )
        self._tasks[task_id] = task
        return task

    def cancel(self, task_id: str) -> bool:
        if task_id in self._tasks:
            self._scheduler.remove_job(task_id)
            del self._tasks[task_id]
            return True
        return False

    def list_tasks(self) -> list[ScheduledTask]:
        return list(self._tasks.values())
