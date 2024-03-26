import functools
import os
from datetime import datetime
import sys
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
import json

profile_prefix = os.getenv("FL_PROFILE_PREFIX", "")

@dataclass
class Profiler:

    filename: str = ""
    log_size: bool = False
    store_per_run: bool = False
    profiles: ClassVar = {}
    path: ClassVar[str] = "/profile"
    has_executed: ClassVar[bool] = False

    @staticmethod
    def __actualsize(input_obj):
        memory_size = 0
        ids = set()
        objects = [input_obj]
        while objects:
            new = []
            for obj in objects:
                if id(obj) not in ids:
                    ids.add(id(obj))
                    memory_size += sys.getsizeof(obj)
                    new.append(obj)
            objects = gc.get_referents(*new)
        return memory_size

    def add_to_dict(self, name, elapsed_time, output_size):
        self.profiles[name] = elapsed_time
        if output_size is not None:
            self.profiles[f"{name}_size"] = output_size


    def __call__(self, func):
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            tic = datetime.now()
            value = func(*args, **kwargs)
            toc = datetime.now()
            elapsed_time = toc - tic
            output_size = None
            if self.log_size:
                output_size = Profiler.__actualsize(value)
            total_seconds = elapsed_time.total_seconds()
            Profiler.add_to_dict(func.__name__, total_seconds if total_seconds >= 0.0001 else 0.0, output_size)
            if self.store_per_run:
                Profiler.store_metrics()
            return value
        return wrapper_timer

    def store_metrics(self):
        cur_prefix = f"{profile_prefix}/" if profile_prefix != "" else ""
        file = f"{Profiler.path}/{cur_prefix}{self.filename}"
        self.profiles["timestamp"] = f"{datetime.now()}"
        if not os.path.exists(Profiler.path):
            os.makedirs(Profiler.path)
        if not os.path.exists(f"{Profiler.path}/{cur_prefix}"):
            os.makedirs(f"{Profiler.path}/{cur_prefix}")
        if not self.has_executed:
            [f.unlink() for f in Path(f"{Profiler.path}/{cur_prefix}").glob("*") if f.is_file()]
            self.has_executed = True
        if not os.path.exists(Profiler.path):
            os.makedirs(Profiler.path)
        with open(file, "a") as f:
            json.dumps(self.profiles)
            f.write(json.dumps(self.profiles) + "\n")
        self.profiles = {}

    def log_metric(self, metric: str, value: float):
        self.profiles[metric] = value