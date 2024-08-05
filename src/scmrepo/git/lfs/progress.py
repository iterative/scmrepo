import logging
import sys
from typing import Any, BinaryIO, Callable, ClassVar, Optional, Union

from fsspec.callbacks import DEFAULT_CALLBACK, Callback, TqdmCallback
from tqdm import tqdm

from scmrepo.progress import GitProgressEvent


class _Tqdm(tqdm):
    """
    maximum-compatibility tqdm-based progressbars
    """

    BAR_FMT_DEFAULT = (
        "{percentage:3.0f}% {desc}|{bar}|"
        "{postfix[info]}{n_fmt}/{total_fmt}"
        " [{elapsed}<{remaining}, {rate_fmt:>11}]"
    )
    # nested bars should have fixed bar widths to align nicely
    BAR_FMT_DEFAULT_NESTED = (
        "{percentage:3.0f}%|{bar:10}|{desc:{ncols_desc}.{ncols_desc}}"
        "{postfix[info]}{n_fmt}/{total_fmt}"
        " [{elapsed}<{remaining}, {rate_fmt:>11}]"
    )
    BAR_FMT_NOTOTAL = "{desc}{bar:b}|{postfix[info]}{n_fmt} [{elapsed}, {rate_fmt:>11}]"
    BYTES_DEFAULTS: ClassVar[dict[str, Any]] = {
        "unit": "B",
        "unit_scale": True,
        "unit_divisor": 1024,
        "miniters": 1,
    }

    def __init__(
        self,
        iterable=None,
        disable=None,
        level=logging.ERROR,
        desc=None,
        leave=False,
        bar_format=None,
        bytes=False,  # noqa: A002
        file=None,
        total=None,
        postfix=None,
        **kwargs,
    ):
        kwargs = kwargs.copy()
        if bytes:
            kwargs = {**self.BYTES_DEFAULTS, **kwargs}
        else:
            kwargs.setdefault("unit_scale", total > 999 if total else True)
        if file is None:
            file = sys.stderr
        super().__init__(
            iterable=iterable,
            disable=disable,
            leave=leave,
            desc=desc,
            bar_format="!",
            lock_args=(False,),
            total=total,
            **kwargs,
        )
        self.postfix = postfix or {"info": ""}
        if bar_format is None:
            if self.__len__():
                self.bar_format = (
                    self.BAR_FMT_DEFAULT_NESTED if self.pos else self.BAR_FMT_DEFAULT
                )
            else:
                self.bar_format = self.BAR_FMT_NOTOTAL
        else:
            self.bar_format = bar_format
        self.refresh()

    def update_to(self, current, total=None):
        if total:
            self.total = total
        self.update(current - self.n)

    def close(self):
        self.postfix["info"] = ""
        # remove ETA (either unknown or zero); remove completed bar
        self.bar_format = self.bar_format.replace("<{remaining}", "").replace(
            "|{bar:10}|", " "
        )
        super().close()

    @property
    def format_dict(self):
        """inject `ncols_desc` to fill the display width (`ncols`)"""
        d = super().format_dict
        ncols = d["ncols"] or 80
        # assumes `bar_format` has max one of ("ncols_desc" & "ncols_info")

        meter = self.format_meter(  # type: ignore[call-arg]
            ncols_desc=1, ncols_info=1, **d
        )
        ncols_left = ncols - len(meter) + 1
        ncols_left = max(ncols_left, 0)
        if ncols_left:
            d["ncols_desc"] = d["ncols_info"] = ncols_left
        else:
            # work-around for zero-width description
            d["ncols_desc"] = d["ncols_info"] = 1
            d["prefix"] = ""
        return d


class LFSCallback(Callback):
    """Callback subclass to generate Git/LFS style progress."""

    def __init__(
        self,
        *args,
        git_progress: Optional[Callable[[GitProgressEvent], None]] = None,
        direction: str = "Downloading",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.direction = direction
        self.git_progress = git_progress

    def call(self, *args, **kwargs):
        super().call(*args, **kwargs)
        self._update_git()

    def _update_git(self):
        if not self.git_progress:
            return
        event = GitProgressEvent(
            phase=f"{self.direction} LFS objects",
            completed=self.value,
            total=self.size,
        )
        self.git_progress(event)

    def branched(self, path_1: Union[str, BinaryIO], path_2: str, **kwargs):
        if self.git_progress:
            return TqdmCallback(
                tqdm_kwargs={
                    "desc": path_1 if isinstance(path_1, str) else path_2,
                    "bytes": True,
                },
                tqdm_cls=_Tqdm,
            )
        return DEFAULT_CALLBACK

    @classmethod
    def as_lfs_callback(
        cls,
        git_progress: Optional[Callable[[GitProgressEvent], None]] = None,
        **kwargs,
    ):
        if git_progress is None:
            return DEFAULT_CALLBACK
        return cls(git_progress=git_progress, **kwargs)
