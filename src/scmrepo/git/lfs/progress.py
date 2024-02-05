from typing import BinaryIO, Callable, Optional, Union

from dvc_objects.fs.callbacks import TqdmCallback
from fsspec.callbacks import DEFAULT_CALLBACK, Callback

from scmrepo.progress import GitProgressEvent


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
                bytes=True, desc=path_1 if isinstance(path_1, str) else path_2
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
