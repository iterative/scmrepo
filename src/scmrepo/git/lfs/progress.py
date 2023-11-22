from typing import Any, BinaryIO, Callable, Dict, Optional, Union

from dvc_objects.fs.callbacks import DEFAULT_CALLBACK, Callback, TqdmCallback

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

    def branch(
        self,
        path_1: Union[str, BinaryIO],
        path_2: str,
        kwargs: Dict[str, Any],
        child: Optional[Callback] = None,
    ):
        if child:
            pass
        elif self.git_progress:
            child = TqdmCallback(
                bytes=True, desc=path_1 if isinstance(path_1, str) else path_2
            )
        else:
            child = DEFAULT_CALLBACK
        return super().branch(path_1, path_2, kwargs, child=child)

    @classmethod
    def as_lfs_callback(
        cls,
        git_progress: Optional[Callable[[GitProgressEvent], None]] = None,
        **kwargs,
    ):
        if git_progress is None:
            return DEFAULT_CALLBACK
        return cls(git_progress=git_progress, **kwargs)
