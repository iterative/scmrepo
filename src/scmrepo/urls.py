import re

# from https://github.com/pypa/pip/blob/303fed36c1771de4063063a866776a9103972317/src/pip/_internal/vcs/git.py#L40
# SCP (Secure copy protocol) shorthand. e.g. 'git@example.com:foo/bar.git'
SCP_REGEX = re.compile(
    r"""^
    # Optional user, e.g. 'git@'
    (\w+@)?
    # Server, e.g. 'github.com'.
    ([^/:]+):
    # The server-side path. e.g. 'user/project.git'. Must start with an
    # alphanumeric character so as not to be confusable with a Windows paths
    # like 'C:/foo/bar' or 'C:\foo\bar'.
    (\w[^:]*)
    $""",
    re.VERBOSE,
)


def is_scp_style_url(url: str) -> bool:
    return bool(SCP_REGEX.match(url))
