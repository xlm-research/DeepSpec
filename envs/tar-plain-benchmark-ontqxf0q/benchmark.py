import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import time

report = Path(__file__).resolve().parent
archive = report.parent / 'deepspec_vllm_torchtitan_envs.tar'
expected = json.loads((report / 'source-manifest.json').read_text())
run_dir = Path(tempfile.mkdtemp(prefix='deepspec-plain-tar-benchmark-', dir='/tmp'))
local_archive = run_dir / archive.name
extract_root = run_dir / 'extracted'
extract_root.mkdir()
environment = extract_root / 'deepspec_vllm_torchtitan_envs'
archive_before = archive.stat()
print(f'LOCAL_ARCHIVE={local_archive}\nEXTRACTED_ENV={environment}', flush=True)

copy_command = ['cp', '-a', '--', str(archive), str(local_archive)]
extract_command = ['tar', '-xpf', str(local_archive), '-C', str(extract_root)]
start = time.perf_counter()
subprocess.run(copy_command, check=True)
copy_seconds = time.perf_counter() - start
print(f'ARCHIVE_COPY_SECONDS={copy_seconds:.6f}', flush=True)
start = time.perf_counter()
subprocess.run(extract_command, check=True)
extract_seconds = time.perf_counter() - start
print(f'EXTRACT_SECONDS={extract_seconds:.6f}', flush=True)

archive_after = archive.stat()
assert (archive_before.st_size, archive_before.st_mtime_ns) == (
    archive_after.st_size, archive_after.st_mtime_ns
), 'Shared archive changed during the benchmark'
assert local_archive.stat().st_size == archive_before.st_size
assert environment.is_dir(), 'Expected environment directory was not extracted'
actual = {}


def scan(directory):
    with os.scandir(directory) as entries:
        for entry in entries:
            info = entry.stat(follow_symlinks=False)
            relative = os.path.relpath(entry.path, environment)
            kind = stat.S_IFMT(info.st_mode)
            value = (info.st_size if stat.S_ISREG(info.st_mode) else
                     os.readlink(entry.path) if stat.S_ISLNK(info.st_mode) else None)
            actual[relative] = [kind, value]
            if stat.S_ISDIR(info.st_mode):
                scan(entry.path)


scan(environment)
missing = sorted(expected.keys() - actual.keys())
extra = sorted(actual.keys() - expected.keys())
different = sorted(key for key in expected.keys() & actual.keys()
                   if expected[key] != actual[key])
results = {
    'format': 'tar',
    'shared_archive': str(archive),
    'local_archive': str(local_archive),
    'extracted_environment': str(environment),
    'archive_bytes': archive_before.st_size,
    'regular_files': sum(value[0] == stat.S_IFREG for value in actual.values()),
    'regular_bytes': sum(value[1] for value in actual.values() if value[0] == stat.S_IFREG),
    'copy_command': copy_command,
    'extract_command': extract_command,
    'copy_seconds': copy_seconds,
    'extract_seconds': extract_seconds,
    'total_seconds': copy_seconds + extract_seconds,
    'archive_creation_included': False,
    'validation_included': False,
    'cache_state': 'uncontrolled; shared archive was recently created',
    'historical_cp_a_seconds': 521.257,
    'historical_rsync_a_seconds': 508.492,
    'validation': {
        'matches_source_paths_types_sizes_and_symlink_targets': not (missing or extra or different),
        'missing_count': len(missing), 'extra_count': len(extra),
        'different_count': len(different),
        'missing_examples': missing[:20], 'extra_examples': extra[:20],
        'different_examples': different[:20],
    },
}
(report / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2), flush=True)
if missing or extra or different:
    raise SystemExit('Extracted environment does not match the source manifest')
