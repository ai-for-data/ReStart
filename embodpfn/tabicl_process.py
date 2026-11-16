"""Keep robot dependencies separate from TabICLv2, without another model API."""
import os
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np


class TabICLProcessPredictor:
    def __init__(self, *, python, checkpoint, device, kv_cache=False, execution='native',
                 cuda_visible_devices=None, tabicl_source=None):
        self.python = Path(python).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.device = device
        if kv_cache not in (False, True, 'repr') or execution not in ('native', 'shared', 'shared-batched', 'pr87'):
            raise ValueError('Declare the PFN execution and original cache mode explicitly.')
        if execution.startswith('shared') and kv_cache is False:
            raise ValueError('Shared execution requires independent native per-output caches.')
        if execution == 'pr87' and kv_cache is not False:
            raise ValueError('Original PR87 multi-target execution requires cache=False.')
        self.kv_cache = {False: 'none', True: 'kv', 'repr': 'repr'}[kv_cache]
        self.execution = execution
        if not self.python.is_file() or not self.checkpoint.is_file():
            raise FileNotFoundError('Specify the existing PFN environment and official regression checkpoint.')
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        injected = env.pop('EMBODPFN_PICA_CUBLAS_LIBRARY_DIRECTORY', None)
        if injected is not None:
            try:
                binding = json.loads(injected)
            except (TypeError, ValueError) as error:
                raise RuntimeError('Invalid PICA server library-binding marker.') from error
            if (not isinstance(binding, dict)
                    or set(binding) != {'directory', 'prior_is_set'}
                    or not isinstance(binding['directory'], str) or not binding['directory']
                    or not isinstance(binding['prior_is_set'], bool)
                    or 'LD_LIBRARY_PATH' not in env):
                raise RuntimeError('Invalid PICA server library-binding environment.')
            paths = env['LD_LIBRARY_PATH'].split(os.pathsep)
            try:
                paths.remove(binding['directory'])
            except ValueError as error:
                raise RuntimeError('The declared PICA server library directory is absent from the PFN child environment.') from error
            if not paths and not binding['prior_is_set']:
                env.pop('LD_LIBRARY_PATH')
            else:
                env['LD_LIBRARY_PATH'] = os.pathsep.join(paths)
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        if cuda_visible_devices is not None:
            # Native XPolicyLab isolates its policy GPU. The separate PFN
            # process may explicitly use another allocated physical GPU;
            # this changes no policy visibility or estimator implementation.
            parts = str(cuda_visible_devices).split(',')
            if not parts or any(not part.isdigit() for part in parts):
                raise ValueError('Declare explicit allocated physical PFN GPU indices.')
            env['CUDA_VISIBLE_DEVICES'] = ','.join(parts)
        command = [str(self.python), '-m', 'embodpfn.tabicl_worker',
             '--checkpoint', str(self.checkpoint), '--device', self.device,
             '--kv-cache', self.kv_cache, '--execution', self.execution]
        if tabicl_source is not None:
            command += ['--tabicl-source', str(Path(tabicl_source).resolve())]
        self.worker = subprocess.Popen(command,
            cwd=root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1)
        self.has_context = False

    def _request(self, message):
        if self.worker.poll() is not None:
            raise RuntimeError('The PFN worker exited; the adapted run cannot continue.')
        self.worker.stdin.write(json.dumps(message) + '\n')
        self.worker.stdin.flush()
        line = self.worker.stdout.readline()
        if not line:
            # Passive failure-only metadata. Preserve the original EOF error,
            # protocol, return paths, operators and number of model calls.
            try:
                diagnostic = {
                    'pfn_worker_stdout_EOF': True, 'parent_pid': os.getpid(),
                    'worker_pid': self.worker.pid,
                    'worker_poll_returncode': self.worker.poll(),
                    'command': message.get('command'), 'execution': self.execution,
                    'device': self.device, 'cgroup_scope': 'this parent process allocated container',
                    'cgroup_memory': {}, 'no_extra_model_or_GPU_operation': True}
                for filename in (
                    '/proc/self/cgroup',
                    '/sys/fs/cgroup/memory.events',
                    '/sys/fs/cgroup/memory.events.local',
                    '/sys/fs/cgroup/memory.current',
                    '/sys/fs/cgroup/memory.peak',
                    '/sys/fs/cgroup/memory.max',
                    '/sys/fs/cgroup/memory/memory.oom_control',
                    '/sys/fs/cgroup/memory/memory.failcnt',
                    '/sys/fs/cgroup/memory/memory.usage_in_bytes',
                    '/sys/fs/cgroup/memory/memory.max_usage_in_bytes',
                    '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
                    try:
                        diagnostic['cgroup_memory'][filename] = {'value': Path(filename).read_text()[:4096]}
                    except OSError as read_error:
                        diagnostic['cgroup_memory'][filename] = {'read_error_type': type(read_error).__name__}
                os.write(2, (json.dumps(diagnostic) + '\n').encode())
            except Exception as diagnostic_error:
                try:
                    os.write(2, (json.dumps({'pfn_EOF_diagnostic_error_type': type(diagnostic_error).__name__}) + '\n').encode())
                except OSError:
                    pass
            raise RuntimeError('The PFN worker returned no result; no baseline fallback is used.')
        result = json.loads(line)
        if result['status'] != 'ok':
            raise RuntimeError('PFN inference failed: ' + result['error_type'] + ': ' + result['message'])
        return result

    def update_context(self, conditions, inverted_noise):
        conditions = np.asarray(conditions)
        inverted_noise = np.asarray(inverted_noise)
        if conditions.ndim != 2 or inverted_noise.ndim < 2 or len(conditions) != len(inverted_noise):
            raise ValueError('Support input/target shapes do not match.')
        if len(conditions) < 2 or not np.isfinite(conditions).all() or not np.isfinite(inverted_noise).all():
            raise ValueError('Need at least two finite support rows.')
        self.has_context = False
        with TemporaryDirectory(prefix='embodpfn-context-') as directory:
            path = Path(directory) / 'support.npz'
            np.savez(path, conditions=conditions, noise=inverted_noise)
            self._request({'command': 'context', 'support': str(path)})
        self.has_context = True
        return self

    def predict_initial_noise(self, conditions):
        if not self.has_context:
            raise RuntimeError('Provide the explicit support context first.')
        with TemporaryDirectory(prefix='embodpfn-tabicl-') as directory:
            directory = Path(directory)
            np.save(directory / 'query.npy', np.asarray(conditions), allow_pickle=False)
            self._request({'command': 'predict', 'query': str(directory / 'query.npy'),
                           'output': str(directory / 'prediction.npy')})
            return np.load(directory / 'prediction.npy', allow_pickle=False)

    def close(self):
        if self.worker.poll() is None:
            try:
                self._request({'command': 'close'})
            finally:
                self.worker.stdin.close()
                try:
                    self.worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.worker.terminate()
                    self.worker.wait()
        self.worker.stdout.close()
