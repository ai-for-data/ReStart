"""Fail closed when an adapter is pointed at a modified or different upstream."""
import inspect
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LIBERO_RELEASES = {
    'original': ('chenpyyy/UniWAM-libero', '538c5ed32bc62bfb79e346a873963fa7e52b8960',
                 '973fc9be40f6aa4db3688b97c74a73ed22eb16867007411584b8aab50827fbb9'),
    'plus': ('chenpyyy/UniWAM-libero-plus', 'daf0fe8ccc93b2ade3a468cd47c47f2d018c789b',
             '63a40bf213b84e7d64fd78535a2808810a0c2996ef3be692fcabbb4c411344e7')}


def checkpoint_asset(policy, checkpoint_receipt=None, *, libero_release=None):
    asset = json.loads((ROOT / 'benchmarks/policy-assets.json').read_text())[policy]
    receipt_path = Path(checkpoint_receipt) if checkpoint_receipt is not None else ROOT / asset['checkpoint_directory'] / 'embodpfn-asset-receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if libero_release is not None:
        repo, revision, checksum = LIBERO_RELEASES[libero_release]
        original = receipt['official_release']
        if (policy != 'uniwam' or receipt.get('kind') != 'official-uniwam-libero-release'
                or original['repository'] != repo or original['revision'] != revision
                or original['weight_sha256'] != checksum
                or receipt['verified_weight_sha256'] != checksum
                or not original['all_original_release_files_verified']):
            raise RuntimeError('Require the exact verified original author LIBERO release.')
        asset.update(weight_sha256=checksum, weight_file=original['weight_file'],
                     checkpoint_stage=original['checkpoint_stage'], official_release=original)
    elif receipt['verified_weight_sha256'] != asset['weight_sha256']:
        raise RuntimeError('The method requires the declared general pretrained checkpoint.')
    asset['runtime_checkpoint_file'] = receipt.get('cloud_path', str(ROOT / asset['checkpoint_directory'] / asset['weight_file']))
    return asset


def require_declared_checkpoint(policy, checkpoint_path, checkpoint_receipt=None, *, libero_release=None):
    asset = checkpoint_asset(policy, checkpoint_receipt,
                             libero_release=libero_release)
    actual = Path(checkpoint_path)
    if actual.is_dir():
        actual = actual / asset['weight_file']
    if actual.resolve(strict=True) != Path(asset['runtime_checkpoint_file']).resolve(strict=True):
        raise ValueError('Pass the exact declared and verified checkpoint file.')


def verify_upstream(policy, model, *, model_source=None, deployed_subtree=None, checkpoint_receipt=None,
                    libero_release=None):
    asset = checkpoint_asset(policy, checkpoint_receipt,
                             libero_release=libero_release)
    source = ROOT / asset['source_directory']
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
    if revision != asset['source_revision'] or dirty:
        raise RuntimeError('The pinned upstream source/configuration must remain pristine.')
    implementation = Path(inspect.getfile(type(model))).resolve()
    if model_source is None:
        if not implementation.is_relative_to(source.resolve()):
            raise RuntimeError('Expected the real policy class from the pinned upstream checkout.')
    elif implementation != (source / model_source).resolve():
        if deployed_subtree is None:
            raise RuntimeError('Select the backend for the actual upstream runtime.')
        relative = Path(model_source).relative_to(deployed_subtree)
        deployed = implementation.parents[len(relative.parts)-1]
        tracked = subprocess.check_output(['git', '-C', str(source), 'ls-files', '--', deployed_subtree], text=True).splitlines()
        if not tracked:
            raise RuntimeError('The official deployed policy subtree is empty.')
        for name in tracked:
            original = source / name
            copied = deployed / Path(name).relative_to(deployed_subtree)
            if (not copied.is_file() or hashlib.sha256(copied.read_bytes()).digest() !=
                    hashlib.sha256(original.read_bytes()).digest()):
                raise RuntimeError('Official deployment copy differs from the pinned policy: ' + name)
    if model.training:
        raise RuntimeError('The official loader must put the frozen policy in eval mode.')
    return asset
