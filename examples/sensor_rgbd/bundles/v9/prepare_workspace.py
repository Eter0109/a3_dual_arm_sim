"""Wire an existing pinned simulator clone and optional shared model assets.

Linux server helper. Does not download, overwrite, copy model weights or train.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

COMMIT = '29fe90e423259f388ebc061926c3ec860b9edc74'


def write_new_or_identical(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'Refusing to overwrite a different existing file: {path}')
    else:
        path.write_bytes(data)


def link_existing(source, destination):
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source:
            raise RuntimeError(f'Refusing to replace {destination}')
    else:
        destination.symlink_to(source, target_is_directory=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--main-repo', type=Path, required=True)
    parser.add_argument('--checkpoint-source', type=Path)
    parser.add_argument('--tokenizer-source', type=Path)
    parser.add_argument('--verify-model-hash', action='store_true')
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    repo = args.main_repo.resolve(strict=True)
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if head != COMMIT:
        raise RuntimeError(f'Wrong simulator commit: {head}')
    if subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=repo,text=True).strip():
        raise RuntimeError('Tracked simulator modifications must be resolved before reproduction')
    if bool(args.checkpoint_source) != bool(args.tokenizer_source):
        raise ValueError('Provide both checkpoint and tokenizer, or neither for geometry-only evaluation')
    job = args.job.resolve()
    job.mkdir(parents=True, exist_ok=True)
    link_existing(repo, job/'benchmark_main_29fe90e')
    write_new_or_identical(repo/'SOURCE_COMMIT.txt', (COMMIT+'\n').encode())
    for relative in ('benchmark_research_v3','artifacts','repo'):
        for source in (bundle/'runtime_assets'/relative).rglob('*'):
            if source.is_file():
                write_new_or_identical(job/source.relative_to(bundle/'runtime_assets'),source.read_bytes())
    if args.checkpoint_source:
        checkpoint = args.checkpoint_source.resolve(strict=True)
        expected = json.loads((bundle/'server_environment_and_checkpoint.json').read_text())['checkpoint_files']
        for name, item in expected.items():
            path = checkpoint/name
            if not path.is_file() or path.stat().st_size != item['size_bytes']:
                raise RuntimeError(f'Missing/different original 30k checkpoint asset: {name}')
            if args.verify_model_hash or path.stat().st_size < 100_000_000:
                with path.open('rb') as stream:
                    actual=hashlib.file_digest(stream,'sha256').hexdigest()
                if actual != item['sha256']:
                    raise RuntimeError(f'Checkpoint checksum mismatch: {name}')
        link_existing(checkpoint,job/'training/checkpoints/030000/pretrained_model')
        tokenizer = args.tokenizer_source.resolve(strict=True)
        write_new_or_identical(job/'artifacts/tokenizer_path.txt',(str(tokenizer)+'\n').encode())
    environment = {
        'REPRO_JOB':str(job), 'MUJOCO_GL':'egl', 'PYOPENGL_PLATFORM':'egl',
        'CUBLAS_WORKSPACE_CONFIG':':4096:8', 'OMP_NUM_THREADS':'2',
        'MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','HF_HOME':str(job/'hf'),
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','A3_STATE_PROJECTOR':'1',
        'A3_UNFREEZE_LLM_LAYERS':'18','A3_UNFREEZE_VISION':'1'}
    write_new_or_identical(job/'repro_env.sh',
        ''.join(f'export {k}={shlex.quote(v)}\n' for k,v in environment.items()).encode())
    print(json.dumps(dict(job=str(job),simulator_commit=head,
                         weights_linked=bool(args.checkpoint_source),training_started=False),indent=2))


if __name__=='__main__':
    main()
