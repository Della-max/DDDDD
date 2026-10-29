"""Acquire only the author's verified Dental test resources; no previous project inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ARCHIVE_URL = 'https://drive.usercontent.google.com/download?id=1Xq-pptfYVsve2wsMJxSgMdyLALxD4AXW&export=download&confirm=t'
ARCHIVE_SHA256 = '23ed01a4664e193ea158d194eda7cd8c9e3c3cbfa17152221e580751d3c0cc97'
CHECKPOINT_SHA256 = 'e3585572100e61c5f3112a8925e7add420b70c8c7a8f23e29be97788ad3c5643'
TEST_SHA256 = 'dffdd191803ddfeb7701d9012b1f38dd88f01ee9333747a19ba86e18870848e0'
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def install(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(source) != sha256(target):
            raise FileExistsError(target)
        return
    with Path(source).open('rb') as src, target.open('xb') as dst:
        shutil.copyfileobj(src, dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='The supplied official dental.pth; exact known hash required')
    args = parser.parse_args()
    if sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('Checkpoint is not the verified public Dental release')
    folder = ROOT / '.downloads'
    folder.mkdir(exist_ok=True)
    archive_path = folder / 'official_gfpp.zip'
    if not archive_path.exists():
        partial = folder / 'official_gfpp.zip.part'
        if partial.exists():
            raise FileExistsError('An incomplete download already exists; inspect it first')
        request = urllib.request.Request(ARCHIVE_URL, headers={'User-Agent': 'GFPackPP-reproduction'})
        with urllib.request.urlopen(request, timeout=60) as response, partial.open('xb') as stream:
            shutil.copyfileobj(response, stream)
        if sha256(partial) != ARCHIVE_SHA256:
            raise ValueError('Official archive digest changed; do not silently accept a different release')
        partial.rename(archive_path)
    if sha256(archive_path) != ARCHIVE_SHA256:
        raise ValueError('Archive failed identity verification')
    files = {}
    destination = ROOT / 'data' / 'official_dental'
    targets = {'data/test/dental_test.pkl': destination / 'dental_test.pkl'}
    targets.update({f'data/polygons/dental/{i}.txt': destination / 'polygons' / f'{i}.txt'
                    for i in range(440)})
    with zipfile.ZipFile(archive_path) as archive:
        for member, target in targets.items():
            payload = archive.read(member)  # CRC checked; names are a fixed whitelist.
            expected = hashlib.sha256(payload).hexdigest()
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if sha256(target) != expected:
                    raise FileExistsError(target)
            else:
                target.write_bytes(payload)
            files[target.relative_to(ROOT).as_posix()] = expected
    if sha256(destination / 'dental_test.pkl') != TEST_SHA256:
        raise ValueError('Official test set hash mismatch')
    install(args.checkpoint, ROOT / 'checkpoints' / 'dental.pth')
    files['checkpoints/dental.pth'] = CHECKPOINT_SHA256
    manifest = {'archive_url': ARCHIVE_URL, 'archive_sha256': ARCHIVE_SHA256,
                'checkpoint_source': 'user-supplied file, identical to archive data/ckpt/dental.pth',
                'files': files}
    manifest_path = destination / 'manifest.json'
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    elif json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
        raise FileExistsError(manifest_path)
    print('Verified official checkpoint, test PKL and 440 polygon files.', flush=True)


if __name__ == '__main__':
    main()
