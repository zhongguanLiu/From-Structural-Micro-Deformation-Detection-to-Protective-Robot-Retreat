import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import experiment_source_snapshot as snapshot


class SourceSnapshotTest(unittest.TestCase):
    def test_bounded_immutable_archive_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'src/pkg').mkdir(parents=True)
            (root / 'src/pkg/node.py').write_text('print(1)')
            (root / 'src/pkg/private.env').write_text('must not archive')
            (root / 'src/pkg/link.py').symlink_to(root / 'src/pkg/private.env')
            (root / 'devel/lib').mkdir(parents=True)
            (root / 'devel/lib/node').write_bytes(b'binary')
            with patch.object(snapshot, 'PACKAGES', ('pkg',)), patch.object(snapshot, 'BINARIES', ('node',)):
                snapshot.snapshot(root, root / 'run', ['gui:=false'])
                manifest = json.loads((root / 'run/provenance/execution_versions.json').read_text())
                archive = root / 'run/provenance/execution_sources.zip'
                self.assertEqual(manifest['archive_sha256'], hashlib.sha256(archive.read_bytes()).hexdigest())
                with zipfile.ZipFile(archive) as saved:
                    self.assertEqual(saved.namelist(), ['pkg/node.py'])
                    self.assertEqual(manifest['source_sha256']['pkg/node.py'], hashlib.sha256(saved.read('pkg/node.py')).hexdigest())
                with self.assertRaises(FileExistsError):
                    snapshot.snapshot(root, root / 'run', [])
