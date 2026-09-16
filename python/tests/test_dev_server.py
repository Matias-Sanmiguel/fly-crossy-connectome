from pathlib import Path

from fly_crossy.dev_server import CHECKPOINT_PATH, controller


ROOT = Path(__file__).resolve().parents[2]


def test_dev_server_uses_committed_release_checkpoint() -> None:
    expected = ROOT / "release/eval-v1/training/connectome/checkpoint.pt"

    assert CHECKPOINT_PATH == expected
    assert CHECKPOINT_PATH.is_file()
    assert controller.node_count == 80
