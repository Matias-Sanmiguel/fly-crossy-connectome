from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fly_crossy.biomechanics import body as body_module
from fly_crossy.biomechanics.body import FlyBodyModel
import mujoco


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPOSITORY_ROOT / "data" / "manifests" / "flybody-v1.json"

EXPECTED_ACTION_NAMES = (
    # Position-controlled head (3).
    "c_thorax-c_head-yaw-position",
    "c_thorax-c_head-roll-position",
    "c_thorax-c_head-pitch-position",
    # Position-controlled active leg DoFs (42; seven per leg).
    "c_thorax-lf_coxa-yaw-position",
    "c_thorax-lf_coxa-roll-position",
    "c_thorax-lf_coxa-pitch-position",
    "lf_coxa-lf_trochanterfemur-roll-position",
    "lf_coxa-lf_trochanterfemur-pitch-position",
    "lf_trochanterfemur-lf_tibia-pitch-position",
    "lf_tibia-lf_tarsus1-pitch-position",
    "c_thorax-lm_coxa-yaw-position",
    "c_thorax-lm_coxa-roll-position",
    "c_thorax-lm_coxa-pitch-position",
    "lm_coxa-lm_trochanterfemur-roll-position",
    "lm_coxa-lm_trochanterfemur-pitch-position",
    "lm_trochanterfemur-lm_tibia-pitch-position",
    "lm_tibia-lm_tarsus1-pitch-position",
    "c_thorax-lh_coxa-yaw-position",
    "c_thorax-lh_coxa-roll-position",
    "c_thorax-lh_coxa-pitch-position",
    "lh_coxa-lh_trochanterfemur-roll-position",
    "lh_coxa-lh_trochanterfemur-pitch-position",
    "lh_trochanterfemur-lh_tibia-pitch-position",
    "lh_tibia-lh_tarsus1-pitch-position",
    "c_thorax-rf_coxa-yaw-position",
    "c_thorax-rf_coxa-roll-position",
    "c_thorax-rf_coxa-pitch-position",
    "rf_coxa-rf_trochanterfemur-roll-position",
    "rf_coxa-rf_trochanterfemur-pitch-position",
    "rf_trochanterfemur-rf_tibia-pitch-position",
    "rf_tibia-rf_tarsus1-pitch-position",
    "c_thorax-rm_coxa-yaw-position",
    "c_thorax-rm_coxa-roll-position",
    "c_thorax-rm_coxa-pitch-position",
    "rm_coxa-rm_trochanterfemur-roll-position",
    "rm_coxa-rm_trochanterfemur-pitch-position",
    "rm_trochanterfemur-rm_tibia-pitch-position",
    "rm_tibia-rm_tarsus1-pitch-position",
    "c_thorax-rh_coxa-yaw-position",
    "c_thorax-rh_coxa-roll-position",
    "c_thorax-rh_coxa-pitch-position",
    "rh_coxa-rh_trochanterfemur-roll-position",
    "rh_coxa-rh_trochanterfemur-pitch-position",
    "rh_trochanterfemur-rh_tibia-pitch-position",
    "rh_tibia-rh_tarsus1-pitch-position",
    # Coupled abdomen (2) and tarsal (6) tendon controls.
    "abdomen_pitch-tendon",
    "abdomen_yaw-tendon",
    "lf_tarsus-tendon",
    "lm_tarsus-tendon",
    "lh_tarsus-tendon",
    "rf_tarsus-tendon",
    "rm_tarsus-tendon",
    "rh_tarsus-tendon",
    # One claw-adhesion control per leg (6); labrum adhesion is not exposed.
    "lf_tarsus5-adhesion",
    "lm_tarsus5-adhesion",
    "lh_tarsus5-adhesion",
    "rf_tarsus5-adhesion",
    "rm_tarsus5-adhesion",
    "rh_tarsus5-adhesion",
)


def _inventory_digest(files: list[dict[str, object]]) -> str:
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _synthetic_asset_manifest(asset_root: Path) -> dict[str, object]:
    (asset_root / "README").write_bytes(b"pinned fixture\n")
    (asset_root / "mesh.obj").write_bytes(b"v 0 0 0\n")
    files = []
    for path in sorted(asset_root.iterdir()):
        content = path.read_bytes()
        files.append(
            {
                "path": path.name,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return {
        "prefix": "flybody_fullsize_meshes_20260623a",
        "fileCount": len(files),
        "totalBytes": sum(int(item["size"]) for item in files),
        "inventorySha256": _inventory_digest(files),
        "files": files,
    }


@pytest.fixture(scope="session")
def body() -> FlyBodyModel:
    return FlyBodyModel.load(MANIFEST_PATH)


def test_body_compiles_the_exact_59_signal_flybody_contract(
    body: FlyBodyModel,
) -> None:
    assert isinstance(body.model, mujoco.MjModel)
    assert body.action_size == 59
    assert body.model.nu == 59
    assert body.joint_names == EXPECTED_ACTION_NAMES
    assert body.action_groups == {
        "position": (0, 45),
        "tendon": (45, 53),
        "adhesion": (53, 59),
    }
    assert body.actuator_ids == tuple(range(59))


def test_body_resolves_stable_leg_site_names(body: FlyBodyModel) -> None:
    assert tuple(body.leg_sites) == (
        "front_left",
        "front_right",
        "middle_left",
        "middle_right",
        "hind_left",
        "hind_right",
    )
    assert len(set(body.leg_sites.values())) == 6
    assert all(0 <= site_id < body.model.nsite for site_id in body.leg_sites.values())


def test_body_requires_exactly_59_finite_actions(body: FlyBodyModel) -> None:
    validated = body.validate_action(np.zeros(59))

    assert validated.shape == (59,)
    with pytest.raises(ValueError, match="59"):
        body.validate_action(np.zeros(58))
    with pytest.raises(ValueError, match="finite"):
        body.validate_action(np.full(59, np.nan))


@pytest.mark.parametrize("bound", ("lower", "upper"))
def test_body_clamps_nothing_silently(body: FlyBodyModel, bound: str) -> None:
    action = (body.lower_limits + body.upper_limits) / 2
    if bound == "lower":
        action[0] = body.lower_limits[0] - 0.01
    else:
        action[0] = body.upper_limits[0] + 0.01

    with pytest.raises(ValueError, match="joint limit"):
        body.validate_action(action)


def test_validated_actions_do_not_alias_caller_memory(body: FlyBodyModel) -> None:
    action = (body.lower_limits + body.upper_limits) / 2
    validated = body.validate_action(action)
    action[0] = body.upper_limits[0]

    assert validated[0] != action[0]
    assert validated.flags.writeable is False


@pytest.mark.parametrize("checksum", (None, "not-a-sha256", "0" * 63))
def test_loader_rejects_a_missing_or_malformed_model_checksum_before_building(
    tmp_path: Path, checksum: str | None
) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    if checksum is None:
        del manifest["modelSha256"]
    else:
        manifest["modelSha256"] = checksum
    invalid_path = tmp_path / "flybody.json"
    invalid_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="modelSha256"):
        FlyBodyModel.load(invalid_path)


def test_loader_rejects_a_model_source_hash_mismatch(tmp_path: Path) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest["modelSha256"] = "0" * 64
    invalid_path = tmp_path / "flybody.json"
    invalid_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="checksum"):
        FlyBodyModel.load(invalid_path)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("source", "https://example.invalid/flybody"),
        ("integrationSource", "https://example.invalid/flygym"),
        ("upstreamCommit", "0" * 40),
        ("licenseFile", "LICENSE"),
        ("noticeFile", "NOTICE"),
    ),
)
def test_loader_rejects_changed_immutable_provenance(
    tmp_path: Path, field: str, replacement: str
) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest[field] = replacement
    invalid_path = tmp_path / "flybody.json"
    invalid_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="immutable manifest field"):
        FlyBodyModel.load(invalid_path)


def test_manifest_pins_the_complete_versioned_flybody_asset_inventory() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    assets = manifest["meshAssets"]

    assert assets["prefix"] == "flybody_fullsize_meshes_20260623a"
    assert assets["fileCount"] == 87
    assert assets["totalBytes"] == 140_210_791
    assert len(assets["files"]) == 87
    assert assets["inventorySha256"] == (
        "d67fa2c2649f7c5c755d3e0770955cd633e4b28ff203c124bd4cc193db4a3453"
    )
    assert assets["inventorySha256"] == _inventory_digest(assets["files"])


@pytest.mark.parametrize(
    "missing_field",
    ("prefix", "fileCount", "totalBytes", "inventorySha256", "files"),
)
def test_asset_verifier_rejects_an_incomplete_manifest(
    tmp_path: Path, missing_field: str
) -> None:
    assets = _synthetic_asset_manifest(tmp_path)
    del assets[missing_field]

    with pytest.raises(ValueError, match="mesh asset manifest"):
        body_module._verify_asset_inventory(tmp_path, assets)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("delete_entry", "incomplete"),
        ("escape_path", "path"),
        ("wrong_size", "size"),
        ("wrong_hash", "checksum"),
    ),
)
def test_asset_verifier_rejects_mutated_inventory_entries(
    tmp_path: Path, mutation: str, message: str
) -> None:
    assets = _synthetic_asset_manifest(tmp_path)
    files = assets["files"]
    assert isinstance(files, list)
    if mutation == "delete_entry":
        files.pop()
    elif mutation == "escape_path":
        files[0]["path"] = "../README"
    elif mutation == "wrong_size":
        files[0]["size"] += 1
    elif mutation == "wrong_hash":
        files[0]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match=message):
        body_module._verify_asset_inventory(tmp_path, assets)


@pytest.mark.parametrize("mutation", ("missing", "extra", "changed"))
def test_asset_verifier_rejects_a_changed_cache_without_touching_real_assets(
    tmp_path: Path, mutation: str
) -> None:
    assets = _synthetic_asset_manifest(tmp_path)
    if mutation == "missing":
        (tmp_path / "mesh.obj").unlink()
    elif mutation == "extra":
        (tmp_path / "unlisted.obj").write_bytes(b"v 1 1 1\n")
    else:
        path = tmp_path / "mesh.obj"
        path.write_bytes(b"v 9 9 9\n")  # same byte count; only SHA-256 detects it

    with pytest.raises(ValueError, match="mesh asset"):
        body_module._verify_asset_inventory(tmp_path, assets)


def test_loader_verifies_mesh_assets_before_compiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_assets(_manifest: object) -> None:
        raise ValueError("mesh assets rejected before compile")

    def compile_must_not_run() -> object:
        pytest.fail("FlyBody compilation ran before mesh verification")

    monkeypatch.setattr(body_module, "_verify_mesh_assets", reject_assets)
    monkeypatch.setattr(body_module, "_build_flygym_model", compile_must_not_run)

    with pytest.raises(ValueError, match="rejected before compile"):
        FlyBodyModel.load(MANIFEST_PATH)


def test_loader_rejects_an_internally_consistent_but_incomplete_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_root = tmp_path / "cache"
    asset_root.mkdir()
    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest["meshAssets"] = _synthetic_asset_manifest(asset_root)
    manifest_path = tmp_path / "flybody.json"
    manifest_path.write_text(json.dumps(manifest))

    monkeypatch.setattr(
        body_module, "lazy_load_asset_dir", lambda _prefix: asset_root
    )
    monkeypatch.setattr(
        body_module,
        "_build_flygym_model",
        lambda: pytest.fail("FlyBody compilation ran for an incomplete inventory"),
    )

    with pytest.raises(ValueError, match="complete pinned FlyBody asset set"):
        FlyBodyModel.load(manifest_path)
