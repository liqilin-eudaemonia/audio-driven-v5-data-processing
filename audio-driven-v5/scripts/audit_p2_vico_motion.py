"""Read-only independent coverage, provenance and numerical audit of motion run."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.linalg import sqrtm


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8*1024*1024), b""):h.update(b)
    return h.hexdigest()


def audit(work):
    p = json.loads((work/"protocol.json").read_text())
    result = json.loads((work/"RESULTS.json").read_text())
    scoring = json.loads((work/"SCORING_PROTOCOL.json").read_text())
    archive = json.loads((work/"SOURCE_ARCHIVE.json").read_text())
    assert scoring["extraction_protocol_sha256"] == sha(work/"protocol.json")
    for path, digest in scoring["implementation_sha256"].items():assert sha(path) == digest, path
    assert len(p["samples"]) == result["clips"] == 43
    assert len({x["id"] for x in p["samples"]}) == 43
    assert result["strict_customlistener_table_reproduction"] is False
    assert json.loads((work/"BATCH_PARITY.json").read_text())["passed"] is True
    assert not list(work.glob("FAILURE*.json"))
    checked = {}
    for name, digest in p["parent_files"].items():
        assert sha(Path(p["parent_generation"])/name) == digest, name
    for path, digest in p["assets"].items():assert sha(path) == digest, path
    for path, digest in p["source_sha256"].items():
        assert sha(archive[path]) == digest, path
        if path not in scoring["changed_scoring_sources"]:assert sha(path) == digest, path
    result_rows = {r["id"]: r for r in result["per_clip"]}
    expected_ids = {x["id"] for x in p["samples"]}
    assert set(result_rows) == expected_ids
    assert {f.stem for f in (work/"features/speaker").glob("*.npz")} == expected_ids
    missing, multiple, per_role = {}, {}, {}
    independent_fd_errors, independent_var_errors, independent_fid_errors = [], [], []
    correlation_errors = []
    for sample in p["samples"]:
        uid = sample["id"];arrays = {}
        for role, info in sample["streams"].items():
            path = work/"features"/role/(uid+".npz")
            receipt = json.loads(path.with_suffix(".json").read_text())
            assert sha(info["path"]) == info["sha256"] == receipt["input_sha256"]
            assert sha(path) == receipt["feature_sha256"]
            assert receipt["protocol_sha256"] == sha(work/"protocol.json")
            with np.load(path) as z:
                data = {k: z[k] for k in z.files}
            assert data["coeff"].shape == (info["frames"], 257)
            assert data["motion"].shape == (info["frames"], 70)
            assert data["landmarks"].shape == (info["frames"], 68, 2)
            assert data["detected"].shape == (info["frames"],)
            assert all(np.isfinite(v).all() for v in data.values())
            expected_motion = np.concatenate([data["coeff"][:, 80:144], data["coeff"][:, 224:227], data["coeff"][:, 254:257]], axis=1)
            assert np.array_equal(expected_motion, data["motion"])
            assert data["detected"].any()
            assert int((~data["detected"]).sum()) == receipt["missing_face_frames"]
            missing[role] = missing.get(role, 0)+receipt["missing_face_frames"]
            multiple[role] = multiple.get(role, 0)+receipt["multi_face_frames"]
            per_role[role] = per_role.get(role, 0)+info["frames"]
            arrays[role] = data["motion"].astype(np.float64)
        s, g, pred = (arrays[k] for k in ("speaker", "gt", "prediction"))
        row = result_rows[uid];assert row["frames"] == len(g)
        assert row["gt_self_distance_max"] == 0
        starts = row["timing_audit"]["window_starts_frames"]
        assert starts == list(range(0, len(g)-119, 30))
        raw = row["metrics_raw"]
        for group, a, b in [("exp", 0, 64), ("angle", 64, 67), ("trans", 67, 70)]:
            independent_fd_errors.append(abs(float(np.sum(np.abs(pred[:, a:b]-g[:, a:b]))/(len(g)*(b-a)))-raw["FD"][group]))
        for group, a, b in [("exp", 0, 64), ("pose", 64, 70)]:
            v = sum(float(np.mean((pred[:, i]-pred[:, i].mean())**2)) for i in range(a, b))/(b-a)
            independent_var_errors.append(abs(v-raw["V_D"][group]))
            assert abs(raw["SND"][group]-raw["FID_fm"][group]-raw["FID_delta_fm"][group]) < 1e-10
            # Distinct covariance-product sqrtm implementation for all real clips.
            for key, x, y in [("FID_fm", g[:, a:b], pred[:, a:b]),
                              ("FID_delta_fm", np.diff(g[:, a:b], axis=0), np.diff(pred[:, a:b], axis=0))]:
                c1, c2 = np.cov(x.T), np.cov(y.T)
                covmean = sqrtm(c1@c2)
                assert np.abs(covmean.imag).max() < 1e-6 if np.iscomplexobj(covmean) else True
                distance = float(np.square(x.mean(0)-y.mean(0)).sum()+np.trace(c1+c2-2*covmean.real))
                independent_fid_errors.append(abs(max(0, distance)-raw[key][group]))
            # Whole-clip TLCC by np.corrcoef, independent of vectorized scorer.
            residuals = []
            for lag in (range(-60, 61) if len(g) > 61 else []):
                if lag > 0:ss, gg, pp = s[:-lag, a:b], g[lag:, a:b], pred[lag:, a:b]
                elif lag < 0:ss, gg, pp = s[-lag:, a:b], g[:lag, a:b], pred[:lag, a:b]
                else:ss, gg, pp = s[:, a:b], g[:, a:b], pred[:, a:b]
                # Check 2 representative channels per group against corrcoef;
                # also independently accumulate the full-coordinate RTLCC.
                sm, gm, pm = ss-ss.mean(0), gg-gg.mean(0), pp-pp.mean(0)
                sg = np.sqrt((sm**2).sum(0)*(gm**2).sum(0));sp = np.sqrt((sm**2).sum(0)*(pm**2).sum(0))
                cg = np.divide((sm*gm).sum(0), sg, out=np.zeros(b-a), where=sg>1e-12)
                cp = np.divide((sm*pm).sum(0), sp, out=np.zeros(b-a), where=sp>1e-12)
                for j in [0, b-a-1]:
                    if sg[j] > 1e-12:correlation_errors.append(abs(cg[j]-np.corrcoef(ss[:, j], gg[:, j])[0, 1]))
                    if sp[j] > 1e-12:correlation_errors.append(abs(cp[j]-np.corrcoef(ss[:, j], pp[:, j])[0, 1]))
                residuals.extend(np.abs(cg-cp).tolist())
            if residuals:assert abs(np.mean(residuals)-raw["RTLCC"][group]) < 1e-12
            else:assert raw["RTLCC"][group] is None
            if not starts:assert raw["RWTLCC"][group] is None
        checked[uid] = dict(frames=len(g), streams=3, gt_self_distance_max=row["gt_self_distance_max"])
    for metric, groups in result["metrics_raw"].items():
        for group, value in groups.items():
            supported = [r["metrics_raw"][metric][group] for r in result_rows.values() if r["metrics_raw"][metric][group] is not None]
            assert abs(np.mean(supported)-value) < 1e-12
            assert len(supported) == result["coverage_clips"][metric][group]
            assert len(supported) == (42 if metric == "RTLCC" else 41 if metric == "RWTLCC" else 43)
            scale = 100 if metric in ("FD", "FID_delta_fm") else 1
            assert abs(value*scale-result["metrics_table_scale"][metric][group]) < 1e-12
    assert set(per_role.values()) == {16288}
    assert max(independent_fd_errors) < 1e-12
    assert max(independent_var_errors) < 1e-12
    assert max(independent_fid_errors) < 1e-5
    assert max(correlation_errors) < 1e-12
    return dict(passed=True, clips=43, streams=129, frames_per_role=per_role, total_decoded_frames=sum(per_role.values()),
        missing_face_frames=missing, multi_face_frames=multiple,
        parent_generation_immutable=True, assets_and_source_hashes_verified=True,
        independent_fd_max_abs_error=max(independent_fd_errors), independent_variance_max_abs_error=max(independent_var_errors),
        scipy_covariance_product_sqrtm_max_abs_error=max(independent_fid_errors), numpy_corrcoef_max_abs_error=max(correlation_errors),
        protocol_sha256=sha(work/"protocol.json"), results_sha256=sha(work/"RESULTS.json"), checked=checked)


if __name__ == "__main__":
    p = argparse.ArgumentParser();p.add_argument("--work", type=Path, required=True);a = p.parse_args()
    report = audit(a.work.resolve())
    (a.work/"FINAL_AUDIT.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "checked"}, indent=2))
