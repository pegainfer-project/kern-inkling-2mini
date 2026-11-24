# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the M4 verify sampling kernel (`source/sampling.cu`, `spec_sampling`).

Phase A (CPU-only, source contract):
  python -m unittest tests/unit/test_sampling.py
Phase B (GPU, after `scripts/build.py --update` / `dspark_manifest.py --build`):
  KERN_RUN_GPU=1 python -m unittest tests/unit/test_sampling.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import dspark_manifest as dm  # noqa: E402

_WORKSPACE = ROOT.parent
KERN_PY_DIR = Path(os.environ.get("KERN_PY_ROOT", _WORKSPACE / "kern/crates/kern-py/python"))
LIB_KERN_SO = Path(os.environ.get("KERN_LIB_SO", _WORKSPACE / "kern/target/release/lib_kern.so"))
KERNELS_DIR = ROOT / "build/sm_120"
SPEC_BUILD = KERNELS_DIR / "spec_build.json"
SAMPLING_SRC = ROOT / "source/sampling.cu"

RUN_GPU = os.environ.get("KERN_RUN_GPU") == "1"


def _load_kern_runtime():
    """Load `kern` and `kern.host` without modifying external packages."""
    try:
        import kern
        from kern.host import Bytes, DTYPES, resolve

        def region_ex(rt, m: dict, name: str):
            import torch

            ptr, nbytes = rt.region(name)
            b = m["buffers"][name]
            shape = [resolve(m, m["vars"][d]["max"] if d in m.get("vars", {}) else d) for d in b["shape"]]
            dt_map = {**DTYPES, "fp8e4m3": torch.float8_e4m3fn}
            return torch.as_tensor(Bytes(ptr, nbytes), device="cuda").view(dt_map[b["dtype"]]).view(*shape)

        return kern, region_ex
    except ImportError:
        tmp_pkg = Path(tempfile.mkdtemp(prefix="kern_py_shim_"))
        pkg_dir = tmp_pkg / "kern"
        pkg_dir.mkdir()
        for fname in ("host.py", "group.py", "trace.py"):
            shutil.copy2(KERN_PY_DIR / "kern" / fname, pkg_dir / fname)
        os.symlink(LIB_KERN_SO, pkg_dir / "_kern.so")
        (pkg_dir / "__init__.py").write_text(
            "from kern._kern import Runtime, __version__\n"
            "try:\n"
            "    from kern._kern import HostGraph\n"
            "except ImportError:\n"
            "    HostGraph = None\n"
        )
        sys.path.insert(0, str(tmp_pkg))
        import kern
        from kern.host import Bytes, DTYPES, resolve

        def region_ex(rt, m: dict, name: str):
            import torch

            ptr, nbytes = rt.region(name)
            b = m["buffers"][name]
            shape = [resolve(m, m["vars"][d]["max"] if d in m.get("vars", {}) else d) for d in b["shape"]]
            dt_map = {**DTYPES, "fp8e4m3": torch.float8_e4m3fn}
            return torch.as_tensor(Bytes(ptr, nbytes), device="cuda").view(dt_map[b["dtype"]]).view(*shape)

        return kern, region_ex


def _minp_body() -> str:
    """Return the source of kern_verify_minp_sample_f32 (last kernel in sampling.cu)."""
    src = SAMPLING_SRC.read_text()
    start = src.index("void kern_verify_minp_sample_f32(")
    return src[start:]


class TestSamplingSourceContract(unittest.TestCase):
    """CPU-only guard for the min-p fallback contract (runs without build artifacts)."""

    def test_minp_predicate_requires_strictly_positive_probability(self):
        body = _minp_body()
        self.assertIn(
            "[&](float x) { return x > 0.0f && x >= pivot; }",
            body,
            "MinP sampling must exclude exactly-zero probabilities from the valid/fallback set",
        )
        self.assertNotIn(
            "return x >= pivot;",
            body,
            "Bare `x >= pivot` accepts zero-probability padding slots when min_p == 0",
        )

    def test_minp_fallback_and_greedy_guards_are_preserved(self):
        body = _minp_body()
        # Greedy rows still exit before touching probs.
        self.assertIn("if (raw_temperature[bx] <= 0.0f) return;", body)
        # Empty valid set still reports valid=false instead of inventing a token.
        self.assertIn("if (temp_storage.last_valid_id == -1)", body)
        self.assertIn("valid[bx] = false;", body)


@unittest.skipUnless(RUN_GPU, "GPU tests skipped in Phase A (set KERN_RUN_GPU=1 in Phase B)")
class TestMinPSamplingGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA GPU not available")
        if not SPEC_BUILD.exists():
            raise unittest.SkipTest(f"Missing {SPEC_BUILD}; run dspark_manifest.py --build")
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) != (12, 0):
            raise unittest.SkipTest(f"spec_sampling cubins target sm_120a, found sm_{major}{minor}")
        cls.torch = torch
        kern_mod, region_fn = _load_kern_runtime()
        cls.kern = kern_mod
        cls.region = staticmethod(region_fn)
        spec = json.loads(SPEC_BUILD.read_text())
        cls.sampling = spec["sampling"]
        info = spec["modules"]["spec_sampling"]
        cls.module = {"spec_sampling": {"source": info["source"], "sha256": info["sha256"]}}

    def _run_minp(self, probs, min_p, seeds):
        """Launch kern_verify_minp_sample_f32 directly on already-computed probs."""
        torch = self.torch
        rows, d = len(seeds), probs.shape[1]
        self.assertEqual(probs.shape[0], len(seeds))
        entry = self.sampling["entries"]["MinPSamplingFromProbKernel"]["entry"]
        params = [
            "inout buffer<f32>",
            "in buffer<f32>",
            "out buffer<i32>",
            "out buffer<u8>",
            "i64",
            "f32",
            "i32",
            "in buffer<i64>",
            "i64",
            "in buffer<u64>",
            "i64",
            "i32",
            "i32",
            "in buffer<f32>",
        ]
        # Signature of kern_verify_minp_sample_f32: probs, min_p_arr, output, valid,
        # indices(null), min_p_val, d, seed_arr, seed_val, offset_arr, offset_val,
        # seed_stride, offset_stride, raw_temperature. The two i64 zeros after `valid`
        # and the one after `seed` are null/scalar values, matching verify_softcap_sample_op.
        args = [
            {"buf": "probs"},
            {"buf": "minp"},
            {"buf": "drawn"},
            {"buf": "valid"},
            {"i64": 0},
            {"f32": 0.0},
            {"i32": d},
            {"buf": "seed"},
            {"i64": 0},
            {"buf": "offset"},
            {"i64": 0},
            {"i32": 1},
            {"i32": 1},
            {"buf": "temperature"},
        ]
        m = {
            "schema_version": 7,
            "model": "minp-sampling-test",
            "vars": {"rows": {"max": rows}},
            "states": {},
            "buffers": {
                "probs": {"dtype": "f32", "shape": ["rows", d], "kind": "inout"},
                "minp": {"dtype": "f32", "shape": ["rows"], "kind": "input"},
                "drawn": {"dtype": "i32", "shape": ["rows"], "kind": "output"},
                "valid": {"dtype": "u8", "shape": ["rows"], "kind": "output"},
                "seed": {"dtype": "i64", "shape": ["rows"], "kind": "input"},
                "offset": {"dtype": "u64", "shape": ["rows"], "kind": "input"},
                "temperature": {"dtype": "f32", "shape": ["rows"], "kind": "input"},
            },
            "modules": self.module,
            "ops": {
                "minp": {
                    "params": params,
                    "impl": {
                        "launches": [
                            {
                                "module": "spec_sampling",
                                "entry": entry,
                                "params": params,
                                "block": [dm.BLOCK_SAMPLING, 1, 1],
                                "grid": ["rows", 1, 1],
                                "shared_mem": self.sampling["sizes"]["minp"],
                                "args": args,
                            }
                        ]
                    },
                }
            },
            "programs": {"run": {"calls": [{"op": "minp", "args": args}]}},
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.json"
            path.write_text(json.dumps(m))
            rt = self.kern.Runtime(str(path), str(KERNELS_DIR), [], 0)
        views = {name: self.region(rt, m, name) for name in m["buffers"]}
        views["probs"].copy_(probs)
        views["minp"].fill_(min_p)
        views["seed"].copy_(torch.tensor(seeds, dtype=torch.int64, device="cuda"))
        views["offset"].zero_()
        views["temperature"].fill_(1.0)
        views["drawn"].fill_(-1)
        views["valid"].zero_()
        torch.cuda.synchronize()
        rt.run("run", {"rows": rows})
        torch.cuda.synchronize()
        return views["drawn"].tolist(), views["valid"].tolist()

    def test_zero_probability_row_never_returns_d_minus_1(self):
        """q == 0 and u == 0 force the fallback; a zero-probability slot must not be 'valid'."""
        torch = self.torch
        cases = [
            (8192, 0.0),
            (dm.VOCAB_PADDED, 0.0),
            (8192, 0.5),  # max_val == 0 -> pivot == 0 for min_p > 0 as well
        ]
        for d, min_p in cases:
            with self.subTest(d=d, min_p=min_p):
                probs = torch.zeros((2, d), dtype=torch.float32, device="cuda")
                drawn, valid = self._run_minp(probs, min_p, seeds=[1234, 5678])
                self.assertEqual(valid, [0, 0], f"d={d} min_p={min_p}: zero-prob row must be invalid")
                for row, token in enumerate(drawn):
                    self.assertNotEqual(token, d - 1, f"d={d} row={row}: sampled the last (zero-prob) slot")

    def test_zero_probability_padding_tail_is_never_sampled(self):
        torch = self.torch
        seeds = list(range(3000, 3016))
        rows = len(seeds)
        probs = torch.zeros((rows, dm.VOCAB_PADDED), dtype=torch.float32, device="cuda")
        probs[:, dm.VOCAB_UNPADDED - 1] = 1.0  # token 200057, the last non-padding id
        drawn, valid = self._run_minp(probs, 0.0, seeds)
        self.assertEqual(valid, [1] * rows)
        self.assertEqual(drawn, [dm.VOCAB_UNPADDED - 1] * rows)

    def test_positive_min_p_keeps_the_accepted_set(self):
        torch = self.torch
        rows, d = 16, 8192
        probs = torch.zeros((rows, d), dtype=torch.float32, device="cuda")
        probs[:, 0], probs[:, 1], probs[:, 2], probs[:, 3] = 0.5, 0.2, 0.1, 0.05
        drawn, valid = self._run_minp(probs, 0.2, seeds=list(range(4000, 4000 + rows)))
        self.assertEqual(valid, [1] * rows)
        self.assertTrue(all(0 <= token <= 2 for token in drawn), f"drawn={drawn}")


if __name__ == "__main__":
    unittest.main()
