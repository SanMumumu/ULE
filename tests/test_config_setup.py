import argparse
import ast
import json
import pathlib


class AttrDict(dict):
    __getattr__ = dict.get

    def __setattr__(self, key, value):
        self[key] = value


def _to_attr(value):
    if isinstance(value, dict):
        out = AttrDict()
        for k, v in value.items():
            out[k] = _to_attr(v)
        return out
    if isinstance(value, list):
        return [_to_attr(v) for v in value]
    return value


class FakeOmegaConf:
    @staticmethod
    def load(path):
        return _to_attr(json.loads(pathlib.Path(path).read_text()))

    @staticmethod
    def create(obj):
        return _to_attr(obj)


def _load_config_setup():
    src_path = pathlib.Path(__file__).resolve().parents[1] / "tools" / "train_utils.py"
    source = src_path.read_text()
    module = ast.parse(source)
    fn_node = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "config_setup"
    )
    fn_src = ast.get_source_segment(source, fn_node)
    namespace = {"OmegaConf": FakeOmegaConf}
    exec(fn_src, namespace)
    return namespace["config_setup"]


def _write_cfg(tmp_path, model_block: str) -> str:
    model = {
        "base_learning_rate": 1.0e-4,
        "cond_prob": 0.7,
        "eval_freq": 100,
        "eval_samples": 8,
        "log_freq": 10,
        "resume": False,
        "max_iter": 20,
        "sit_config": {
            "input_size": 8,
            "in_channels": 4,
            "encoder_depth": 2,
            "bn_momentum": 0.1,
        },
        "params": {},
    }

    if model_block == "model_cfg":
        model["cfg_scale"] = 5.5
        model["params"]["w"] = 0
    elif model_block == "params_cfg":
        model["params"]["cfg_scale"] = 4.0
        model["params"]["w"] = 0
    elif model_block == "legacy_w_only":
        model["params"]["w"] = 0
    else:
        raise ValueError(f"Unknown model block key: {model_block}")

    content = {
        "vae": {
            "amp": False,
            "max_iter": 10,
            "params": {
                "embed_dim": 4,
                "perceptual_weight": 1.0,
                "lossconfig": {"params": {"disc_start": 0}},
            },
            "vaeconfig": {
                "channels": 64,
                "resolution": 32,
                "cond_frames": 2,
                "pred_frames": 4,
                "in_channels": 3,
                "out_channels": 3,
                "splits": 1,
            },
        },
        "model": model,
    }
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(json.dumps(content))
    return str(cfg)


def _base_args(cfg_path: str) -> argparse.Namespace:
    return argparse.Namespace(vae_config=cfg_path, cfg_scale=3.0)


def test_config_setup_prefers_model_cfg_scale(tmp_path):
    config_setup = _load_config_setup()
    cfg = _write_cfg(tmp_path, "model_cfg")
    args = config_setup(_base_args(cfg))
    assert args.cfg_scale == 5.5


def test_config_setup_uses_params_cfg_scale_when_model_level_missing(tmp_path):
    config_setup = _load_config_setup()
    cfg = _write_cfg(tmp_path, "params_cfg")
    args = config_setup(_base_args(cfg))
    assert args.cfg_scale == 4.0


def test_config_setup_does_not_fallback_to_legacy_w(tmp_path):
    config_setup = _load_config_setup()
    cfg = _write_cfg(tmp_path, "legacy_w_only")
    args = config_setup(_base_args(cfg))
    assert args.cfg_scale == 3.0
