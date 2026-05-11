import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from data2 import get_qclevr_dataloaders
from model_extracted_fb_crosscov import Conv2dEIRNN
from utils import AttrDict, seed


def load_state_dict_flexibly(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        cleaned[k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print("Missing keys while loading checkpoint:", missing)
    if unexpected:
        print("Unexpected keys while loading checkpoint:", unexpected)


def standardize_columns(X: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Column-wise z-score normalization: (X - mean_col) / std_col."""
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    return (X - mean) / std


def top_right_singular_vector_explicit(X: torch.Tensor, Y: torch.Tensor):
    """Build C = X.T @ Y and compute the top right singular vector with SVD."""
    C = (X.T @ Y) / max(1, X.shape[0] - 1)
    _, S, Vh = torch.linalg.svd(C, full_matrices=False)
    vec = Vh[0].contiguous()
    vec = vec / vec.norm().clamp_min(1e-12)
    ratio = float((S[0] ** 2 / (S ** 2).sum().clamp_min(1e-12)).item())
    return vec, ratio


def top_right_singular_vector_power(
    X: torch.Tensor,
    Y: torch.Tensor,
    num_iter: int = 50,
    eps: float = 1e-12,
):
    """Top right singular vector of C = X.T @ Y without explicitly building C.

    Uses alternating power iteration:
        u <- normalize(C v)   = normalize(X.T @ (Y @ v))
        v <- normalize(C.T u) = normalize(Y.T @ (X @ u))

    X, Y are [n_samples, D]. This avoids storing the [D, D] matrix.
    """
    D = X.shape[1]
    v = torch.randn(D, device=X.device, dtype=X.dtype)
    v = v / v.norm().clamp_min(eps)

    for _ in range(num_iter):
        u = X.T @ (Y @ v)
        u = u / u.norm().clamp_min(eps)
        v = Y.T @ (X @ u)
        v = v / v.norm().clamp_min(eps)

    # Approximate top singular value for logging only.
    u = X.T @ (Y @ v)
    sigma = u.norm() / max(1, X.shape[0] - 1)
    # Exact rank-1 explained ratio would require more singular values; we skip it.
    return v.contiguous(), float(sigma.item())


def unpack_batch(batch):
    if len(batch) == 3:
        cue, mixture, labels = batch
    elif len(batch) == 4:
        cue, mixture, labels, _ = batch
    elif len(batch) == 5:
        cue, mixture, labels, _, _ = batch
    else:
        raise ValueError(f"Unexpected batch structure with {len(batch)} elements")
    return cue, mixture


def collect_phase_hidden(model, dataloader, device, num_batches, source_phase, target_phase):
    base_model = getattr(model, "_orig_mod", model)
    num_layers = len(base_model.layers)
    source_rows = [[] for _ in range(num_layers)]
    target_rows = [[] for _ in range(num_layers)]

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            cue, mixture = unpack_batch(batch)
            cue = cue.to(device, non_blocking=(device.type == "cuda")) if cue is not None else None
            mixture = mixture.to(device, non_blocking=(device.type == "cuda"))

            _, _, phase_hidden = model(cue, mixture, return_phase_hidden=True)
            if phase_hidden[source_phase] is None:
                raise ValueError(f"source_phase={source_phase!r} was not available.")
            if phase_hidden[target_phase] is None:
                raise ValueError(f"target_phase={target_phase!r} was not available.")

            h_pyrs_source, _ = phase_hidden[source_phase]
            h_pyrs_target, _ = phase_hidden[target_phase]

            for layer_idx in range(num_layers):
                src_rows = []
                tgt_rows = []
                for t in range(base_model.num_steps):
                    src_rows.append(h_pyrs_source[t][layer_idx].detach().float().flatten(1))
                    tgt_rows.append(h_pyrs_target[t][layer_idx].detach().float().flatten(1))
                source_rows[layer_idx].append(torch.cat(src_rows, dim=0))
                target_rows[layer_idx].append(torch.cat(tgt_rows, dim=0))

    return source_rows, target_rows


def main():
    parser = argparse.ArgumentParser(
        description="Extract standardized cross-time covariance feedback vectors."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--source-phase", type=str, default="cue", choices=["cue", "mixture"])
    parser.add_argument("--target-phase", type=str, default="mixture", choices=["cue", "mixture"])
    parser.add_argument("--method", type=str, default="power", choices=["power", "explicit"])
    parser.add_argument("--power-iters", type=int, default=50)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--cpu-svd", action="store_true", help="Move X/Y to CPU before SVD/power iteration. Slower but may avoid GPU OOM.")
    args = parser.parse_args()

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    config = AttrDict(config)

    if config.seed is not None:
        seed(
            config.seed,
            deterministic=config.train.deterministic,
            cudnn_benchmark=config.train.cudnn_benchmark,
            allow_tf32=config.train.allow_tf32,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = dict(config.model)
    model_config["extracted_fb"] = False
    model_config["extracted_fb_vector_path"] = None
    source_model = Conv2dEIRNN(**model_config).to(device)
    load_state_dict_flexibly(source_model, args.checkpoint, device)

    train_loader, _ = get_qclevr_dataloaders(
        data_root=config.data.root,
        assets_path=config.data.assets_path,
        train_batch_size=config.data.batch_size,
        val_batch_size=config.data.val_batch_size,
        resolution=config.model.input_size,
        holdout=config.data.holdout,
        mode=config.data.mode,
        primitive=config.data.primitive,
        num_workers=config.data.num_workers,
        seed=config.seed,
    )

    source_rows, target_rows = collect_phase_hidden(
        source_model,
        train_loader,
        device,
        args.num_batches,
        args.source_phase,
        args.target_phase,
    )

    base_model = getattr(source_model, "_orig_mod", source_model)
    vectors = []
    scores = []
    work_device = torch.device("cpu") if args.cpu_svd else device

    for layer_idx in range(len(base_model.layers)):
        X = torch.cat(source_rows[layer_idx], dim=0).to(work_device)
        Y = torch.cat(target_rows[layer_idx], dim=0).to(work_device)
        if X.shape != Y.shape:
            raise ValueError(f"Layer {layer_idx}: source/target mismatch {X.shape} vs {Y.shape}")

        # Important requested step: column-wise z-score normalization.
        X = standardize_columns(X, eps=args.eps)
        Y = standardize_columns(Y, eps=args.eps)

        if args.method == "explicit":
            vec, score = top_right_singular_vector_explicit(X, Y)
            score_name = "rank1_singular_value_ratio"
        else:
            vec, score = top_right_singular_vector_power(X, Y, num_iter=args.power_iters)
            score_name = "approx_top_singular_value"

        vectors.append(vec.cpu())
        scores.append(score)
        print(f"Layer {layer_idx}: X/Y={tuple(X.shape)}, score={score:.6g}")

        del X, Y, vec
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "vectors": vectors,
        "scores": scores,
        "score_name": score_name,
        "h_pyr_dims": list(base_model.h_pyr_dims),
        "input_sizes": list(base_model.input_sizes),
        "num_steps": int(base_model.num_steps),
        "method": f"standardized_cross_time_covariance_{args.method}",
        "source_phase": args.source_phase,
        "target_phase": args.target_phase,
        "standardize_columns": True,
        "eps": args.eps,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"Saved extracted feedback vectors to: {output_path}")


if __name__ == "__main__":
    main()
