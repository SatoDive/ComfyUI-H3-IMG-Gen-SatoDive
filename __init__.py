"""SatoDive H3: all-in-one nodes for MiniMax H3 stills. Standalone: the one-frame latent and the still decode are built in."""
import math
import torch
import torch.nn.functional as F
import comfy.nested_tensor
import comfy.samplers
import comfy.utils
import folder_paths
import nodes as comfy_nodes

ASPECTS = {"16:9": (16, 9), "3:2": (3, 2), "4:3": (4, 3), "1:1": (1, 1),
           "3:4": (3, 4), "2:3": (2, 3), "9:16": (9, 16), "21:9": (21, 9)}
MEGAPIXELS = ["2.5", "3", "4", "6", "8"]


class SatoDiveH3Size:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "aspect": (list(ASPECTS.keys()) + ["Custom"], {"default": "16:9"}),
            "megapixels": (MEGAPIXELS, {"default": "2.5", "tooltip": "Target area. H3 stills are best from 3 MP up."}),
        }, "optional": {
            "width": ("INT", {"default": 1920, "min": 64, "max": 8192, "step": 1}),
            "height": ("INT", {"default": 1088, "min": 64, "max": 8192, "step": 1}),
        }}

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "size"
    CATEGORY = "SatoDive/H3"
    DESCRIPTION = "Width/height (multiples of 32) for a given aspect ratio and megapixel target."

    def size(self, aspect, megapixels, width=None, height=None):
        if aspect == "Custom":
            snap32 = lambda v: max(64, int(round(v / 32.0)) * 32)
            return (snap32(width or 1920), snap32(height or 1088))
        aw, ah = ASPECTS[aspect]
        ratio = aw / ah
        area = float(megapixels) * 1_000_000
        h = math.sqrt(area / ratio)
        w = h * ratio
        snap = lambda v: max(64, int(round(v / 32.0)) * 32)
        return (snap(w), snap(h))


class SatoDiveH3LatentUpscale:
    """Spatially resize a one-frame H3 latent (video stream only; audio stream untouched) for a low-denoise refine pass."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",),
            "scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 2.5, "step": 0.05}),
            "method": (["bicubic", "bilinear", "nearest-exact"], {"default": "bicubic"}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "up"
    CATEGORY = "SatoDive/H3"
    DESCRIPTION = "Upscales the video latent of an H3 still. Pair with a second sampler at denoise ~0.4-0.5."

    def up(self, samples, scale, method):
        lat = samples["samples"]
        if not getattr(lat, "is_nested", False):
            return (samples,)
        video, audio = lat.unbind()
        b, c, t, h, w = video.shape
        if t != 1:
            return (samples,)
        # DiT patchifies 2x2, so the latent grid must stay even
        nh = max(2, int(round(h * scale / 2.0)) * 2)
        nw = max(2, int(round(w * scale / 2.0)) * 2)
        kw = {} if method == "nearest-exact" else {"align_corners": False}
        v = F.interpolate(video[:, :, 0].float(), size=(nh, nw), mode=method, **kw).to(video.dtype).unsqueeze(2)
        out = samples.copy()
        out["samples"] = comfy.nested_tensor.NestedTensor((v, audio))
        return (out,)


# ---------------------------------------------------------------------------
# All-in-one nodes: Prompt & Size (3 nodes -> 1) and Generate (sampling + LoRA + refine + decode -> 1)
# ---------------------------------------------------------------------------

def _run(cls, **kw):
    """Call a native ComfyUI node class, whether it is a V3 (execute) or legacy (FUNCTION) node."""
    if hasattr(cls, "define_schema") and hasattr(cls, "execute"):
        out = cls.execute(**kw)
        return out.result if hasattr(out, "result") else out
    res = getattr(cls(), cls.FUNCTION)(**kw)
    return res.result if hasattr(res, "result") else res


# ---------------------------------------------------------------------------
# Built-in one-frame H3 still latent + decode (no other custom node needed).
# Same method as Fizgig H3 Still by Peter Neill (MIT, github.com/shootthesound/ComfyUI-Fizgig-H3-Still):
# H3's image convention is ONE latent frame; the stock VAE Decode bands a lone latent frame, so the frame is
# replicated into a full 5-latent group, decoded, and pixel frame 3 (past the decoder's causal lead-in) is kept.
# ---------------------------------------------------------------------------
_STILL_FPS, _AUDIO_LATENT_FPS = 24, 40


class _StillLatent:
    def make(self, width, height, batch_size=1):
        import comfy.model_management
        dev = comfy.model_management.intermediate_device()
        lh, lw = (height // 16) // 2 * 2, (width // 16) // 2 * 2   # the DiT patchifies 2x2: even latent grid
        video = torch.zeros([batch_size, 24, 1, lh, lw], device=dev)
        audio = torch.zeros([batch_size, 32, 2, max(1, round(1 / _STILL_FPS * _AUDIO_LATENT_FPS))], device=dev)
        return ({"samples": comfy.nested_tensor.NestedTensor((video, audio))},)


class _StillDecode:
    GROUP, KEEP = 5, 3

    def decode(self, vae, samples):
        import comfy.model_management as mm
        latent = samples["samples"]
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[0]
        fsm = getattr(vae, "first_stage_model", None)
        if latent.ndim != 5 or latent.shape[2] != 1 or not hasattr(fsm, "_adaptive_decode"):
            images = vae.decode(latent)          # clips / other VAEs: the stock path
            if len(images.shape) == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            return (images,)
        group_shape = (1, latent.shape[1], self.GROUP, latent.shape[3], latent.shape[4])
        mm.load_models_gpu([vae.patcher], memory_required=vae.memory_used_decode(group_shape, vae.vae_dtype),
                           force_full_load=getattr(vae, "disable_offload", False))
        out = []
        with torch.no_grad():
            for b in range(latent.shape[0]):
                z = latent[b:b + 1].to(device=vae.device, dtype=vae.vae_dtype)
                lm = fsm.latents_mean.view(1, -1, 1, 1, 1).to(z)
                ls = fsm.latents_std.view(1, -1, 1, 1, 1).to(z)
                raw = fsm._adaptive_decode((z * ls + lm).repeat(1, 1, self.GROUP, 1, 1))
                px = fsm._finalize_pixels(raw[:, :, self.KEEP:self.KEEP + 1])
                out.append(px[:, :, 0].movedim(1, -1).to(mm.intermediate_device()))
                del raw
        return (torch.cat(out),)


_STILL = {"FizgigH3StillLatent": _StillLatent, "FizgigH3StillDecode": _StillDecode}


def _fizgig(name):
    """The one-frame latent / decode helpers (built in - the pack needs no other custom node)."""
    return _STILL[name]()


def _native(name):
    """Lazy lookup of native ComfyUI nodes (only the one asked for must exist)."""
    try:
        from comfy_extras import nodes_custom_sampler as ncs, nodes_minimax_h3 as h3
    except ImportError as e:
        raise RuntimeError("This ComfyUI has no native MiniMax H3 nodes (comfy_extras/nodes_minimax_h3.py). "
                           "Update ComfyUI to a version that includes MiniMax H3. (%s)" % e)
    src = {"noise": (ncs, "RandomNoise"), "ksampler": (ncs, "KSamplerSelect"), "sched": (ncs, "BasicScheduler"),
           "guider": (ncs, "BasicGuider"), "sca": (ncs, "SamplerCustomAdvanced"),
           "i2v": (h3, "MiniMaxH3ImageToVideo"), "ref": (h3, "MiniMaxH3ReferenceToVideo")}
    mod, attr = src[name]
    return getattr(mod, attr)


class SatoDiveH3Prompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",), "vae": ("VAE",),
            "prompt": ("STRING", {"multiline": True, "dynamic_prompts": True, "default": ""}),
            "aspect": (list(ASPECTS.keys()) + ["Custom"], {"default": "16:9", "tooltip": "Custom = use width and height below."}),
            "megapixels": (MEGAPIXELS, {"default": "2.5", "tooltip": "Ignored when aspect = Custom."}),
            "width": ("INT", {"default": 1920, "min": 256, "max": 8192, "step": 32, "tooltip": "Used when aspect = Custom (snapped to a multiple of 32)."}),
            "height": ("INT", {"default": 1088, "min": 256, "max": 8192, "step": 32, "tooltip": "Used when aspect = Custom (snapped to a multiple of 32)."}),
        }, "optional": {
            "ref_image_1": ("IMAGE", {"tooltip": "Reference image. Use <Picture N> in the prompt; N counts only connected slots, in slot order."}),
            "ref_image_2": ("IMAGE", {"tooltip": "Reference image. Use <Picture N> in the prompt; N counts only connected slots, in slot order."}),
            "ref_image_3": ("IMAGE", {"tooltip": "Reference image. Use <Picture N> in the prompt; N counts only connected slots, in slot order."}),
            "ref_image_4": ("IMAGE", {"tooltip": "Reference image. Use <Picture N> in the prompt; N counts only connected slots, in slot order."}),
            "ref_image_5": ("IMAGE", {"tooltip": "Reference image. Use <Picture N> in the prompt; N counts only connected slots, in slot order."}),
            "ref_image_6": ("IMAGE", {"tooltip": "Reference image. Use <Picture N> in the prompt; N counts only connected slots, in slot order."}),
            "ref_image_7": ("IMAGE", {"tooltip": "Reference image. Use <Picture N> in the prompt; N counts only connected slots, in slot order."}),
            "ref_image_8": ("IMAGE", {"tooltip": "Reference image. Use <Picture N> in the prompt; N counts only connected slots, in slot order."}),
            "ref_image_9": ("IMAGE", {"tooltip": "Reference image. Use <Picture N> in the prompt; N counts only connected slots, in slot order."}),
            "ref_image_size": (["match", "max"], {"default": "match", "tooltip": "match = refs scaled (down only) to the output area; max = 2048px short edge, best identity but slower."}),
        }}

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "build"
    CATEGORY = "SatoDive/H3"
    DESCRIPTION = "Prompt + size preset + one-frame H3 latent in a single node (H3 supports batch size 1)."

    def build(self, clip, vae, prompt, aspect, megapixels, width=1920, height=1088, ref_image_size="match", **kw):
        w, h = SatoDiveH3Size().size(aspect, megapixels, width, height)
        refs = {"ref_image_%d" % i: kw["ref_image_%d" % i] for i in range(1, 10) if kw.get("ref_image_%d" % i) is not None}
        # conditioning does not depend on clip length without keyframes; use the minimum (5)
        if refs:  # reference / edit mode: <Picture i> tags in the prompt
            cond = _run(_native("ref"), clip=clip, vae=vae, prompt=prompt, width=w, height=h, length=5,
                        ref_image_size=ref_image_size, ref_images=refs)[0]
        else:
            cond = _run(_native("i2v"), clip=clip, vae=vae, prompt=prompt, width=w, height=h, length=5)[0]
        lat = _fizgig("FizgigH3StillLatent").make(width=w, height=h, batch_size=1)[0]
        return (cond, lat)


PRESETS = {  # lora_strength, steps, refine
    "Fast 2-pass": (0.38, 20, True),
    "Fast draft": (0.38, 20, False),
    "Max quality": (0.0, 50, False),
}


def _tail_sigmas(model, scheduler, start_sigma, steps):
    """Sigma schedule that starts at `start_sigma` (model-agnostic; a raw denoise fraction maps to very
    different noise levels under a flow shift, e.g. denoise 0.45 at shift 12 starts at sigma ~0.91)."""
    full = _run(_native("sched"), model=model, scheduler=scheduler, steps=1024, denoise=1.0)[0]
    vals = [float(v) for v in full.tolist()]
    start = next((i for i, v in enumerate(vals) if v <= start_sigma), max(0, len(vals) - 2))
    tail = vals[start:]
    if len(tail) < 2:
        tail = [vals[-2], 0.0]
    idx = sorted(set(round(k * (len(tail) - 1) / steps) for k in range(steps + 1)))
    picked = [tail[i] for i in idx]
    return torch.tensor(picked, dtype=full.dtype, device=full.device)


def _sample(model, positive, latent, seed, steps, denoise, sampler_name, scheduler, start_sigma=None):
    noise = _run(_native("noise"), noise_seed=seed)[0]
    sampler = _run(_native("ksampler"), sampler_name=sampler_name)[0]
    if start_sigma is None:
        sigmas = _run(_native("sched"), model=model, scheduler=scheduler, steps=steps, denoise=denoise)[0]
    else:
        sigmas = _tail_sigmas(model, scheduler, start_sigma, steps)
    guider = _run(_native("guider"), model=model, conditioning=positive)[0]
    return _run(_native("sca"), noise=noise, guider=guider, sampler=sampler, sigmas=sigmas, latent_image=latent)[0]


def _pixel_upscale(img, model_name, target_mp):
    """Model upscale (ESRGAN-style) on the decoded image, optionally resized to a target area."""
    from comfy_extras import nodes_upscale_model as nu
    um = _run(nu.UpscaleModelLoader, model_name=model_name)[0]
    out = _run(nu.ImageUpscaleWithModel, upscale_model=um, image=img)[0]
    if target_mp != "native":
        _, h, w, _ = out.shape
        k = math.sqrt(float(target_mp) * 1e6 / (h * w))
        nw, nh = max(32, int(round(w * k / 8.0)) * 8), max(32, int(round(h * k / 8.0)) * 8)
        if (nw, nh) != (w, h):
            out = comfy.utils.common_upscale(out.movedim(-1, 1), nw, nh, "lanczos", "disabled").movedim(1, -1)
    return out


def _auto_strength(scale):
    """Heuristic start sigma for the refine pass: a bigger upscale needs more noise to invent detail."""
    return round(min(0.85, max(0.45, 0.40 + 0.30 * (scale - 1.0))), 3)


def _lap_scores(imgs):
    """Sharpness score per image [B,H,W,C] (variance of the Laplacian of luminance). Works on torch or numpy."""
    g = imgs[..., :3].mean(-1)
    lap = g[:, 1:-1, 1:-1] * 4 - g[:, :-2, 1:-1] - g[:, 2:, 1:-1] - g[:, 1:-1, :-2] - g[:, 1:-1, 2:]
    flat = lap.reshape(lap.shape[0], -1)
    return (flat * flat).mean(-1) - flat.mean(-1) ** 2


def _select_latent(drafts, index):
    """Pick draft `index` (0-based) out of a batched H3 latent (video and audio streams together)."""
    lat = drafts["samples"]
    out = drafts.copy()
    if getattr(lat, "is_nested", False):
        video, audio = lat.unbind()
        i = min(max(index, 0), video.shape[0] - 1)
        out["samples"] = comfy.nested_tensor.NestedTensor((video[i:i + 1], audio[i:i + 1]))
    else:
        i = min(max(index, 0), lat.shape[0] - 1)
        out["samples"] = lat[i:i + 1]
    return out


def _stack_latents(lats):
    """Stack single-draft H3 latents into one batched latent (H3 cannot sample batches itself)."""
    if len(lats) == 1:
        return lats[0]
    out = lats[0].copy()
    if getattr(lats[0]["samples"], "is_nested", False):
        parts = [l["samples"].unbind() for l in lats]
        out["samples"] = comfy.nested_tensor.NestedTensor(
            (torch.cat([p[0] for p in parts], 0), torch.cat([p[1] for p in parts], 0)))
    else:
        out["samples"] = torch.cat([l["samples"] for l in lats], 0)
    return out


def _apply_lora(model, lora_name, strength):
    if lora_name != "None" and strength > 0:
        return comfy_nodes.LoraLoaderModelOnly().load_lora_model_only(model, lora_name, strength)[0]
    return model


def _refine_and_finish(m, vae, positive, lat, seed, scale, steps, strength, sampler_name, scheduler,
                       upscale_model, upscale_to_mp):
    if scale > 1.0:
        if strength <= 0:
            strength = _auto_strength(scale)
        lat = SatoDiveH3LatentUpscale().up(lat, scale, "bicubic")[0]
        lat = _sample(m, positive, lat, seed + 1, steps, 1.0, sampler_name, scheduler, start_sigma=strength)
    img = _fizgig("FizgigH3StillDecode").decode(vae=vae, samples=lat)[0]
    if upscale_model != "None":
        img = _pixel_upscale(img, upscale_model, upscale_to_mp)
    return img


class SatoDiveH3Generate:
    @classmethod
    def INPUT_TYPES(cls):
        sched = getattr(comfy.samplers, "SCHEDULER_NAMES", comfy.samplers.KSampler.SCHEDULERS)
        return {"required": {
            "model": ("MODEL",), "vae": ("VAE",), "positive": ("CONDITIONING",), "latent": ("LATENT",),
            "preset": (list(PRESETS.keys()) + ["Custom"], {"default": "Fast 2-pass"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
            "steps": ("INT", {"default": 20, "min": 1, "max": 200, "tooltip": "Custom preset only."}),
            "lora_strength": ("FLOAT", {"default": 0.38, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Custom preset only."}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "er_sde"}),
            "scheduler": (sched, {"default": "simple"}),
            "refine_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 2.5, "step": 0.05, "tooltip": "Custom: 1.0 = no refine pass."}),
            "refine_steps": ("INT", {"default": 10, "min": 1, "max": 100}),
            "refine_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.01, "tooltip": "Noise level (sigma) the refine pass starts from. 0 = auto (from refine_scale). Lower keeps more of the draft."}),
            "upscale_model": (["None"] + folder_paths.get_filename_list("upscale_models"), {"tooltip": "Optional pixel upscale (ESRGAN-style) applied after decode."}),
            "upscale_to_mp": (["native", "4", "6", "8", "12", "16"], {"default": "native", "tooltip": "native = the model's own factor; otherwise resize the result to about this many megapixels."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "SatoDive/H3"
    DESCRIPTION = ("Turbo LoRA + draft sampling + optional latent-upscale refine + one-frame still decode, in one node. "
                   "Presets: Fast 2-pass / Fast draft / Max quality / Custom.")

    def generate(self, model, vae, positive, latent, preset, seed, lora_name, steps, lora_strength,
                 sampler_name, scheduler, refine_scale, refine_steps, refine_strength,
                 upscale_model="None", upscale_to_mp="native"):
        if preset in PRESETS:
            lora_strength, steps, refine = PRESETS[preset]
            if refine:
                refine_scale, refine_steps, refine_strength = 1.5, 10, 0.0
        else:
            refine = refine_scale > 1.0
        m = _apply_lora(model, lora_name, lora_strength)
        lat = _sample(m, positive, latent, seed, steps, 1.0, sampler_name, scheduler)
        img = _refine_and_finish(m, vae, positive, lat, seed, refine_scale if refine else 1.0, refine_steps,
                                 refine_strength, sampler_name, scheduler, upscale_model, upscale_to_mp)
        return (img,)


class SatoDiveH3Draft:
    """Sample a batch of cheap drafts (use batch_size 3-8 on Prompt & Size), decode them for a preview grid, score them."""
    @classmethod
    def INPUT_TYPES(cls):
        sched = getattr(comfy.samplers, "SCHEDULER_NAMES", comfy.samplers.KSampler.SCHEDULERS)
        return {"required": {
            "model": ("MODEL",), "vae": ("VAE",), "positive": ("CONDITIONING",), "latent": ("LATENT",),
            "drafts": ("INT", {"default": 4, "min": 1, "max": 8, "tooltip": "How many drafts. Rendered one after another (H3 only supports batch size 1)."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
            "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
            "lora_strength": ("FLOAT", {"default": 0.38, "min": 0.0, "max": 2.0, "step": 0.01}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "er_sde"}),
            "scheduler": (sched, {"default": "simple"}),
        }}

    RETURN_TYPES = ("MODEL", "LATENT", "IMAGE", "INT")
    RETURN_NAMES = ("model", "drafts", "preview", "best")
    FUNCTION = "draft"
    CATEGORY = "SatoDive/H3"
    DESCRIPTION = ("Renders several drafts one at a time (seed, seed+1, ...). Preview shows them all; 'best' is the sharpest one (1-based). "
                   "Feed model/drafts/best into H3 Refine Winner - SatoDive.")

    def draft(self, model, vae, positive, latent, drafts, seed, lora_name, steps, lora_strength, sampler_name, scheduler):
        m = _apply_lora(model, lora_name, lora_strength)
        dec = _fizgig("FizgigH3StillDecode")
        lats, imgs = [], []
        for i in range(drafts):  # H3 supports batch size 1 only -> one draft at a time
            lat = _sample(m, positive, latent, seed + i, steps, 1.0, sampler_name, scheduler)
            lats.append(lat)
            imgs.append(dec.decode(vae=vae, samples=lat)[0])
        images = torch.cat(imgs, 0) if len(imgs) > 1 else imgs[0]
        best = int(_lap_scores(images).argmax()) + 1
        return (m, _stack_latents(lats), images, best)


class SatoDiveH3Refine:
    """Refine + upscale ONE draft. Change `pick` and only this node re-runs; the drafts stay cached upstream."""
    @classmethod
    def INPUT_TYPES(cls):
        sched = getattr(comfy.samplers, "SCHEDULER_NAMES", comfy.samplers.KSampler.SCHEDULERS)
        return {"required": {
            "model": ("MODEL",), "vae": ("VAE",), "positive": ("CONDITIONING",), "drafts": ("LATENT",),
            "pick": ("INT", {"default": 0, "min": 0, "max": 8, "tooltip": "1-based draft to refine. 0 = use the auto-picked sharpest."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "refine_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 2.5, "step": 0.05}),
            "refine_steps": ("INT", {"default": 10, "min": 1, "max": 100}),
            "refine_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.01, "tooltip": "0 = auto (from refine_scale)."}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "er_sde"}),
            "scheduler": (sched, {"default": "simple"}),
            "upscale_model": (["None"] + folder_paths.get_filename_list("upscale_models"),),
            "upscale_to_mp": (["native", "4", "6", "8", "12", "16"], {"default": "native"}),
        }, "optional": {"auto_pick": ("INT", {"forceInput": True})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "refine"
    CATEGORY = "SatoDive/H3"
    DESCRIPTION = "Refine (latent upscale + short re-sample) and optional pixel upscale of the chosen draft."

    def refine(self, model, vae, positive, drafts, pick, seed, refine_scale, refine_steps, refine_strength,
               sampler_name, scheduler, upscale_model, upscale_to_mp, auto_pick=None):
        idx = pick if pick > 0 else (auto_pick if auto_pick else 1)
        lat = _select_latent(drafts, idx - 1)
        img = _refine_and_finish(model, vae, positive, lat, seed, refine_scale, refine_steps, refine_strength,
                                 sampler_name, scheduler, upscale_model, upscale_to_mp)
        return (img,)



# ---------------------------------------------------------------------------
# SIMPLE MAIN NODE: one node, one sampling pass, you choose the exact resolution.
# ---------------------------------------------------------------------------


def _detail_refine(img, model_name, strength):
    """Add fine detail WITHOUT changing resolution: upscale with an ESRGAN-style model, shrink back to the
    original size (supersampling), then blend with the original by `strength` (0 = untouched, 1 = full effect)."""
    if model_name == "None" or strength <= 0:
        return img
    from comfy_extras import nodes_upscale_model as nu
    um = _run(nu.UpscaleModelLoader, model_name=model_name)[0]
    big = _run(nu.ImageUpscaleWithModel, upscale_model=um, image=img)[0]
    _, h, w, _ = img.shape
    small = comfy.utils.common_upscale(big.movedim(-1, 1), w, h, "area", "disabled").movedim(1, -1)
    small = small.to(device=img.device, dtype=img.dtype)[..., :img.shape[-1]]
    s = min(1.0, float(strength))
    return (img * (1.0 - s) + small * s).clamp(0.0, 1.0)


_COND_CACHE = {"key": None, "val": None}


def _tensor_hash(t):
    import hashlib
    return hashlib.sha1(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class SatoDiveH3Image:
    """Prompt + references + size + ONE sampling pass + decode. Output resolution = the size you choose."""
    @classmethod
    def INPUT_TYPES(cls):
        sched = getattr(comfy.samplers, "SCHEDULER_NAMES", comfy.samplers.KSampler.SCHEDULERS)
        ref_tip = "Reference image. Use <Picture N> in the prompt; N counts only connected slots, in slot order."
        opt = {"ref_image_%d" % i: ("IMAGE", {"tooltip": ref_tip}) for i in range(1, 10)}
        opt["ref_image_size"] = (["max", "match"], {"default": "max", "tooltip": "max = references at full quality (2048px short edge, best identity, SLOWER and heavier on VRAM). match = scaled down to the output area (much faster, weaker likeness)."})
        return {"required": {
            "model": ("MODEL",), "clip": ("CLIP",), "vae": ("VAE",),
            "prompt": ("STRING", {"multiline": True, "dynamic_prompts": True, "default": ""}),
            "size_mode": (["Aspect + megapixels", "Custom size"], {"default": "Aspect + megapixels", "tooltip": "Aspect + megapixels: pick a ratio and an area. Custom size: type the exact width and height."}),
            "aspect": (list(ASPECTS.keys()), {"default": "16:9", "tooltip": "Used only in 'Aspect + megapixels' mode."}),
            "megapixels": ("FLOAT", {"default": 3.0, "min": 0.25, "max": 16.0, "step": 0.25, "tooltip": "Used only in 'Aspect + megapixels' mode. This IS the final size, no hidden upscale."}),
            "width": ("INT", {"default": 1920, "min": 64, "max": 8192, "step": 1, "tooltip": "Used only in 'Custom size' mode."}),
            "height": ("INT", {"default": 1088, "min": 64, "max": 8192, "step": 1, "tooltip": "Used only in 'Custom size' mode."}),
            "exact_size": ("BOOLEAN", {"default": True, "tooltip": "The model works in multiples of 32. ON: the result is resized by the few leftover pixels so it is EXACTLY the size you typed. OFF: keep the native multiple-of-32 size."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 200, "tooltip": "Match your LoRA: a 4-step turbo LoRA needs ~4 steps (more is just slower). Other turbo LoRA: ~20. No LoRA: ~50."}),
            "lora_name": (["None"] + folder_paths.get_filename_list("loras"), {"tooltip": "Turbo LoRA (optional)."}),
            "lora_strength": ("FLOAT", {"default": 0.38, "min": 0.0, "max": 2.0, "step": 0.01}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "er_sde"}),
            "scheduler": (sched, {"default": "simple"}),
            "detail_model": (["None"] + folder_paths.get_filename_list("upscale_models"), {"tooltip": "Optional. Upscale model used ONLY to add fine detail (faces far away, textures). The resolution does NOT change. None = off."}),
            "detail_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "How much of the detail pass is mixed in. 0 = nothing, 1 = full. Lower keeps the image closer to the original."}),
        }, "optional": opt}

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "generate"
    CATEGORY = "SatoDive/H3"
    DESCRIPTION = ("One node, one pass: what you set is what you get. Pick aspect + megapixels or type an exact size. "
                   "No refine pass, no hidden upscale. Use 'H3 Final Size' afterwards only if you want to resize or upscale.")

    def generate(self, model, clip, vae, prompt, size_mode, aspect, megapixels, width, height, exact_size,
                 seed, steps, lora_name, lora_strength, sampler_name, scheduler, detail_model="None", detail_strength=0.7,
                 ref_image_size="max", **kw):
        snap = lambda v: max(64, int(round(v / 32.0)) * 32)
        if size_mode == "Custom size":
            want_w, want_h = int(width), int(height)
        else:
            aw, ah = ASPECTS[aspect]
            ratio = aw / ah
            hh = math.sqrt(float(megapixels) * 1_000_000 / ratio)
            want_w, want_h = int(round(hh * ratio)), int(round(hh))
        gen_w, gen_h = snap(want_w), snap(want_h)

        refs = {"ref_image_%d" % i: kw["ref_image_%d" % i] for i in range(1, 10) if kw.get("ref_image_%d" % i) is not None}
        # Re-encoding the prompt and references (big text encoder + VAE) is slow and, on small GPUs, forces model
        # swapping. If only the seed / steps / LoRA changed, reuse the previous conditioning.
        key = (id(clip), id(vae), prompt, gen_w, gen_h, ref_image_size,
               tuple((k, _tensor_hash(v)) for k, v in sorted(refs.items())))
        if _COND_CACHE["key"] == key:
            cond = _COND_CACHE["val"]
        else:
            if refs:
                cond = _run(_native("ref"), clip=clip, vae=vae, prompt=prompt, width=gen_w, height=gen_h, length=5,
                            ref_image_size=ref_image_size, ref_images=refs)[0]
            else:
                cond = _run(_native("i2v"), clip=clip, vae=vae, prompt=prompt, width=gen_w, height=gen_h, length=5)[0]
            _COND_CACHE["key"], _COND_CACHE["val"] = key, cond
        latent = _fizgig("FizgigH3StillLatent").make(width=gen_w, height=gen_h, batch_size=1)[0]

        m = _apply_lora(model, lora_name, lora_strength)
        lat = _sample(m, cond, latent, seed, steps, 1.0, sampler_name, scheduler)
        img = _fizgig("FizgigH3StillDecode").decode(vae=vae, samples=lat)[0]
        img = _detail_refine(img, detail_model, detail_strength)

        if exact_size and (img.shape[2], img.shape[1]) != (want_w, want_h):
            img = comfy.utils.common_upscale(img.movedim(-1, 1), want_w, want_h, "lanczos", "disabled").movedim(1, -1)
        return (img, int(img.shape[2]), int(img.shape[1]))


class SatoDiveH3FinalSize:
    """Optional last step: resize (or model-upscale) to an exact final resolution."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mode": (["Exact size", "Scale by factor", "Target megapixels", "Keep size (detail only)"], {"default": "Exact size", "tooltip": "Keep size (detail only): the upscale model adds detail, then the image is shrunk back to its current resolution."}),
            "width": ("INT", {"default": 3840, "min": 64, "max": 16384, "step": 1, "tooltip": "Exact size mode."}),
            "height": ("INT", {"default": 2160, "min": 64, "max": 16384, "step": 1, "tooltip": "Exact size mode."}),
            "factor": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.05, "tooltip": "Scale by factor mode."}),
            "megapixels": ("FLOAT", {"default": 8.0, "min": 0.25, "max": 64.0, "step": 0.25, "tooltip": "Target megapixels mode (keeps the aspect ratio)."}),
            "fit": (["stretch", "crop", "pad"], {"default": "crop", "tooltip": "Exact size mode when the aspect ratio differs: stretch, crop the overflow, or pad with black."}),
            "method": (["lanczos", "bicubic", "bilinear", "area"], {"default": "lanczos"}),
            "upscale_model": (["None"] + folder_paths.get_filename_list("upscale_models"), {"tooltip": "Optional ESRGAN-style model applied first (real detail). The result is then resized to your chosen size."}),
            "detail_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Keep size (detail only) mode: how much of the detail pass is mixed in."}),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "run"
    CATEGORY = "SatoDive/H3"
    DESCRIPTION = "Final resolution, exactly as you ask. Optional upscale model first, then one resize."

    def run(self, image, mode, width, height, factor, megapixels, fit, method, upscale_model, detail_strength=0.7):
        if mode == "Keep size (detail only)":
            out = _detail_refine(image, upscale_model, detail_strength)
            return (out, int(out.shape[2]), int(out.shape[1]))
        img = image
        if upscale_model != "None":
            from comfy_extras import nodes_upscale_model as nu
            um = _run(nu.UpscaleModelLoader, model_name=upscale_model)[0]
            img = _run(nu.ImageUpscaleWithModel, upscale_model=um, image=img)[0]
        _, h, w, _ = img.shape
        if mode == "Scale by factor":
            tw, th = max(8, int(round(w * factor))), max(8, int(round(h * factor)))
            crop = "disabled"
        elif mode == "Target megapixels":
            k = math.sqrt(float(megapixels) * 1e6 / (w * h))
            tw, th = max(8, int(round(w * k))), max(8, int(round(h * k)))
            crop = "disabled"
        else:
            tw, th = int(width), int(height)
            crop = "center" if fit == "crop" else "disabled"
        x = img.movedim(-1, 1)
        if mode == "Exact size" and fit == "pad":
            k = min(tw / w, th / h)
            iw, ih = max(1, int(round(w * k))), max(1, int(round(h * k)))
            x = comfy.utils.common_upscale(x, iw, ih, method, "disabled")
            canvas = torch.zeros((x.shape[0], x.shape[1], th, tw), dtype=x.dtype, device=x.device)
            oy, ox = (th - ih) // 2, (tw - iw) // 2
            canvas[:, :, oy:oy + ih, ox:ox + iw] = x
            x = canvas
        else:
            x = comfy.utils.common_upscale(x, tw, th, method, crop)
        out = x.movedim(1, -1)
        return (out, int(out.shape[2]), int(out.shape[1]))


NODE_CLASS_MAPPINGS = {"SatoDiveH3Size": SatoDiveH3Size, "SatoDiveH3LatentUpscale": SatoDiveH3LatentUpscale,
                       "SatoDiveH3Prompt": SatoDiveH3Prompt, "SatoDiveH3Generate": SatoDiveH3Generate,
                       "SatoDiveH3Draft": SatoDiveH3Draft, "SatoDiveH3Refine": SatoDiveH3Refine,
                       "SatoDiveH3Image": SatoDiveH3Image, "SatoDiveH3FinalSize": SatoDiveH3FinalSize}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SatoDiveH3Size": "H3 Size Presets - SatoDive",
    "SatoDiveH3LatentUpscale": "H3 Latent Upscale - SatoDive",
    "SatoDiveH3Prompt": "H3 Prompt & Size - SatoDive",
    "SatoDiveH3Generate": "H3 Image Generation - SatoDive",
    "SatoDiveH3Draft": "H3 Draft Grid - SatoDive",
    "SatoDiveH3Refine": "H3 Refine Winner - SatoDive",
    "SatoDiveH3Image": "H3 Image (Simple) - SatoDive",
    "SatoDiveH3FinalSize": "H3 Final Size - SatoDive"
}
