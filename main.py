#!/usr/bin/env python
import argparse
import datetime
import glob
import inspect
import os
import sys
from inspect import Parameter
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
import wandb
from natsort import natsorted
from omegaconf import OmegaConf
from packaging import version
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_only
from astropy.io import fits
from sgm.util import exists, instantiate_from_config, isheatmap
import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly")
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None")
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: the use_reentrant parameter")

MULTINODE_HACKS = True

def arcsinh_rgb(imgs, mode="CHW", m=0.03, clip=True):
    bands = ["g", "r", "z"]
    rgbscales = dict(g=(2, 6.0), r=(1, 3.4), z=(0, 2.2))

    I = 0
    for img, band in zip(imgs, bands):
        plane, scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)

    Q = 50
    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    I += (I == 0.) * 1e-9
    H, W = I.shape
    if mode == "HWC":
        rgb = np.zeros((H, W, 3), np.float32)
        for img, band in zip(imgs, bands):
            plane, scale = rgbscales[band]
            rgb[:, :, plane] = (img * scale + m) * fI / I
    elif mode == "CHW":
        rgb = np.zeros((3, H, W), np.float32)
        for img, band in zip(imgs, bands):
            plane, scale = rgbscales[band]
            rgb[plane] = (img * scale + m) * fI / I
    if clip:
        rgb = np.clip(rgb, 0, 1)
    return rgb


def scale(img, Q=2000):
    return np.arcsinh(img * Q) / np.sqrt(Q)


def default_trainer_args():
    argspec = dict(inspect.signature(Trainer.__init__).parameters)
    argspec.pop("self")
    default_args = {
        param: argspec[param].default
        for param in argspec
        if argspec[param].default != Parameter.empty
    }
    return default_args


class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency,
        max_images,
        clamp=False,
        increase_log_steps=False,
        rescale=True,
        disabled=False,
        log_on_batch_idx=False,
        log_first_step=False,
        log_images_kwargs=None,
        log_before_first_step=False,
        enable_autocast=True,
    ):
        super().__init__()
        self.enable_autocast = enable_autocast
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_before_first_step = log_before_first_step
        self.last_val_batch = None
        self.last_val_batch_idx = None

    @rank_zero_only
    def log_local(
        self,
        save_dir,
        split,
        images,
        global_step,
        current_epoch,
        batch_idx,
        pl_module: Union[None, pl.LightningModule] = None,
    ):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            if isheatmap(images[k]):
                pass
            else:
                grid = torchvision.utils.make_grid(images[k], nrow=4, padding=0)
                if self.rescale:
                    grid = (grid + 1.0) / 2.0
                grid = grid.numpy()
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.fits".format(
                    k, global_step, current_epoch, batch_idx
                )
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                img = np.array(grid)
                fits.writeto(path, img, overwrite=True)

                if exists(pl_module):
                    assert isinstance(
                        pl_module.logger, WandbLogger
                    ), "logger_log_image currently only supports WandbLogger"

                    if img.shape[0] == 3:
                        transformed_img = arcsinh_rgb(img, mode="CHW").transpose(1, 2, 0)
                    else:
                        middle_slice = img[img.shape[0] // 2 - 1:img.shape[0] // 2 + 2, :, :]
                        transformed_img = scale(middle_slice).transpose(1, 2, 0)
                    pl_module.logger.log_image(
                        key=f"{split}/{k}",
                        images=[transformed_img],
                        step=pl_module.global_step,
                    )

    @rank_zero_only
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (
            self.check_frequency(check_idx)
            and hasattr(pl_module, "log_images")
            and callable(pl_module.log_images)
            and self.max_images > 0
        ):
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            gpu_autocast_kwargs = {
                "enabled": self.enable_autocast,
                "dtype": torch.get_autocast_gpu_dtype(),
                "cache_enabled": torch.is_autocast_cache_enabled(),
            }
            with torch.no_grad(), torch.amp.autocast('cuda', **gpu_autocast_kwargs):
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                if not isheatmap(images[k]):
                    images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().float().cpu()

            self.log_local(
                pl_module.logger.save_dir,
                split,
                images,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
                pl_module=pl_module if isinstance(pl_module.logger, WandbLogger) else None,
            )

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
            check_idx > 0 or self.log_first_step
        ):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.log_before_first_step and pl_module.global_step == 0:
            print(f"{self.__class__.__name__}: logging before training")
            self.log_img(pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, "calibrate_grad_norm"):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

        self.last_val_batch = batch
        self.last_val_batch_idx = batch_idx

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log the last validation batch image at the end of each epoch regardless of frequency."""
        if self.disabled or pl_module.global_step <= 0 or self.last_val_batch is None:
            return

        if hasattr(pl_module, "log_images") and callable(pl_module.log_images) and self.max_images > 0:
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            gpu_autocast_kwargs = {
                "enabled": self.enable_autocast,
                "dtype": torch.get_autocast_gpu_dtype(),
                "cache_enabled": torch.is_autocast_cache_enabled(),
            }
            with torch.no_grad(), torch.amp.autocast('cuda', **gpu_autocast_kwargs):
                images = pl_module.log_images(
                    self.last_val_batch, split="val_last", **self.log_images_kwargs
                )

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                if not isheatmap(images[k]):
                    images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().float().cpu()

            self.log_local(
                pl_module.logger.save_dir,
                "val_last",
                images,
                pl_module.global_step,
                pl_module.current_epoch,
                self.last_val_batch_idx,
                pl_module=pl_module if isinstance(pl_module.logger, WandbLogger) else None,
            )

            if is_train:
                pl_module.train()

            self.last_val_batch = None
            self.last_val_batch_idx = None


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-n", "--name", type=str, const=True, default="", nargs="?", help="postfix for logdir")
    parser.add_argument("--no_date", type=str2bool, nargs="?", const=True, default=False, help="if True, skip date in logdir name")
    parser.add_argument("-r", "--resume", type=str, const=True, default="", nargs="?", help="resume from logdir or checkpoint in logdir")
    parser.add_argument("--tags", type=str, default="", help="wandb tags, comma-separated, e.g. 'run1,test'")
    parser.add_argument("-b", "--base", nargs="*", metavar="base_config.yaml",
                        help="paths to base configs, merged left-to-right",
                        default=["configs/generation/astroIR_SBn.yaml"])
    parser.add_argument("-t", "--train", type=str2bool, const=True, default=True, nargs="?", help="train")
    parser.add_argument("--no-test", type=str2bool, const=True, default=False, nargs="?", help="disable test")
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument("-d", "--debug", type=str2bool, nargs="?", const=True, default=False, help="enable post-mortem debugging")
    parser.add_argument("-s", "--seed", type=int, default=1024, help="seed for seed_everything")
    parser.add_argument("-f", "--postfix", type=str, default="", help="post-postfix for default name")
    parser.add_argument("-w", "--projectname", type=str, default="DeepAstre")
    parser.add_argument("-l", "--logdir", type=str, default="logs", help="directory for logging")
    parser.add_argument("--scale_lr", type=str2bool, nargs="?", const=True, default=False, help="scale base-lr by ngpu * batch_size * n_accumulate")
    parser.add_argument("--legacy_naming", type=str2bool, nargs="?", const=True, default=False, help="name run based on config file name")
    parser.add_argument("--enable_tf32", type=str2bool, nargs="?", const=True, default=True, help="enable TensorFloat32 for matmuls and cuDNN")
    parser.add_argument("--startup", type=str, default=None, help="startup time from distributed script")
    parser.add_argument("--wandb", type=str2bool, nargs="?", const=True, default=True, help="log to wandb")
    parser.add_argument("--no_base_name", type=str2bool, nargs="?", const=True, default=False, help="omit config name from logdir")
    if version.parse(torch.__version__) >= version.parse("2.0.0"):
        parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="single checkpoint file to resume from")

    default_args = default_trainer_args()
    for key in default_args:
        parser.add_argument("--" + key, default=default_args[key])
    return parser


@rank_zero_only
def init_wandb(save_dir, opt, config, group_name, name_str):
    print(f"setting WANDB_DIR to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = save_dir
    if opt.debug:
        wandb.init(project=opt.projectname, mode="offline", group=group_name)
    else:
        wandb.init(
            project=opt.projectname,
            config=config,
            settings=wandb.Settings(code_dir="./sgm"),
            group=group_name,
            name=name_str,
            mode="offline",
        )


class SetupCallback(Callback):
    def __init__(
        self,
        resume,
        now,
        logdir,
        ckptdir,
        cfgdir,
        config,
        lightning_config,
        debug,
        ckpt_name=None,
    ):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.debug = debug
        self.ckpt_name = ckpt_name
        self.copy_from = os.path.join(os.path.dirname(__file__), "sgm")

    def on_exception(self, trainer: pl.Trainer, pl_module, exception):
        if not self.debug and trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, self.ckpt_name if self.ckpt_name else "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if self.copy_from and os.path.isdir(self.copy_from):
                import shutil
                copy_dst = os.path.join(self.logdir, "sgm")
                shutil.copytree(self.copy_from, copy_dst, dirs_exist_ok=True)
                print(f"Copied {self.copy_from} -> {copy_dst}")

            if "callbacks" in self.lightning_config:
                if "metrics_over_trainsteps_checkpoint" in self.lightning_config["callbacks"]:
                    os.makedirs(
                        os.path.join(self.ckptdir, "trainstep_checkpoints"),
                        exist_ok=True,
                    )
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            if MULTINODE_HACKS:
                import time
                time.sleep(5)
            OmegaConf.save(self.config, os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(
                OmegaConf.create({"lightning": self.lightning_config}),
                os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)),
            )
        else:
            if not MULTINODE_HACKS and not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


def get_checkpoint_name(logdir):
    ckpt = os.path.join(logdir, "checkpoints", "last**.ckpt")
    ckpt = natsorted(glob.glob(ckpt))
    print('available "last" checkpoints:')
    print(ckpt)
    if len(ckpt) > 1:
        print("got most recent checkpoint")
        ckpt = sorted(ckpt, key=lambda x: os.path.getmtime(x))[-1]
        print(f"Most recent ckpt is {ckpt}")
        with open(os.path.join(logdir, "most_recent_ckpt.txt"), "w") as f:
            f.write(ckpt + "\n")
        try:
            version = int(ckpt.split("/")[-1].split("-v")[-1].split(".")[0])
        except Exception as e:
            print("version confusion but not bad")
            print(e)
            version = 1
    else:
        ckpt = ckpt[0]
        version = 1
    melk_ckpt_name = f"last-v{version}.ckpt"
    print(f"Current melk ckpt name: {melk_ckpt_name}")
    return ckpt, melk_ckpt_name


class CheckpointLinker(Callback):
    """Hard-links best.ckpt to the current best checkpoint every link_interval steps and each epoch end."""

    def __init__(self, link_interval=1000):
        super().__init__()
        self.link_interval = link_interval

    def _update_best_link(self, trainer):
        src = trainer.checkpoint_callback.best_model_path
        if not src or not os.path.exists(src):
            return
        dst = os.path.join(os.path.dirname(src), "best.ckpt")
        if os.path.lexists(dst):
            os.remove(dst)
        os.link(src, dst)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.link_interval == 0:
            self._update_best_link(trainer)

    def on_train_epoch_end(self, trainer, pl_module):
        self._update_best_link(trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._update_best_link(trainer)


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both. "
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    melk_ckpt_name = None
    name = None
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
            _, melk_ckpt_name = get_checkpoint_name(logdir)
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt, melk_ckpt_name = get_checkpoint_name(logdir)

        print("#" * 100)
        print(f'Resuming from checkpoint "{ckpt}"')
        print("#" * 100)

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            if opt.no_base_name:
                name = ""
            else:
                if opt.legacy_naming:
                    cfg_fname = os.path.split(opt.base[0])[-1]
                    cfg_name = os.path.splitext(cfg_fname)[0]
                else:
                    assert "configs" in os.path.split(opt.base[0])[0], os.path.split(opt.base[0])[0]
                    cfg_path = os.path.split(opt.base[0])[0].split(os.sep)[
                        os.path.split(opt.base[0])[0].split(os.sep).index("configs") + 1:
                    ]
                    cfg_name = os.path.splitext(os.path.split(opt.base[0])[-1])[0]
                    cfg_name = "-".join(cfg_path) + f"-{cfg_name}"
                name = "_" + cfg_name
        else:
            name = ""
        if not opt.no_date:
            nowname = now + name + opt.postfix
        else:
            nowname = name + opt.postfix
            if nowname.startswith("_"):
                nowname = nowname[1:]
        logdir = os.path.join(opt.logdir, nowname)
        print(f"LOGDIR: {logdir}")

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed, workers=True)

    if opt.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Enabling TF32 for PyTorch {torch.__version__}")
    else:
        print(f"Using default TF32 settings for PyTorch {torch.__version__}:")
        print(f"torch.backends.cuda.matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}")
        print(f"torch.backends.cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}")

    try:
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())

        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        trainer_config["accelerator"] = "gpu"

        standard_args = default_trainer_args()
        for k in standard_args:
            if getattr(opt, k) != standard_args[k]:
                trainer_config[k] = getattr(opt, k)

        ckpt_resume_path = opt.resume_from_checkpoint
        if not "devices" in trainer_config and trainer_config["accelerator"] != "gpu":
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["devices"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        model = instantiate_from_config(config.model)
        trainer_kwargs = dict()

        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                    "project": opt.projectname,
                    "log_model": False,
                    "tags": opt.tags.split(",") if opt.tags else None,
                },
            },
            "csv": {
                "target": "pytorch_lightning.loggers.CSVLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                },
            },
        }
        default_logger_cfg = default_logger_cfgs["wandb" if opt.wandb else "csv"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:03}-{step:09}",
                "verbose": True,
                "save_last": True,
                "save_top_k": 3,
            },
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")

        default_strategy_config = {
            "target": "pytorch_lightning.strategies.DDPStrategy",
            "params": {"find_unused_parameters": True},
        }

        if "strategy" in lightning_config:
            strategy_cfg = lightning_config.strategy
        else:
            strategy_cfg = OmegaConf.create()
            default_strategy_config["params"] = {"find_unused_parameters": True}
        strategy_cfg = OmegaConf.merge(default_strategy_config, strategy_cfg)
        print(f"strategy config: \n ++++++++++++++ \n {strategy_cfg} \n ++++++++++++++ ")
        trainer_kwargs["strategy"] = instantiate_from_config(strategy_cfg)

        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                    "debug": opt.debug,
                    "ckpt_name": melk_ckpt_name,
                },
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {"batch_frequency": 1000, "max_images": 4, "clamp": True},
            },
            "learning_rate_logger": {
                "target": "pytorch_lightning.callbacks.LearningRateMonitor",
                "params": {"logging_interval": "step"},
            },
            "checkpoint_linker": {
                "target": "__main__.CheckpointLinker",
                "params": {
                    "link_interval": lightning_config.callbacks.checkpoint_linker.params.link_interval
                },
            },
        }
        if version.parse(pl.__version__) >= version.parse("1.4.0"):
            default_callbacks_cfg.update({"checkpoint_callback": modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if "metrics_over_trainsteps_checkpoint" in callbacks_cfg:
            print("Caution: Saving checkpoints every n train steps without deleting. This might require some free space.")
            default_metrics_over_trainsteps_ckpt_dict = {
                "metrics_over_trainsteps_checkpoint": {
                    "target": "pytorch_lightning.callbacks.ModelCheckpoint",
                    "params": {
                        "dirpath": os.path.join(ckptdir, "trainstep_checkpoints"),
                        "filename": "{epoch:06}-{step:09}",
                        "verbose": True,
                        "save_top_k": -1,
                        "every_n_train_steps": 10000,
                        "save_weights_only": True,
                    },
                }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if "ignore_keys_callback" in callbacks_cfg and ckpt_resume_path is not None:
            callbacks_cfg.ignore_keys_callback.params["ckpt_path"] = ckpt_resume_path
        elif "ignore_keys_callback" in callbacks_cfg:
            del callbacks_cfg["ignore_keys_callback"]

        trainer_kwargs["callbacks"] = [
            instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
        ]
        if not "plugins" in trainer_kwargs:
            trainer_kwargs["plugins"] = list()

        trainer_opt = vars(trainer_opt)
        trainer_kwargs = {
            key: val for key, val in trainer_kwargs.items() if key not in trainer_opt
        }

        trainer = Trainer(**trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir

        data = instantiate_from_config(config.data)
        data.prepare_data()

        print("#### Data #####")
        try:
            for k in data.datasets:
                print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
        except:
            print("datasets not yet initialized.")

        if "batch_size" in config.data.params:
            bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        else:
            bs, base_lr = (
                config.data.params.train.loader.batch_size,
                config.model.base_learning_rate,
            )
        if not cpu:
            ngpu = len(lightning_config.trainer.devices.strip(",").split(","))
        else:
            ngpu = 1
        if "accumulate_grad_batches" in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr
                )
            )
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")

        def melk(*args, **kwargs):
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, melk_ckpt_name if melk_ckpt_name else "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        import signal
        signal.signal(signal.SIGUSR1, melk)

        if opt.train:
            try:
                trainer.fit(model, data, ckpt_path=ckpt_resume_path)
            except Exception:
                if not opt.debug:
                    melk()
                raise

    except RuntimeError as err:
        raise err
    finally:
        if opt.wandb:
            wandb.finish()
