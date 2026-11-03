from transformers import Trainer
from typing import List, Optional
import src.custom_utils as utils
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from torch.utils.data import Sampler
from transformers.trainer_utils import SaveStrategy
from peft import PeftModel
from transformers.integrations import is_deepspeed_zero3_enabled
import deepspeed
from contextlib import nullcontext
from pathlib import Path
import torch.distributed as dist
from src.models import MLLMPreTrainedModel

logger = utils.get_logger()


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l)
                                 for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(
        *[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(
        mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(
        lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i: i + megabatch_size]
                      for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i: i + megabatch_size]
                        for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i: i + megabatch_size].tolist()
                   for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i],
                          reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(
        megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lens: Optional[List[int]],
        generator=None
    ):
        self.batch_size = batch_size
        self.world_size = world_size
        self.lens = lens
        self.generator = generator

    def __len__(self):
        return len(self.lens)

    def __iter__(self):
        indices = get_modality_length_grouped_indices(
            self.lens, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class CustomTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._extra_losses: dict[str, torch.Tensor] = {}
        self._extra_losses_counts: dict[str, int] = {}
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if key.startswith("loss_") and value is not None:
                    if key not in self._extra_losses:
                        self._extra_losses[key] = torch.tensor(
                            0.0, device=loss.device)
                        self._extra_losses_counts[key] = 0

                    # Only accumulate non-zero values (0 is a placeholder)
                    if isinstance(value, torch.Tensor):
                        val = value.detach()
                    else:
                        val = torch.tensor(value, device=loss.device)

                    if val.item() != 0.0:
                        self._extra_losses[key] = self._extra_losses[key].to(
                            loss.device) + val
                        self._extra_losses_counts[key] += 1

        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if self.args.train_proj_only:
            ctx = nullcontext()
            if is_deepspeed_zero3_enabled():
                ctx = deepspeed.zero.GatheredParameters(self.model.projector.parameters(), modifier_rank=0)
            with ctx:
                Path(output_dir).mkdir(mode=0o775, parents=True, exist_ok=True)
                filename = Path(output_dir) / 'projector.pt'
                torch.save(self.model.projector.state_dict(), filename)
                if hasattr(self.model, 'tgt_projector') and not isinstance(self.model.tgt_projector, nn.Identity):
                    filename = Path(output_dir) / 'tgt_projector.pt'
                    torch.save(self.model.tgt_projector.state_dict(), filename)
                if hasattr(self.model, 'logit_scale') and isinstance(self.model.logit_scale, nn.Parameter) and (self.model.logit_scale.requires_grad):
                    filename = Path(output_dir) / 'logit_scale.pt'
                    torch.save(self.model.logit_scale.data, filename)
            self.model.config.save_pretrained(output_dir)
        else:
            super().save_model(output_dir, _internal_call)
        self.processing_class.save_pretrained(output_dir)
        if isinstance(self.model, PeftModel):
            self.model.config.save_pretrained(output_dir)

    def get_decay_parameter_names(self, model) -> List[str]:
        """
        Get all parameter names that weight decay will be applied to.

        This function filters out parameters in two ways:
        1. By layer type (instances of layers specified in ALL_LAYERNORM_LAYERS)
        2. By parameter name patterns (containing 'bias', 'layernorm', or 'rmsnorm')
        """
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS, [
                                               "bias", "layernorm", "rmsnorm", "lora_", "logit_scale"])
        return decay_parameters

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            if self.args.proj_learning_rate is None and self.args.tgt_proj_learning_rate is None and self.args.logit_scale_lr is None:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
                self.opt_param_grps_names = []

            else:
                if isinstance(self.model, PeftModel):
                    def is_proj(x): return False if self.args.proj_learning_rate is None else (
                        '.projector.' in x)
                    def is_tgt_proj(x): return False if self.args.tgt_proj_learning_rate is None else (
                        '.tgt_projector.' in x)
                else:
                    def is_proj(x): return False if self.args.proj_learning_rate is None else x.startswith(
                        'projector.')
                    def is_tgt_proj(x): return False if self.args.tgt_proj_learning_rate is None else x.startswith(
                        'tgt_projector.')

                def is_logit_scale(x): return False if self.args.logit_scale_lr is None else x.endswith(
                    'logit_scale')

                # Base parameters (LLM)
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and not is_proj(n) and not is_tgt_proj(n) and not is_logit_scale(n))
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and not is_proj(n) and not is_tgt_proj(n) and not is_logit_scale(n))
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.learning_rate
                    },
                ]

                self.opt_param_grps_names = ['', '']

                # Add projector parameters with custom learning rate
                if self.args.proj_learning_rate is not None:
                    proj_lr = self.args.proj_learning_rate
                    optimizer_grouped_parameters.extend([
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and is_proj(n))
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": proj_lr
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and is_proj(n))
                            ],
                            "weight_decay": 0.0,
                            "lr": proj_lr
                        }
                    ])
                    self.opt_param_grps_names.extend(['proj', 'proj'])
                    logger.info(f"Projector learning rate: {proj_lr}")

                # Add target projector parameters with custom learning rate
                if self.args.tgt_proj_learning_rate is not None:
                    tgt_proj_lr = self.args.tgt_proj_learning_rate
                    optimizer_grouped_parameters.extend([
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and is_tgt_proj(n))
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": tgt_proj_lr
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and is_tgt_proj(n))
                            ],
                            "weight_decay": 0.0,
                            "lr": tgt_proj_lr
                        }
                    ])
                    self.opt_param_grps_names.extend(['tgt_proj', 'tgt_proj'])
                    logger.info(
                        f"Target projector learning rate: {tgt_proj_lr}")

                # Add logit_scale parameter with custom learning rate (no weight decay —
                # it is a temperature scalar/vector, not a weight matrix)
                if self.args.logit_scale_lr is not None:
                    logit_scale_lr = self.args.logit_scale_lr
                    optimizer_grouped_parameters.append({
                        "params": [
                            p for n, p in opt_model.named_parameters() if (p.requires_grad and is_logit_scale(n))
                        ],
                        "weight_decay": 0.0,
                        "lr": logit_scale_lr
                    })
                    self.opt_param_grps_names.append('logit_scale')
                    logger.info(f"Logit scale learning rate: {logit_scale_lr}")

                # Filter out empty parameter groups while keeping names in sync
                filtered_groups = []
                filtered_names = []
                for i, grp in enumerate(optimizer_grouped_parameters):
                    if len(grp['params']) > 0:
                        filtered_groups.append(grp)
                        filtered_names.append(self.opt_param_grps_names[i])
                optimizer_grouped_parameters = filtered_groups
                self.opt_param_grps_names = filtered_names

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                    self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop(
                    "optimizer_dict")

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs)

            if "bitsandbytes" in str(optimizer_cls) and optimizer_kwargs.get("optim_bits", None) == 8:
                raise ValueError(
                    "bitsandbytes not currently supported on this custom Trainer")

        return self.optimizer

    def _get_train_sampler(self, train_dataset: Optional[Dataset] = None) -> Optional[torch.utils.data.Sampler]:
        if self.args.group_by_mod_lens:
            lens = self.train_dataset.mod_lens
            return LengthGroupedSampler(
                batch_size=self.args.train_batch_size,
                world_size=utils.get_world_size() * self.args.gradient_accumulation_steps,
                lens=lens,
            )
        else:
            return super()._get_train_sampler(train_dataset)

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        dist.barrier()
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            # if is_torch_xla_available():
            #     xm.mark_step()

            logs: dict[str, float] = {}

            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)

            for key in list(self._extra_losses.keys()):
                extra_loss = self._extra_losses[key]
                extra_count = self._extra_losses_counts[key]

                gathered_loss = self._nested_gather(extra_loss).sum().item()
                gathered_count = self._nested_gather(
                    torch.tensor(
                        extra_count, device=extra_loss.device, dtype=torch.long)
                ).sum().item()

                if gathered_count > 0:
                    logs[key] = round(gathered_loss / gathered_count, 4)

                self._extra_losses[key] = torch.tensor(
                    0.0, device=extra_loss.device)
                self._extra_losses_counts[key] = 0

            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(
                    grad_norm, torch.Tensor) else grad_norm

            if self.args.proj_learning_rate is not None or self.args.tgt_proj_learning_rate is not None or self.args.logit_scale_lr is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    last_lrs = [grp['lr']
                                for grp in self.optimizer.param_groups]
                else:
                    last_lrs = self.lr_scheduler.get_last_lr()

                assert len(self.opt_param_grps_names) == len(
                    last_lrs), f"Expected {len(self.opt_param_grps_names)} learning rates, got {len(last_lrs)}"
                for i, (name, lr) in enumerate(zip(self.opt_param_grps_names, last_lrs)):
                    if name:
                        logs[f"learning_rate_{name}"] = lr

            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            if utils.get_rank() == 0 and hasattr(model, 'logit_scale') and isinstance(model.logit_scale, nn.Parameter):
                logit_scale = model.logit_scale.data
                for i in range(len(logit_scale)):
                    logs[f"logit_scale_{i}"] = logit_scale[i].item()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(
                metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control)
