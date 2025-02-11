import sys
import time
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union, List
from warnings import warn

import torch
from omegaconf import DictConfig, ListConfig

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from torchtune import config, modules, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import padded_collate_packed
from torchtune.datasets import ConcatDataset
from torchtune.modules.common_utils import slice_str_to_array
from torchtune.modules.early_exit_loss import early_exit_loss, EarlyExitCurriculum
from torchtune.modules.layer_dropout import prepare_layer_dropout
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import DummyProfiler, PROFILER_KEY
from torchtune.training.lr_schedulers import get_lr

from tqdm import tqdm

log = utils.get_logger("DEBUG")

class EarlyExitFinetuneRecipeSingleDevice(FTRecipeInterface):
    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        if self._dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        if self._log_peak_memory_stats and self._device.type != "cuda":
            log.info(
                "log_peak_memory_stats was set to True, however, training uses cpu. Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        # Training cfg
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._optimizer_in_bwd = cfg.optimizer_in_bwd
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)

        # Optimizer in backward is not compatible with gradient accumulation or gradient clipping
        if self._optimizer_in_bwd:
            if self._clip_grad_norm is not None:
                raise RuntimeError(
                    "Gradient clipping is not supported with optimizer in bwd."
                    "Please set clip_grad_norm=None, or optimizer_in_bwd=False."
                )
            if self._gradient_accumulation_steps > 1:
                raise RuntimeError(
                    "Gradient accumulation is not supported with optimizer in bwd."
                    "Please set gradient_accumulation_steps=1, or optimizer_in_bwd=False."
                )

        # activation checkpointing/offloading
        self._enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )
        self._enable_activation_offloading = cfg.get(
            "enable_activation_offloading", False
        )
        if self._enable_activation_offloading:
            if self._device.type != "cuda":
                raise RuntimeError(
                    "enable_activation_offloading should only be True when training on CUDA"
                )
            if not self._enable_activation_checkpointing:
                raise RuntimeError(
                    "enable_activation_offloading should only be True when enable_activation_checkpointing is True"
                )
        elif (
            self._enable_activation_checkpointing
            and cfg.checkpointer.model_type != "LLAMA3_VISION"
        ):
            log.info(
                "Hint: enable_activation_checkpointing is True, but enable_activation_offloading isn't. "
                "Enabling activation offloading should reduce memory further."
            )

        # Recipe state properties
        self.seed = training.set_seed(seed=cfg.seed)
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0

        # Early Exit Properties
        cfg_early_exit_loss = cfg.get("early_exit_loss", None)
        if cfg_early_exit_loss:
            self._do_early_exit_loss = True
            self._early_exit_loss_scale = cfg_early_exit_loss.get("scale", 1.0)
            self._early_exit_loss_scale_type = _get_component_from_path(
                cfg_early_exit_loss.get(
                    "scale_fn", "torchtune.modules.early_exit_loss.sum_l_loss_scale"
                )
            )
        else:
            self._do_early_exit_loss = False
            self._early_exit_loss_scale = None
            self._early_exit_loss_scale_type = None

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            should_load_recipe_state=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

        if self._resume_from_checkpoint:
            self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        try:
            self.epochs_run = ckpt_dict[training.EPOCHS_KEY]

            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[training.SEED_KEY]
            if self.max_steps_per_epoch != ckpt_dict[training.MAX_STEPS_KEY]:
                warn(
                    message=(
                        "Config value for max_steps_per_epoch does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.MAX_STEPS_KEY]}"
                    )
                )
                self.max_steps_per_epoch = ckpt_dict[training.MAX_STEPS_KEY]

            if self.total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]:
                warn(
                    message=(
                        "Config value for total_epochs does not match the checkpoint value, "
                        f"using the config value: {self.total_epochs}"
                    )
                )

        except KeyError as e:
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state. "
                "Are you sure you passed in the right recipe checkpoint?"
            ) from e

    def setup(self, cfg: DictConfig) -> None:
        self._metric_logger = config.instantiate(cfg.metric_logger)
        self._metric_logger.log_config(cfg)

        checkpoint_dict = self.load_checkpoint(cfg.checkpointer)

        self._compile = cfg.get("compile", False)
        if cfg.device == "npu" and cfg.compile:
            raise ValueError(
                "NPU does not support model compilation. Please set `compile: False` in the config."
            )

        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            compile_model=self._compile,
            model_state_dict=checkpoint_dict[training.MODEL_KEY],
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            optimizer_in_bwd=cfg.optimizer_in_bwd,
            opt_state_dict=(
                checkpoint_dict[training.OPT_KEY] if self._resume_from_checkpoint else None
            ),
        )

        self._loss_fn = config.instantiate(cfg.loss)

        if self._compile:
            training.compile_loss(self._loss_fn)

        if self._loss_fn.__class__.__name__ == "CEWithChunkedOutputLoss":
            self._model.set_num_output_chunks(self._loss_fn.num_output_chunks)

        collate_name = cfg.get("collate_fn", "torchtune.data.padded_collate_sft")
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
            collate_fn=collate_name,
        )

        self._steps_per_epoch = (
            len(self._dataloader) // self._gradient_accumulation_steps
        )
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
        self.global_step = self.epochs_run * self._steps_per_epoch

        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

        self.ignore_labels_cache = torch.full(
            (cfg.batch_size, 1), self._loss_fn.ignore_index, device=self._device
        )

        (
            self._do_output_hidden_states,
            self._early_exit_loss_curriculum,
        ) = self._setup_early_exit_loss(cfg.get("early_exit_loss", None))

        cfg_layer_dropout = cfg.get("layer_dropout", None)
        if cfg_layer_dropout:
            prepare_layer_dropout(
                self._model.layers,
                prob_max=cfg_layer_dropout.get("prob", 0.0),
                prob_layer_scale=cfg_layer_dropout.get("layers_scale", "uniform"),
                layers_str=cfg_layer_dropout.get("layers", ":"),
                disable_on_eval=cfg_layer_dropout.get("disable_on_eval", True),
            )

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
        else:
            assert (
                cfg_profiler.get("_component_")
                == "torchtune.training.setup_torch_profiler"
            ), "Only torch profiler supported currently"

        profiler, profiler_cfg = config.instantiate(cfg_profiler)

        log.info(f" Profiler config after instantiation: {profiler_cfg}")

        self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
        if profiler_cfg["enabled"]:
            self.profiler_wait_steps = profiler_cfg["wait_steps"]
            self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
            self.profiler_active_steps = profiler_cfg["active_steps"]

        return profiler

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        compile_model: bool,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)

        if compile_model:
            training.compile_model(model)

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        model.load_state_dict(model_state_dict)
        training.validate_expected_param_dtype(
            model.named_parameters(), dtype=self._dtype
        )

        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading
        )

        log.info(f"Model is initialized with precision {self._dtype}.")

        if self._device.type != "cpu":
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        return model

    def _setup_optimizer(
        self,
        cfg_optimizer: DictConfig,
        optimizer_in_bwd: bool = False,
        opt_state_dict: Optional[Dict[str, Any]] = None,
    ) -> Optional[Optimizer]:
        if optimizer_in_bwd:
            optim_dict = {
                p: config.instantiate(cfg_optimizer, [p])
                for p in self._model.parameters()
            }
            training.register_optim_in_bwd_hooks(
                model=self._model, optim_dict=optim_dict
            )
            self._optim_ckpt_wrapper = training.create_optim_in_bwd_wrapper(
                model=self._model, optim_dict=optim_dict
            )
            if opt_state_dict is not None:
                try:
                    self._optim_ckpt_wrapper.load_state_dict(opt_state_dict)
                except BaseException as e:
                    raise RuntimeError(
                        "Failed loading in-backward optimizer checkpoints."
                        "Please make sure run being restored from was using in-backward optimizer."
                    ) from e
            log.info("In-backward optimizers are set up.")
            return None
        else:
            optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
            if opt_state_dict:
                optimizer.load_state_dict(opt_state_dict)
            log.info("Optimizer is initialized.")
            return optimizer

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
        collate_fn: str,
    ) -> Tuple[DistributedSampler, DataLoader]:
        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
            packed = getattr(ds, "packed", False)
        else:
            ds = config.instantiate(cfg_dataset, self._tokenizer)
            packed = cfg_dataset.get("packed", False)

        if "left_pad_sequence" in collate_fn:
            raise RuntimeError("left_pad_sequence collator is only for inference.")
        collate_fn = _get_component_from_path(collate_fn)

        sampler = DistributedSampler(
            ds,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            seed=0,
        )
        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            collate_fn=(
                partial(
                    collate_fn,
                    padding_idx=self._tokenizer.pad_id,
                    ignore_idx=self._loss_fn.ignore_index,
                )
                if not packed
                else padded_collate_packed
            ),
        )

        log.info("Dataset and Sampler are initialized.")
        return sampler, dataloader

    def _setup_early_exit_loss(
        self,
        cfg_early_exit_loss: DictConfig,
    ) -> Tuple[List[bool], EarlyExitCurriculum]:
        do_output_hidden_states = None
        early_exit_loss_curriculum = None

        if cfg_early_exit_loss:
            assert (
                hasattr(self._loss_fn, "reduction")
                and self._loss_fn.reduction == "mean"
            ), "Currently early exit loss is only implemented for loss functions that apply a mean reduction."

            do_output_hidden_states = slice_str_to_array(
                cfg_early_exit_loss.get("layers", ":"), len(self._model.layers)
            )
            train_last_layer = cfg_early_exit_loss.get("include_last_layer", True)
            verbose = cfg_early_exit_loss.get("verbose", False)

            early_exit_loss_curriculum = cfg_early_exit_loss.get("curriculum", None)
            if early_exit_loss_curriculum:
                early_exit_loss_curriculum = _get_component_from_path(
                    early_exit_loss_curriculum
                )(
                    do_output_hidden_states=do_output_hidden_states,
                    max_steps=self.total_epochs * self._steps_per_epoch,
                    train_last_layer=train_last_layer,
                    last_step=self.global_step,
                    verbose=verbose,
                )
                do_output_hidden_states = early_exit_loss_curriculum.get()
            else:
                if train_last_layer:
                    do_output_hidden_states[len(self._model.layers) - 1] = True

            self.think_start_token = cfg_early_exit_loss.get("think_start_token", "<think>")
            self.think_end_token = cfg_early_exit_loss.get("think_end_token", "</think>")
            print(f"Think Start and End tokens: {self.think_start_token}, {self.think_end_token}")

        return do_output_hidden_states, early_exit_loss_curriculum

    def save_checkpoint(
        self,
        epoch: int,
    ) -> None:
        checkpoint_dict = {}
        intermediate_checkpoint = epoch + 1 < self.total_epochs

        log.info(
            "Saving checkpoint. This may take some time..."
        )
        start = time.perf_counter()

        checkpoint_dict.update({training.MODEL_KEY: self._model.state_dict()})

        if intermediate_checkpoint:
            start = time.perf_counter()
            log.info("Getting optimizer state dict...")
            if not self._optimizer_in_bwd:
                opt_state_dict = self._optimizer.state_dict()
            else:
                opt_state_dict = self._optim_ckpt_wrapper.state_dict()

            checkpoint_dict.update(
                {
                    training.OPT_KEY: opt_state_dict,
                    training.SEED_KEY: self.seed,
                    training.EPOCHS_KEY: self.epochs_run,
                    training.TOTAL_EPOCHS_KEY: self.total_epochs,
                    training.MAX_STEPS_KEY: self.max_steps_per_epoch,
                }
            )

        self._checkpointer.save_checkpoint(
            checkpoint_dict,
            epoch=epoch,
            intermediate_checkpoint=intermediate_checkpoint,
        )
        log.info(f"Saving checkpoint took {time.perf_counter() - start:.2f} secs")

    def create_think_mask(self, input_ids, think_start_tokens, think_end_tokens):
        def find_token_sequence_positions(sequence, target):
                seq_len = sequence.size(0)
                target_len = target.size(0)
                
                if target_len > seq_len:
                    return []
                    
                positions = []
                i = 0
                while i <= seq_len - target_len:
                    if torch.all(sequence[i:i + target_len] == target):
                        positions.append(i)
                        i += target_len  
                    else:
                        i += 1
                return positions
                    
        think_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=input_ids.device)
        
        for b in range(input_ids.size(0)):
            # Find all start and end positions
            starts = find_token_sequence_positions(input_ids[b], think_start_tokens)
            ends = find_token_sequence_positions(input_ids[b], think_end_tokens)
            
            # Skip if we don't have matching pairs
            if len(starts) == 0 or len(ends) == 0:
                continue
                
            # Match start-end pairs
            # We only consider ends that come after starts
            current_start_idx = 0
            current_end_idx = 0
            
            while current_start_idx < len(starts) and current_end_idx < len(ends):
                start_pos = starts[current_start_idx]
                end_pos = ends[current_end_idx]
                
                # If end comes before start, move to next end
                if end_pos <= start_pos:
                    current_end_idx += 1
                    continue
                    
                # We found a valid pair
                # Mark the section between start and end (including start tokens, excluding end tokens)
                think_mask[b, start_pos:end_pos] = True
                
                # Move to next pair
                current_start_idx += 1
                current_end_idx += 1
        
        return think_mask

    def train(self) -> None:
        if self._compile:
            log.info(
                "NOTE: torch.compile is enabled and model is compiled in first forward. Expect a relatively slow first iteration."
            )

        if not self._optimizer_in_bwd:
            self._optimizer.zero_grad()
        else:
            for opt in self._optim_ckpt_wrapper.optim_map.values():
                opt.zero_grad()

        t0 = time.perf_counter()
        running_loss = 0
        num_tokens = 0

        if self._do_output_hidden_states is not None:
            self._model.output_hidden_states = [
                i
                for i in range(len(self._do_output_hidden_states))
                if self._do_output_hidden_states[i]
            ]

        self._profiler.start()
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            self._sampler.set_epoch(curr_epoch)

            pbar = tqdm(total=self._steps_per_epoch)
            for idx, batch in enumerate(self._dataloader):
                if (
                    self.max_steps_per_epoch is not None
                    and (idx // self._gradient_accumulation_steps)
                    == self.max_steps_per_epoch
                ):
                    break

                if (
                    curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                    and self._device.type == "cuda"
                ):
                    torch.cuda.memory._record_memory_history()

                utils.batch_to_device(batch, self._device)

                current_num_tokens = (
                    batch["labels"] != self._loss_fn.ignore_index
                ).sum()
                num_tokens += current_num_tokens

                labels = batch.pop("labels")

                with self.activations_handling_ctx:
                    outputs = self._model(**batch)
                    if self._model.output_hidden_states:
                        logits = outputs.pop(-1)
                        hidden_states = {
                            i: h
                            for i, h in zip(self._model.output_hidden_states, outputs)
                        }
                    else:
                        logits = outputs

                labels = torch.hstack(
                    (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
                )
                if not isinstance(logits, list):
                    labels = labels.reshape(-1)
                    logits = logits.reshape(-1, logits.size(-1))

                if self._model.output_hidden_states:
                    # think_start_token = self._tokenizer.encode(self.think_start_token)[0]
                    # think_end_token = self._tokenizer.encode(self.think_end_token)[0]    

                    think_start_tokens = self._tokenizer.encode(self.think_start_token)[:-1] # because for some reason this tokenizer is adding <|im_end|> at the end 
                    think_end_tokens = self._tokenizer.encode(self.think_end_token)[:-1]

                    think_start_tokens = torch.tensor(think_start_tokens, device=self._device)
                    think_end_tokens = torch.tensor(think_end_tokens, device=self._device)
                    
                    input_ids = batch["tokens"]                                     

                    # think_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=labels.device)                    
                    
                    # # Find think sections and create masks
                    # for b in range(input_ids.size(0)):
                    #     starts = (input_ids[b] == think_start_token).nonzero().flatten()
                    #     ends = (input_ids[b] == think_end_token).nonzero().flatten()
                    #     for start, end in zip(starts, ends):
                    #         think_mask[b, start:end] = True

                    think_mask = self.create_think_mask(
                        input_ids,
                        think_start_tokens,
                        think_end_tokens
                    )                    
                    
                    think_mask = think_mask.reshape(-1)
                    
                    # Separate labels for think and answer tokens 
                    think_labels = labels.clone()
                    answer_labels = labels.clone()
                    
                    think_labels[~think_mask] = self._loss_fn.ignore_index
                    answer_labels[think_mask] = self._loss_fn.ignore_index
                    
                    # Calculate losses separately
                    think_loss = early_exit_loss(
                        self._model,
                        hidden_states,
                        think_labels,
                        self._loss_fn,
                        self._early_exit_loss_scale,
                        self._early_exit_loss_scale_type,
                        is_think=True
                    ) if think_mask.any() else 0.0
                    
                    answer_loss = early_exit_loss(
                        self._model,
                        hidden_states,
                        answer_labels,
                        self._loss_fn,
                        self._early_exit_loss_scale,
                        self._early_exit_loss_scale_type,
                        is_think=False
                    ) if (~think_mask).any() else 0.0
                    
                    current_loss = (think_loss + answer_loss) * current_num_tokens
                else:
                    current_loss = self._loss_fn(logits, labels) * current_num_tokens

                del logits
                running_loss += current_loss
                current_loss.backward()

                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    if not self._optimizer_in_bwd:
                        training.scale_grads(self._model, 1 / num_tokens)
                        if self._clip_grad_norm is not None:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self._model.parameters(),
                                max_norm=float(self._clip_grad_norm),
                            )
                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)

                    self.global_step += 1

                    loss_to_log = running_loss.item() / num_tokens
                    pbar.update(1)
                    pbar.set_description(
                        f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log}"
                    )

                    if self.global_step % self._log_every_n_steps == 0:
                        time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "loss": loss_to_log,
                            "lr": get_lr(
                                (
                                    self._optimizer
                                    if not self._optimizer_in_bwd
                                    else self._optim_ckpt_wrapper
                                ),
                            ),
                            "tokens_per_second": num_tokens / time_per_step,
                        }
                        if self._log_peak_memory_stats:
                            log_dict.update(
                                training.get_memory_stats(device=self._device)
                            )
                        if self._clip_grad_norm is not None:
                            log_dict.update({"grad_norm": grad_norm})
                        self._metric_logger.log_dict(
                            log_dict,
                            step=self.global_step,
                        )

                    running_loss = 0
                    num_tokens = 0
                    t0 = time.perf_counter()

                    if self._early_exit_loss_curriculum:
                        self._early_exit_loss_curriculum.step()
                        self._do_output_hidden_states = (
                            self._early_exit_loss_curriculum.get()
                        )
                        self._model.output_hidden_states = [
                            i
                            for i in range(len(self._do_output_hidden_states))
                            if self._do_output_hidden_states[i]
                        ]

                    if (
                        curr_epoch == 0
                        and self.profiler_profile_memory
                        and idx
                        == self.profiler_wait_steps
                        + self.profiler_warmup_steps
                        + self.profiler_active_steps
                        and self._device.type == "cuda"
                    ):
                        torch.cuda.memory._record_memory_history(enabled=None)

                    self._profiler.step()

            self.epochs_run += 1            
            # self.save_checkpoint(epoch=curr_epoch) # lets save only at the end
        
        self.save_checkpoint(epoch=curr_epoch)

        self._profiler.stop()

    def cleanup(self) -> None:
        self._metric_logger.close()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="EarlyExitFinetuneRecipeSingleDevice", cfg=cfg)
    recipe = EarlyExitFinetuneRecipeSingleDevice(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
