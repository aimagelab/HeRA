from typing import List, Optional
import torch
import json
from pathlib import Path
from src.models.conversations import CONV_TEMPLATES, CONV_MAPPING, DEFAULT_IMAGE_TOKEN
import src.custom_utils as utils
from transformers import (
    HfArgumentParser,
    set_seed,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoImageProcessor
)
from src.train.args import CustomTrainingArguments, DataArguments, ModelArguments, postprocess_args
from src.train.trainers import CustomTrainer
from argparse import Namespace
from src.models import (
    MLLMConfig, MLLMForCausalLM,
    MLLMAlignConfig, MLLMAlignModel
)
from src.models.model_utils import build_vision_encoder, load_torch_weights
from src.train.collators import SupervisedCollator
from PIL import Image
from transformers.integrations import is_deepspeed_zero3_enabled
import deepspeed
from src.data.lmdb_image_reader import LMDBImageReader

logger = utils.get_logger()


class LlavaDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer,
        image_processor,
        args: DataArguments,
        lmdb_root: Optional[str] = None
    ):
        super().__init__()
        self.data_path = data_path
        self.image_folder = Path(image_folder if image_folder else '')
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.args = args
        self.conv_template = CONV_TEMPLATES(args.conv_template)
        self.img_placeholder = Image.new('RGB', (400, 400))

        self.lmdb_reader = LMDBImageReader(lmdb_root) if lmdb_root else None
        if self.lmdb_reader:
            logger.info(f"LMDB image reader initialized with root: {lmdb_root}")

        n_samples = utils.DEBUG_N_DATASET_SAMPLES
        if n_samples:
            logger.info(
                f"Using only {n_samples} dataset samples for debugging.")
        with open(data_path, 'r') as f:
            if data_path.endswith('.json'):
                self.data = json.load(f)
                if n_samples:
                    self.data = self.data[:n_samples]
            elif data_path.endswith('.jsonl'):
                jsonl_iter = (next(f)
                              for _ in range(n_samples)) if n_samples else f
                self.data = [json.loads(line) for line in jsonl_iter]
            else:
                raise ValueError(
                    f"Unsupported data file format: {data_path}, only .json and .jsonl are supported.")
        logger.info(f'Loaded {len(self.data)} samples from {data_path}')

    def __len__(self):
        return len(self.data)

    @property
    def mod_lens(self):
        lens = []
        for sample in self.data:
            cur_len = sum(len(conv['value'].split())
                          for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            lens.append(cur_len)
        return lens

    def __getitem__(self, idx):
        sample = self.data[idx]

        img_err = False
        if sample.get('image', None):
            try:
                if 'lmdb_file' in sample and self.lmdb_reader:
                    image = self.lmdb_reader.read_image(
                        lmdb_name=sample['lmdb_file'],
                        image_key=sample['image']
                    )
                else:
                    image_path = self.image_folder.joinpath(sample['image'])
                    image = Image.open(image_path).convert('RGB')

                image_mask = True
                if self.args.image_aspect_ratio == 'pad':
                    image = utils.expand2square(image, tuple(int(x*255)
                                                             for x in self.image_processor.image_mean))
            except (FileNotFoundError, KeyError) as e:
                logger.warning(e)
                image = self.img_placeholder
                image_mask = False
                img_err = True
        else:
            image = self.img_placeholder
            image_mask = False

        input_ids = None
        attention_mask = None
        labels = None
        tok_kwargs = {}

        if self.args.prompt_max_length is not None:
            tok_kwargs['max_length'] = self.args.prompt_max_length
            tok_kwargs['truncation'] = True

        if self.conv_template == CONV_TEMPLATES.PLAIN:
            text = DEFAULT_IMAGE_TOKEN + '\n' + \
                sample['conversations'][1]['value']
            inputs_text = self.tokenizer(
                text, return_tensors="pt", add_special_tokens=False, **tok_kwargs)
            input_ids = inputs_text.input_ids[0]
            attention_mask = inputs_text.attention_mask[0]
            image_token_id = self.tokenizer.convert_tokens_to_ids(
                DEFAULT_IMAGE_TOKEN)
            labels = input_ids.clone()
            labels[labels == image_token_id] = -100
        else:
            conv = CONV_MAPPING[self.conv_template].new_empty()
            conv.add_sys_prompt()
            has_img_tok = False
            for turn in sample['conversations']:
                if turn['from'] == 'human':
                    role = 'user'
                    if DEFAULT_IMAGE_TOKEN in turn['value']:
                        turn['value'] = turn['value'].replace(
                            DEFAULT_IMAGE_TOKEN, '').strip()
                        turn['value'] = (DEFAULT_IMAGE_TOKEN +
                                         '\n' + turn['value']).strip()
                        has_img_tok = True
                elif turn['from'] == 'gpt':
                    role = 'assistant'
                conv.add_message(dict(role=role, content=turn['value']))
                if image_mask and not has_img_tok:
                    logger.warning(
                        f"No image token found in sample with image {sample['image']}. Skipping ...")
                    img_err = True

            text = conv.get_prompt()
            text_inputs = self.tokenizer(
                text, return_tensors="pt", **tok_kwargs)
            input_ids = text_inputs.input_ids[0]
            attention_mask = text_inputs.attention_mask[0]

            if not img_err:
                tokenized_text = self.tokenizer.convert_ids_to_tokens(
                    input_ids)
                labels = conv.get_labels(input_ids, tokenized_text)

        if img_err:
            labels = torch.full((len(input_ids),),
                                fill_value=-100, dtype=input_ids.dtype)
            image_mask = False

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            image=image,
            image_mask=image_mask
        )


def build_model_commons(args: ModelArguments) -> Namespace:
    config_cls = MLLMConfig
    model_cls = MLLMForCausalLM

    if args.ra_llm_layers or args.ra_llm_layers_heads:
        config_cls = MLLMAlignConfig
        model_cls = MLLMAlignModel
    is_mllm_ra = issubclass(model_cls, MLLMAlignModel)

    if args.checkpoint_path:
        model = model_cls.from_pretrained(args.checkpoint_path, lm_kwargs=dict(
            attn_implementation=args.attn_implementation))
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    else:
        text_config = AutoConfig.from_pretrained(args.language_model_name)
        if utils.DEBUG_LOAD_FIRST_LLM_LAYERS:
            text_config.num_hidden_layers = utils.DEBUG_LOAD_FIRST_LLM_LAYERS
            logger.info(
                f"Using a {utils.DEBUG_LOAD_FIRST_LLM_LAYERS}-layer language model for debugging")
        if args.disable_rope_scaling and getattr(text_config, 'rope_scaling', None) is not None:
            logger.info(
                f"Disabling RoPE scaling (was {text_config.rope_scaling}); reverting to vanilla RoPE with rope_theta={getattr(text_config, 'rope_theta', None)}")
            text_config.rope_scaling = None
        logger.info(f"Attention implementation: {args.attn_implementation}")
        language_model = AutoModelForCausalLM.from_pretrained(
            args.language_model_name,
            config=text_config,
            attn_implementation=args.attn_implementation
        )

        vision_model = build_vision_encoder(args.vision_model_name)

        model_kwargs = dict(
            language_model=language_model,
            vision_model=vision_model
        )
        config_kwargs = dict(
            text_config=text_config,
            vision_config=vision_model.config,
            vision_layer_idx=args.vision_layer_idx,
            skip_left_visual_tokens=args.skip_left_visual_tokens,
            projector_type=args.projector_type,
            projector_bias=args.projector_bias,
            bidir_attn_visual_tokens=args.bidir_attn_visual_tokens,
            image_size=args.image_size,
            image_aspect_ratio=args.image_aspect_ratio
        )

        if is_mllm_ra:
            tgt_vision_model = None
            if args.tgt_vision_model_name and args.tgt_vision_model_name != args.vision_model_name:
                tgt_vision_model = build_vision_encoder(
                    args.tgt_vision_model_name)
            model_kwargs['tgt_vision_model'] = tgt_vision_model

            config_kwargs['tgt_vision_model_name'] = args.tgt_vision_model_name
            config_kwargs['tgt_vision_layer_idx'] = args.tgt_vision_layer_idx
            config_kwargs['tgt_skip_left_visual_tokens'] = args.tgt_skip_left_visual_tokens
            config_kwargs['tgt_projector_type'] = args.tgt_projector_type
            config_kwargs['tgt_proj_intermediate_size'] = args.tgt_proj_intermediate_size
            config_kwargs['ra_llm_layers'] = args.ra_llm_layers
            config_kwargs['ra_llm_layers_heads'] = args.ra_llm_layers_heads
            config_kwargs['ra_llm_layers_heads_gqa'] = args.ra_llm_layers_heads_gqa
            config_kwargs['ra_loss_w'] = args.ra_loss_w
            config_kwargs['hera'] = args.hera
            config_kwargs['hera_topk'] = args.hera_topk
            config_kwargs['hera_learn_logit_scale'] = args.hera_learn_logit_scale
            config_kwargs['hera_logit_scale_init_val'] = args.hera_logit_scale_init_val
            config_kwargs['hera_untie_logit_scale'] = args.hera_untie_logit_scale

        config = config_cls(**config_kwargs)
        model = model_cls(config, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(
            model.config.text_config.name_or_path)
    if args.load_proj_weights:
        load_torch_weights(model.projector, args.load_proj_weights)
        logger.info(f"Loaded projector weights from {args.load_proj_weights}")
        if hasattr(model, 'tgt_projector'):
            tgt_proj_weights = Path(
                args.load_proj_weights).parent / 'tgt_projector.pt'
            if tgt_proj_weights.exists():
                load_torch_weights(model.tgt_projector, tgt_proj_weights)
                logger.info(
                    f"Loaded tgt projector weights from {tgt_proj_weights}")
        if hasattr(model, 'logit_scale'):
            logit_scale_path = Path(args.load_proj_weights).parent / 'logit_scale.pt'
            if logit_scale_path.exists():
                logit_scale = torch.load(logit_scale_path, map_location='cpu')
                with deepspeed.zero.GatheredParameters(
                    model.logit_scale,
                    modifier_rank=0,
                    enabled=is_deepspeed_zero3_enabled()
                ):
                    if model.logit_scale.numel() != logit_scale.numel():
                        logger.warning(f"Found 'logit_scale' of shape {logit_scale.shape} in {logit_scale_path},\n but current `logit_scale` shape is {model.logit_scale.shape}. Loading only first {len(model.logit_scale)} values")
                        logit_scale = logit_scale[:len(model.logit_scale)]
                    model.logit_scale.data.copy_(logit_scale)                
                logger.info(f"Loaded logit scale ({logit_scale}) from {logit_scale_path}")

    if is_mllm_ra:
        model.register_modules_for_alignment()

    set_seed(args.seed)

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            ret = tokenizer.add_special_tokens(
                dict(pad_token=tokenizer.unk_token))
        else:
            tokenizer.pad_token = tokenizer.eos_token
            ret = tokenizer.add_special_tokens(
                dict(pad_token=tokenizer.eos_token))
        assert ret == 0, "Failed to add pad token to tokenizer"

    image_processor = AutoImageProcessor.from_pretrained(
        model.config.vision_config.name_or_path)
    model.init_vision_tokenizer(tokenizer)
    model.init_image_processor(image_processor)

    tgt_image_processor = None
    if args.tgt_image_processor:
        tgt_image_processor = AutoImageProcessor.from_pretrained(
            args.tgt_vision_model_name, trust_remote_code=True)

    if args.override_bidir_attn_visual_tokens:
        model.config.bidir_attn_visual_tokens = args.bidir_attn_visual_tokens
        logger.info(
            f"`model.config.bidir_attn_visual_tokens` set to {model.config.bidir_attn_visual_tokens}")

    if args.image_aspect_ratio is not None:
        model.config.image_aspect_ratio = args.image_aspect_ratio
    logger.info(
        f"`model.config.image_aspect_ratio` set to {model.config.image_aspect_ratio}")

    model.requires_grad_(True)
    model.get_vision_model().requires_grad_(False)
    if hasattr(model, 'tgt_vision_model'):
        model.tgt_vision_model.requires_grad_(False)
    if args.train_proj_only:
        model.get_language_model().requires_grad_(False)
    if args.freeze_proj:
        model.projector.requires_grad_(False)
    if args.freeze_tgt_proj:
        model.tgt_projector.requires_grad_(False)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False))

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings(
        ).register_forward_hook(make_inputs_require_grad)

    if args.load_model_weights:
        to_load = MLLMForCausalLM.from_pretrained(args.load_model_weights)        
        with deepspeed.zero.GatheredParameters(
            to_load.parameters(), 
            modifier_rank=0,
            enabled=is_deepspeed_zero3_enabled()
        ):
            to_load = to_load.state_dict()
        with deepspeed.zero.GatheredParameters(
            model.parameters(), 
            modifier_rank=0,
            enabled=is_deepspeed_zero3_enabled()
        ):
            model.load_state_dict(to_load, strict=False)
        logger.info(f"Loaded model weights from {args.load_model_weights}")

    return Namespace(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        tgt_image_processor=tgt_image_processor,
    )


def build_data_commons(
    args: DataArguments,
    image_processor,
    tokenizer,
    tgt_image_processor=None,
) -> Namespace:
    train_dataset = LlavaDataset(
        args.train_data_path,
        args.train_image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        args=args,
        lmdb_root=args.lmdb_root
    )

    eval_dataset = None
    if args.eval_data_path:
        eval_dataset = LlavaDataset(
            args.eval_data_path,
            args.eval_image_folder,
            tokenizer=tokenizer,
            image_processor=image_processor,
            args=args,
            lmdb_root=args.lmdb_root
        )

    data_collator = SupervisedCollator(
        image_processor, tokenizer,
        tgt_image_processor=tgt_image_processor,
        tgt_image_size=args.tgt_image_size
    )

    return Namespace(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )


if __name__ == '__main__':
    parser = HfArgumentParser(
        (CustomTrainingArguments, ModelArguments, DataArguments))
    training_args, model_args, data_args = postprocess_args(
        *parser.parse_args_into_dataclasses())
    set_seed(training_args.seed)

    local_rank = utils.get_local_rank()
    torch.cuda.set_device(local_rank)
    print(f"Rank [{utils.get_rank()}]: set device ID {local_rank}")

    model_commons = build_model_commons(model_args)

    data_commons = build_data_commons(
        data_args,
        image_processor=model_commons.image_processor,
        tokenizer=model_commons.tokenizer,
        tgt_image_processor=model_commons.tgt_image_processor,
    )

    trainer = CustomTrainer(
        model=model_commons.model,
        args=training_args,
        train_dataset=data_commons.train_dataset,
        eval_dataset=data_commons.eval_dataset,
        data_collator=data_commons.data_collator,
        processing_class=model_commons.tokenizer
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    logger.info('Training completed')
