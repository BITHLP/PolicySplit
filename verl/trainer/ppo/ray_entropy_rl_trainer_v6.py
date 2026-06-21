import json
import os
import uuid
import re
from collections import defaultdict
from pprint import pprint
from typing import Dict, Optional, Type
from copy import deepcopy

import numpy as np
import ray
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from tensordict import TensorDict

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _timer,
    WorkerType,
    Role,
    AdvantageEstimator,
    ResourcePoolManager,
    compute_advantage,
    compute_response_mask,
    apply_kl_penalty,
)
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss, get_kl_controller
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.metric import (
    reduce_metrics,
)
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.device import get_torch_device


HE_SYS_PROMPT = "You are now in the \"High Entropy\" mode. Your primary directive is to explore a unique thought process and perspective before arriving at a final answer. Do not rely solely on your typical, most direct reasoning path. Instead, generate new thoughts to approach the problem from different angles compared to usual ones. Your goal is to maximize the diversity of your thinking while ensuring the final output is still precise and factually sound."


def clear_user_and_assistant(text):
    # begin with 'user\n' and end with '\nassistant\n'
    text = text.strip(' \n')
    if text.startswith('user\n'):
        text = text[5:]
    if text.endswith('\nassistant'):
        text = text[:-10]
    return text

def update_batch_extra_info(batch: DataProto, key: str, values: list):
    extra_info_array = batch.non_tensor_batch["extra_info"]
    new_extra_info_array = []
    for item, v in zip(extra_info_array, values):
        new_item = item.copy()
        new_item[key] = v
        new_extra_info_array.append(new_item)
    assert len(new_extra_info_array) == len(values)
    batch.non_tensor_batch["extra_info"] = np.array(new_extra_info_array)

def apply_correctness_mask(
    token_level_rewards: torch.Tensor,
    index: np.ndarray,
    scores: list,
    correctness_mask_method: str = "none",
):
    if correctness_mask_method == "none":
        return token_level_rewards

    elif correctness_mask_method == "all_same":
        with torch.no_grad():
            normalized_rewards = token_level_rewards.clone()
            id2all_scores = defaultdict(list)
            for idx, score in zip(index, scores):
                id2all_scores[idx].append(score)
            for idx in id2all_scores:
                if len(set(id2all_scores[idx])) > 1:  # not all answers achieve the same score
                    normalized_rewards[index == idx] = 0.0
        return normalized_rewards

    else:
        raise ValueError(f"correctness_mask_method {correctness_mask_method} not supported")

def compute_grpo_process_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    normalize_method: str = "zscore",
    lowest_reward_mask_ratio: float = 0.0,
    epsilon: float = 1e-6,
):
    masked_rewards = token_level_rewards * response_mask
    id2all_rewards = defaultdict(list)
    id2mean = {}
    id2std = {}
    id2min = {}
    id2max = {}
    id2threshold = {}

    with torch.no_grad():
        bsz, seq_len = masked_rewards.shape
        for i in range(bsz):
            valid_rewards = masked_rewards[i].masked_select(response_mask[i].bool())
            if valid_rewards.numel() > 0:
                id2all_rewards[index[i]].append(valid_rewards)

        for idx in id2all_rewards:
            group_rewards = torch.cat(id2all_rewards[idx])

            if group_rewards.numel() <= 1:
                id2mean[idx] = torch.tensor(0.0, device=token_level_rewards.device)
                id2std[idx] = torch.tensor(1.0, device=token_level_rewards.device)
                id2min[idx] = torch.min(group_rewards)
                id2max[idx] = torch.max(group_rewards)
                id2threshold[idx] = torch.quantile(group_rewards, lowest_reward_mask_ratio)
            else:
                id2mean[idx] = torch.mean(group_rewards)
                id2std[idx] = torch.std(group_rewards)
                id2min[idx] = torch.min(group_rewards)
                id2max[idx] = torch.max(group_rewards)
                id2threshold[idx] = torch.quantile(group_rewards, lowest_reward_mask_ratio)

        normalized_rewards = masked_rewards.clone()
        for i in range(bsz):
            idx = index[i]
            mean = id2mean[idx]
            std = id2std[idx]
            min_val = id2min[idx]
            max_val = id2max[idx]
            threshold = id2threshold[idx]
            above_threshold_mask = (masked_rewards[i] >= threshold)
            valid_mask = response_mask[i].bool()
            final_mask = valid_mask & above_threshold_mask
            if normalize_method == "zscore":
                normalized_rewards[i][valid_mask] = normalized_rewards[i][valid_mask] - mean
                normalized_rewards[i][valid_mask] = normalized_rewards[i][valid_mask] / (std + epsilon)
            elif normalize_method == "mean":
                normalized_rewards[i][valid_mask] = normalized_rewards[i][valid_mask] - mean
                normalized_rewards[i][valid_mask] = normalized_rewards[i][valid_mask] / (max_val - min_val + epsilon)
            elif normalize_method == "none":
                pass
            else:
                raise ValueError(f"normalize_method {normalize_method} not supported")
            normalized_rewards[i][~final_mask] = 0.0

        res = normalized_rewards

    return res

def apply_mixed_kl_penalty(
    data: DataProto,
    entropys: torch.Tensor,
    alpha: core_algos.AdaptiveKLController,
    eta1: core_algos.AdaptiveKLController,
    eta2: core_algos.AdaptiveKLController,
    kl_penalty="kl",
    multi_turn=False,
    penalty_normalize_method: str = "zscore",
    lowest_entropy_mask_ratio: float = 0.0,
    correctness_mask_method: str = "none",
):
    data.batch["entropys"] = entropys
    responses = data.batch["responses"]
    index = data.non_tensor_batch["uid"]
    token_level_scores = data.batch["token_level_scores"]
    scores = token_level_scores.sum(-1).cpu().tolist()
    response_length = responses.size(1)
    batch_size = data.batch.batch_size[0]
    sample_size = batch_size // 4
    alpha_coef = alpha.value
    eta1_coef = eta1.value
    eta2_coef = eta2.value
    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # quarter 1: org prompts, org responses
    # quarter 2: he prompts + he responses
    # quarter 3: org prompts + he responses
    # quarter 4: he prompts + org responses
    batch_quarter_1 = data.batch[:sample_size]
    batch_quarter_2 = data.batch[sample_size:2*sample_size]
    batch_quarter_3 = data.batch[2*sample_size:3*sample_size]
    batch_quarter_4 = data.batch[3*sample_size:]

    # original KL loss
    ref_kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    ref_kld = ref_kld * response_mask
    current_ref_kl = verl_F.masked_mean(ref_kld, mask=response_mask, axis=-1)  # average over sequence
    current_ref_kl = torch.mean(current_ref_kl, dim=0).item()

    # KL loss between two policies
    org_he_kl = core_algos.kl_penalty(batch_quarter_1["old_log_probs"], batch_quarter_4["old_log_probs"], kl_penalty=kl_penalty)  # (sample_size, response_length)
    current_org_he_kl = verl_F.masked_mean(org_he_kl, mask=response_mask[:sample_size, :], axis=-1)
    current_org_he_kl = torch.mean(current_org_he_kl, dim=0).item()
    he_org_kl = core_algos.kl_penalty(batch_quarter_2["old_log_probs"], batch_quarter_3["old_log_probs"], kl_penalty=kl_penalty)  # (sample_size, response_length)
    current_he_org_kl = verl_F.masked_mean(he_org_kl, mask=response_mask[sample_size:2*sample_size, :], axis=-1)
    current_he_org_kl = torch.mean(current_he_org_kl, dim=0).item()

    # logprob
    current_he_logprob = verl_F.masked_mean(batch_quarter_2["old_log_probs"], mask=response_mask[sample_size:2*sample_size, :], axis=-1)
    current_he_logprob = torch.mean(current_he_logprob, dim=0).item()
    current_org_logprob = verl_F.masked_mean(batch_quarter_1["old_log_probs"], mask=response_mask[3*sample_size:, :], axis=-1)
    current_org_logprob = torch.mean(current_org_logprob, dim=0).item()

    # entropy loss
    entropy_loss_he_quarter_2 = alpha_coef * batch_quarter_2["entropys"] + (1 - alpha_coef) * batch_quarter_3["entropys"]  # (sample_size, response_length)
    entropy_loss_he_quarter_4 = alpha_coef * batch_quarter_4["entropys"] + (1 - alpha_coef) * batch_quarter_1["entropys"]  # (sample_size, response_length)
    entropy_loss_he_quarter_1 = torch.zeros_like(batch_quarter_1["entropys"])  # (sample_size, response_length)
    entropy_loss_he_quarter_3 = torch.zeros_like(batch_quarter_3["entropys"])  # (sample_size, response_length)
    entropy_loss_he = torch.cat([entropy_loss_he_quarter_1, entropy_loss_he_quarter_2, entropy_loss_he_quarter_3, entropy_loss_he_quarter_4], dim=0)  # (batch_size, response_length)
    entropy_loss_he = entropy_loss_he * response_mask
    entropy_loss_org_quarter_1 = alpha_coef * batch_quarter_1["entropys"] + (1 - alpha_coef) * batch_quarter_4["entropys"]  # (sample_size, response_length)
    entropy_loss_org_quarter_3 = alpha_coef * batch_quarter_3["entropys"] + (1 - alpha_coef) * batch_quarter_2["entropys"]  # (sample_size, response_length)
    entropy_loss_org_quarter_2 = torch.zeros_like(batch_quarter_2["entropys"])  # (sample_size, response_length)
    entropy_loss_org_quarter_4 = torch.zeros_like(batch_quarter_4["entropys"])  # (sample_size, response_length)
    entropy_loss_org = torch.cat([entropy_loss_org_quarter_1, entropy_loss_org_quarter_2, entropy_loss_org_quarter_3, entropy_loss_org_quarter_4], dim=0)  # (batch_size, response_length)
    entropy_loss_org = entropy_loss_org * response_mask

    assert lowest_entropy_mask_ratio >= 0.0 and lowest_entropy_mask_ratio <= 1.0, f"lowest_entropy_mask_ratio {lowest_entropy_mask_ratio} not in [0.0, 1.0]"
    internal_rewards_he = compute_grpo_process_advantage(
        entropy_loss_he,
        response_mask,
        index,
        normalize_method=penalty_normalize_method,
        lowest_reward_mask_ratio=lowest_entropy_mask_ratio,
    )
    internal_rewards_he = apply_correctness_mask(
        internal_rewards_he,
        index,
        scores,
        correctness_mask_method,
    )
    internal_rewards_org = compute_grpo_process_advantage(
        entropy_loss_org,
        response_mask,
        index,
        normalize_method=penalty_normalize_method,
        lowest_reward_mask_ratio=lowest_entropy_mask_ratio,
    )
    internal_rewards_org = apply_correctness_mask(
        internal_rewards_org,
        index,
        scores,
        correctness_mask_method,
    )
    internal_rewards = eta1_coef * internal_rewards_he - eta2_coef * internal_rewards_org

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    data.batch["token_level_rewards"] = token_level_scores

    data.batch.pop("entropys")

    metrics = {
        "actor/ref_kl": current_ref_kl,
        "actor/org_he_kl": current_org_he_kl,
        "actor/he_org_kl": current_he_org_kl,
        "actor/he_logprob": current_he_logprob,
        "actor/org_logprob": current_org_logprob,
    }

    return data, metrics, internal_rewards

def compute_advantage_with_internal_rewards(batch: DataProto, adv_estimator, internal_rewards, kappa=0, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        internal_rewards (torch.Tensor): The internal rewards computed from the KL penalty.
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    batch = compute_advantage(
        batch,
        adv_estimator=adv_estimator,
        gamma=gamma,
        lam=lam,
        num_repeat=num_repeat,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        multi_turn=multi_turn,
        **kwargs,
    )
    if kappa > 0:
        assert kappa > 1e-5
        adv = batch.batch["advantages"].clone()
        boundary = torch.abs(adv) / kappa
        internal_rewards = torch.clamp(internal_rewards, -boundary, boundary)

    batch.batch["advantages"] += internal_rewards
    batch.batch["returns"] += internal_rewards
    return batch

def debug_print_data(batch: DataProto):
    print('='*30)
    print('-'*10 + 'BATCH' + '-'*10)
    for key in batch.batch.keys():
        print("{}: {}".format(key, batch.batch[key].shape))
        print(batch.batch[key])
    print('-'*10 + 'NON-TENSOR-BATCH' + '-'*10)
    for key in batch.non_tensor_batch.keys():
        print("{}: {}".format(key, batch.non_tensor_batch[key].shape))
        print(batch.non_tensor_batch[key])
    print('='*30)


class RayEntropyPPOTrainerV6(RayPPOTrainer):
    """
        batch: DataProto
            -   non_tensor_batch: ['data_source', 'ability', 'reward_model', 'extra_info', 'raw_prompt_ids', 'index', 'tools_kwargs']
            -   batch: ['input_ids', 'attention_mask', 'position_ids']
        {gen/rewrite/continue}_batch: DataProto
            -   non_tensor_batch: ['raw_prompt_ids', 'tools_kwargs']
            -   batch: ['input_ids', 'attention_mask', 'position_ids']
        {gen/rewrite/continue/pseudo_gen}_batch_output: DataProto
            -   non_tensor_batch: ['tools_kwargs']
            -   batch
                - prompts (batch_size * max_prompt_length): real prompts, prefilled by eos
                - attention_mask (batch_size * (max_prompt_length+max_response_length)): 0 for prefilled & postfilled tokens, 1 for all real tokens
                - position_ids (batch_size * (max_prompt_length+max_response_length)): 0 for prefilled tokens, i>=0 for real & postfilled tokens
                - responses (batch_size * max_response_length): response tokens, postfilled by eos
                - rollout_log_probs (batch_size * max_response_length): log probs of responses, postfilled by -1e0
                - input_ids (batch_size * (max_prompt_length+max_response_length)): real prompt + response tokens, prefilled & postfilled by eos
        """
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
            processor,
            reward_fn,
            val_reward_fn,
            train_dataset,
            val_dataset,
            collate_fn,
            train_sampler,
            device_name,
        )
        if config.algorithm.use_kl_in_reward:
            assert config.algorithm.kl_ctrl.type == 'fixed', 'Only fixed kl controller is supported.'
            alpha_ctrl = deepcopy(config.algorithm.kl_ctrl)
            alpha_ctrl.kl_coef = config.algorithm.alpha
            eta1_ctrl = deepcopy(config.algorithm.kl_ctrl)
            eta1_ctrl.kl_coef = config.algorithm.eta1
            eta2_ctrl = deepcopy(config.algorithm.kl_ctrl)
            eta2_ctrl.kl_coef = config.algorithm.eta2
            self.kl_ctrl_in_reward_alpha = get_kl_controller(alpha_ctrl)
            self.kl_ctrl_in_reward_eta1 = get_kl_controller(eta1_ctrl)
            self.kl_ctrl_in_reward_eta2 = get_kl_controller(eta2_ctrl)

    def build_batch_with_prompts(self, raw_prompts: list[str], max_prompt_length=None, padding_side='left') -> DataProto:
        if max_prompt_length is None:
            max_prompt_length = self.train_dataset.max_prompt_length
        # process data following RLHFDataset
        assert self.tokenizer.truncation_side == 'right'
        model_inputs = self.tokenizer(
            raw_prompts,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            padding_side=padding_side,
            max_length=max_prompt_length,
        )
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        position_ids = compute_position_id_with_mask(attention_mask)
        batch = DataProto(
            non_tensor_batch={
                # 'raw_prompt_ids': np.array(input_ids)
            },
            batch=TensorDict(
                {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'position_ids': position_ids,
                },
                batch_size=(len(raw_prompts),)
            )
        )
        return batch

    def get_mixed_batch(self, batch: DataProto) -> DataProto:
        """
            Repeat the batch twice, the first half is the original batch, and the second half is the batch with HE system prompt.
            TODO: non_tensor_batch are not considered.
        """
        org_prompts = self.tokenizer.batch_decode(
            batch.batch["input_ids"],
            skip_special_tokens=True,
        )
        org_prompts = list(map(clear_user_and_assistant, org_prompts))
        org_messages = [
            [{"role": "user", "content": org_prompt}]
            for org_prompt in org_prompts
        ]
        he_messages = [
            [{"role": "system", "content": HE_SYS_PROMPT}, {"role": "user", "content": org_prompt}]
            for org_prompt in org_prompts
        ]
        messages = org_messages + he_messages
        raw_prompts = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=True)
        batch = self.build_batch_with_prompts(raw_prompts)
        repeated_batch = batch.repeat(interleave=False)
        batch.non_tensor_batch = repeated_batch.non_tensor_batch
        batch.meta_info = repeated_batch.meta_info
        return batch

    def get_cartesian_product_batch(self, batch: DataProto) -> DataProto:
        """
            Given a batch of prompts and responses, where the first half of the batch is the original batch, and the second half is the batch with HE system prompt.
            Return a batch, where
                - the first quarter: org prompts + org responses
                - the second quarter: he prompts + he responses
                - the third quarter: org prompts + he responses
                - the fourth quarter: he prompts + org responses
            TODO: non_tensor_batch are not considered.
        """
        prompts = self.tokenizer.batch_decode(
            batch.batch["prompts"],
            skip_special_tokens=True,
        )
        prompts = list(map(clear_user_and_assistant, prompts))
        responses = self.tokenizer.batch_decode(
            batch.batch["responses"],
            skip_special_tokens=True,
        )
        org_prompts = prompts[:len(prompts)//2]
        he_prompts = prompts[len(prompts)//2:]
        org_responses = responses[:len(responses)//2]
        he_responses = responses[len(responses)//2:]
        def get_new_gen_batch_with_prompts_and_responses(prompts: list[str], responses: list[str]):
            input_messages = [[{'role': 'user', 'content': prompt}] for prompt in prompts]
            input_raw_prompts = self.tokenizer.apply_chat_template(input_messages, add_generation_prompt=True, tokenize=False, enable_thinking=True)
            output_raw_prompts = responses
            input_batch = self.build_batch_with_prompts(
                input_raw_prompts,
                max_prompt_length=self.config.data.max_prompt_length,
                padding_side='left',
            )
            output_batch = self.build_batch_with_prompts(
                output_raw_prompts,
                max_prompt_length=self.config.data.max_response_length,
                padding_side='right',
            )
            batch_prompts = input_batch.batch['input_ids']
            batch_responses = output_batch.batch['input_ids']
            batch_attention_ids = torch.cat([input_batch.batch['attention_mask'], output_batch.batch['attention_mask']], dim=1)
            batch_position_ids = compute_position_id_with_mask(batch_attention_ids)
            batch_rollout_log_probs = torch.full_like(batch_responses, -1e0)
            batch_input_ids = torch.cat([batch_prompts, batch_responses], dim=1)
            pseudo_batch = DataProto(
                batch=TensorDict(
                    {
                        'prompts': batch_prompts,
                        'attention_mask': batch_attention_ids,
                        'position_ids': batch_position_ids,
                        'responses': batch_responses,
                        'rollout_log_probs': batch_rollout_log_probs,
                        'input_ids': batch_input_ids,
                    },
                    batch_size=len(prompts),
                )
            )
            return pseudo_batch
        org_prompt_he_response_batch = get_new_gen_batch_with_prompts_and_responses(org_prompts, he_responses)
        he_prompt_org_response_batch = get_new_gen_batch_with_prompts_and_responses(he_prompts, org_responses)
        return DataProto.concat([batch, org_prompt_he_response_batch, he_prompt_org_response_batch])

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path, **kwargs):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }
        for k, v in kwargs.items():
            base_data[k] = v

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # convert batch to mixed batch by adding HE system prompts
                sample_size = batch.batch.batch_size[0]

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                # non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                non_tensor_batch_keys_to_pop = []
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                # if "raw_prompt" in batch.non_tensor_batch:
                #     non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                # TODO: In batch and gen_batch, 'raw_prompt_ids' and
                #       'raw_prompt' of non_tensor_batch not changed,
                #       we need to change them to mixed versions.
                #       Currently, we just ignore them, as the rollout
                #       worker will use batch['input_ids'] to generate
                #       mixed raw_prompt_ids.
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                gen_batch = self.get_mixed_batch(gen_batch)

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):

                    # generate a batch
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    with _timer("merge_batch", timing_raw):
                        assert batch.batch.batch_size[0] == sample_size
                        assert gen_batch_output.batch.batch_size[0] == sample_size * 2 * self.config.actor_rollout_ref.rollout.n
                        # self cartesian product of gen_batch_output
                        gen_batch_output = self.get_cartesian_product_batch(gen_batch_output)
                        assert gen_batch_output.batch.batch_size[0] == sample_size * 4 * self.config.actor_rollout_ref.rollout.n
                        # add extra information at batch.non_tensor_batch
                        batch = batch.repeat(repeat_times=4, interleave=False)
                        uuids = [str(uuid.uuid4()) for _ in range(2*sample_size)] * 2
                        batch.non_tensor_batch["uid"] = np.array(uuids, dtype=object)
                        in_batch_ids = list(range(sample_size)) * 4
                        is_he_prompt = [False] * sample_size + [True] * sample_size + [False] * sample_size + [True] * sample_size
                        is_he_response = [False] * sample_size + [True] * sample_size + [True] * sample_size + [False] * sample_size
                        update_batch_extra_info(batch, 'is_he_prompt', is_he_prompt)
                        update_batch_extra_info(batch, 'is_he_response', is_he_response)
                        update_batch_extra_info(batch, 'in_batch_ids', in_batch_ids)
                        update_batch_extra_info(batch, 'uuids', uuids)
                        # merge batch and gen_batch_output
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    assert not self.config.trainer.balance_batch, "balance_batch is not supported for entropy_v6"
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        entropy_estimation = self.config.algorithm.get("entropy_estimation", "logprob")
                        if entropy_estimation == "logprob":
                            entropys = batch.batch["old_log_probs"].clone().detach()
                            entropys = -entropys
                        elif entropy_estimation == "entropy":
                            entropys = entropys.clone().detach()
                        else:
                            raise ValueError(f"Unknown entropy estimation method: {entropy_estimation}")

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics, internal_rewards = apply_mixed_kl_penalty(
                                batch,
                                entropys,
                                alpha=self.kl_ctrl_in_reward_alpha,
                                eta1=self.kl_ctrl_in_reward_eta1,
                                eta2=self.kl_ctrl_in_reward_eta2,
                                kl_penalty=self.config.algorithm.kl_penalty,
                                penalty_normalize_method=self.config.algorithm.get("penalty_normalize_method", "zscore"),
                                lowest_entropy_mask_ratio = self.config.algorithm.get("lowest_entropy_mask_ratio", 0.0),
                                correctness_mask_method=self.config.algorithm.get("correctness_mask_method", "none"),
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        kappa = self.config.algorithm.get("kappa", 0)
                        batch = compute_advantage_with_internal_rewards(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            internal_rewards=internal_rewards,
                            kappa=kappa,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            is_he_prompt = [x["is_he_prompt"] for x in batch.non_tensor_batch["extra_info"].tolist()]
                            is_he_response = [x["is_he_response"] for x in batch.non_tensor_batch["extra_info"].tolist()]
                            in_batch_ids = [x["in_batch_ids"] for x in batch.non_tensor_batch["extra_info"].tolist()]
                            uuids = [x["uuids"] for x in batch.non_tensor_batch["extra_info"].tolist()]
                            ground_truth = [x["ground_truth"] for x in batch.non_tensor_batch["reward_model"].tolist()]
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                                is_he_prompt=is_he_prompt,
                                is_he_response=is_he_response,
                                in_batch_ids=in_batch_ids,
                                uuids=uuids,
                                ground_truth=ground_truth,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                org_mask = np.array([not item["is_he_prompt"] and not item["is_he_response"] for item in batch.non_tensor_batch["extra_info"]])
                he_mask = np.array([item["is_he_prompt"] or item["is_he_response"] for item in batch.non_tensor_batch["extra_info"]])
                org_batch = batch[org_mask]
                he_batch = batch[he_mask]
                org_data_metrics = compute_data_metrics(batch=org_batch, use_critic=self.use_critic)
                he_data_metrics = compute_data_metrics(batch=he_batch, use_critic=self.use_critic)
                k_list = list(org_data_metrics.keys())
                for key in k_list:
                    if any([x in key for x in ['advantages', 'returns', 'prompt_length']]):
                        continue
                    org_data_metrics[f'sub_batch_org/{key}'] = org_data_metrics.pop(key)
                    he_data_metrics[f'sub_batch_he/{key}'] = he_data_metrics.pop(key)
                metrics.update(org_data_metrics)
                metrics.update(he_data_metrics)
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return