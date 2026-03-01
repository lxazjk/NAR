import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
import torch.multiprocessing as mp

from nanovllm.config import Config, create_proximity_mask
from nanovllm.sampling_params import SamplingParams, NARSamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        self.model_type = config.model_type
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        
        if self.model_type != "nar":
            self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
            config.eos = self.tokenizer.eos_token_id
        else:
            self.tokenizer = None
            config.eos = -1
        
        self.scheduler = Scheduler(config)
        
        if self.model_type == "nar" and config.use_proximity_mask:
            self.proximity_mask = create_proximity_mask(
                max_seq_length=config.max_model_len,
                cls_token_num=config.cls_token_num,
                block_size=config.block_size,
                batch_size=config.max_num_seqs,
            )
            self.model_runner.proximity_mask = self.proximity_mask
        
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str) and self.tokenizer:
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        
        if self.tokenizer:
            outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        else:
            outputs = [{"token_ids": token_ids} for token_ids in outputs]
        
        if use_tqdm:
            pbar.close()
        return outputs


class NAREngine(LLMEngine):
    
    def __init__(self, model, **kwargs):
        kwargs.setdefault('model_type', 'nar')
        kwargs.setdefault('use_proximity_mask', True)
        super().__init__(model, **kwargs)
        
        self.cfg_scale = self.config.cfg_scale
        self.cfg_interval = self.config.cfg_interval
        self.block_size = self.config.block_size
        self.cls_token_num = self.config.cls_token_num
        self.grid_size = int(self.block_size ** 0.5)
    
    def generate_image(
        self,
        condition: torch.Tensor,
        sampling_params: NARSamplingParams | None = None,
        use_tqdm: bool = True,
    ) -> torch.Tensor:
        if sampling_params is None:
            sampling_params = NARSamplingParams(
                cfg_scale=self.cfg_scale,
                cfg_interval=self.cfg_interval,
                block_size=self.block_size,
            )
        
        batch_size = condition.shape[0]
        max_new_tokens = self.block_size
        
        if sampling_params.cfg_scale > 1.0:
            if self.config.model_type == 'c2i':
                cond_null = torch.ones_like(condition) * self.config.num_classes
                cond_combined = torch.cat([condition, cond_null])
            else:
                cond_null = torch.zeros_like(condition)
                cond_combined = torch.cat([condition, cond_null])
        else:
            cond_combined = condition
        
        generated_tokens = self._generate_nar_tokens(
            cond_combined, 
            max_new_tokens,
            sampling_params,
            use_tqdm,
        )
        
        return generated_tokens
    
    def _generate_nar_tokens(
        self,
        condition: torch.Tensor,
        max_new_tokens: int,
        sampling_params: NARSamplingParams,
        use_tqdm: bool,
    ) -> torch.Tensor:
        batch_size = condition.shape[0] if sampling_params.cfg_scale <= 1.0 else condition.shape[0] // 2
        
        new_tokens = [[] for _ in range(self.grid_size)]
        cur_token = None
        input_pos = self.cls_token_num
        
        iterations = 2 * self.grid_size - 1
        generated_token_num = 0
        
        if use_tqdm:
            pbar = tqdm(total=iterations, desc="Generating tokens", dynamic_ncols=True)
        
        cfg_flag = True
        for itera in range(iterations):
            accept_token_num = itera + 1 if itera < self.grid_size else iterations - itera
            
            if sampling_params.cfg_interval > -1 and generated_token_num > sampling_params.cfg_interval:
                cfg_flag = False
            
            next_token = self._decode_proximity_tokens(
                cur_token, input_pos, sampling_params, cfg_flag, accept_token_num, itera < self.grid_size
            )
            cur_token = next_token
            
            input_pos_list = []
            for i in range(self.grid_size):
                j = itera - i
                if j < 0 or j >= self.grid_size:
                    continue
                input_pos_list.append(self.cls_token_num + i * self.grid_size + j)
            input_pos = input_pos_list
            
            i = 0
            for token_arr in new_tokens:
                if i >= cur_token.shape[1]:
                    break
                if len(token_arr) < self.grid_size:
                    token_arr.append(cur_token[:, i].view(-1, 1))
                    i += 1
            
            generated_token_num += accept_token_num
            
            if use_tqdm:
                pbar.update(1)
        
        if use_tqdm:
            pbar.close()
        
        new_tokens_tensor = torch.cat([torch.cat(token_arr, dim=-1) for token_arr in new_tokens], dim=-1)
        return new_tokens_tensor[:, 1:]
    
    def _decode_proximity_tokens(
        self,
        cur_token: torch.Tensor | None,
        input_pos: int | list[int],
        sampling_params: NARSamplingParams,
        cfg_flag: bool,
        accept_token_num: int,
        accept_first_last: bool,
    ) -> torch.Tensor:
        if cur_token is None:
            return self._prefill_proximity_tokens(input_pos, sampling_params, cfg_flag, accept_token_num)
        
        return self._decode_step(cur_token, input_pos, sampling_params, cfg_flag, accept_token_num, accept_first_last)
    
    def _prefill_proximity_tokens(
        self,
        input_pos: int,
        sampling_params: NARSamplingParams,
        cfg_flag: bool,
        accept_token_num: int,
    ) -> torch.Tensor:
        from nanovllm.utils.context import set_context, get_context, reset_context
        
        seqs = self.scheduler.running if self.scheduler.running else []
        if not seqs:
            dummy_seq = Sequence([0] * self.cls_token_num, sampling_params)
            seqs = [dummy_seq]
        
        input_ids, positions = self.model_runner.prepare_prefill(seqs)
        
        context = get_context()
        proximity_mask = self.proximity_mask if hasattr(self, 'proximity_mask') else None
        
        logits = self.model_runner.run_model(input_ids, positions, True)
        
        logits = self._apply_cfg(logits, sampling_params.cfg_scale, cfg_flag)
        
        token_ids = self._sample_tokens(logits, sampling_params)
        
        reset_context()
        
        result = torch.zeros((1 if sampling_params.cfg_scale <= 1.0 else 2, 1), dtype=torch.long, device='cuda')
        result[:, 0] = torch.tensor(token_ids[:result.shape[0]])
        return result
    
    def _decode_step(
        self,
        cur_token: torch.Tensor,
        input_pos: list[int],
        sampling_params: NARSamplingParams,
        cfg_flag: bool,
        accept_token_num: int,
        accept_first_last: bool,
    ) -> torch.Tensor:
        from nanovllm.utils.context import set_context, get_context, reset_context
        
        batch_size = cur_token.shape[0]
        
        input_ids_list = []
        positions_list = []
        for pos in input_pos:
            input_ids_list.append(cur_token)
            positions_list.append(pos)
        
        input_ids = torch.cat(input_ids_list, dim=0)
        positions = torch.tensor(positions_list, dtype=torch.long, device='cuda')
        
        slot_mapping = torch.zeros(len(input_pos) * batch_size, dtype=torch.int32, device='cuda')
        context_lens = torch.ones(len(input_pos) * batch_size, dtype=torch.int32, device='cuda') * (self.cls_token_num + 1)
        block_tables = torch.zeros(len(input_pos) * batch_size, 1, dtype=torch.int32, device='cuda')
        
        proximity_mask = None
        if hasattr(self, 'proximity_mask'):
            proximity_mask = self.proximity_mask[:batch_size]
        
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            proximity_mask=proximity_mask,
        )
        
        logits = self.model_runner.run_model(input_ids.flatten(), positions, False)
        
        logits = self._apply_cfg(logits, sampling_params.cfg_scale, cfg_flag)
        
        token_ids = self._sample_tokens(logits, sampling_params)
        
        reset_context()
        
        result = torch.zeros((batch_size, accept_token_num), dtype=torch.long, device='cuda')
        for i in range(min(accept_token_num, len(token_ids))):
            result[:, i] = token_ids[i]
        
        return result
    
    def _apply_cfg(
        self,
        logits: torch.Tensor,
        cfg_scale: float,
        cfg_flag: bool,
    ) -> torch.Tensor:
        if cfg_scale <= 1.0:
            return logits
        
        if logits.shape[0] % 2 != 0:
            return logits
        
        cond_logits, uncond_logits = torch.split(logits, logits.shape[0] // 2, dim=0)
        if cfg_flag:
            return uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            return cond_logits
    
    def _sample_tokens(
        self,
        logits: torch.Tensor,
        sampling_params: NARSamplingParams,
    ) -> list[int]:
        logits = logits.float() / max(sampling_params.temperature, 1e-5)
        
        if sampling_params.top_k > 0:
            top_k = min(sampling_params.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        if sampling_params.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > sampling_params.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        probs = torch.softmax(logits, dim=-1)
        token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        return token_ids.tolist() if token_ids.dim() > 0 else [token_ids.item()]
