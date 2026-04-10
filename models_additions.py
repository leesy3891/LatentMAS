    # ----------------------------------------------------------------
    # Analysis helpers: layer-wise hidden-state collection
    # Add these methods to the ModelWrapper class in models.py
    # ----------------------------------------------------------------

    @torch.no_grad()
    def forward_collect_layerwise(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[Tuple, List[torch.Tensor], torch.Tensor]:
        """Forward pass returning per-layer last-position hidden states and updated KV cache.

        Returns:
            past_key_values: updated KV cache
            layer_hiddens_cpu: list of [B, D] CPU float16 tensors (one per layer incl. embedding)
            last_hidden_gpu: [B, D] tensor on GPU (last transformer layer, for continuing computation)
        """
        device = self.device
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0 and attention_mask is not None:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        kwargs = dict(
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        if input_ids is not None:
            kwargs["input_ids"] = input_ids.to(device)
        else:
            kwargs["inputs_embeds"] = inputs_embeds.to(device)

        outputs = self.model(**kwargs)
        layer_hiddens_cpu = [h[:, -1, :].detach().cpu().to(torch.float16) for h in outputs.hidden_states]
        last_hidden_gpu = outputs.hidden_states[-1][:, -1, :].detach().clone()
        return outputs.past_key_values, layer_hiddens_cpu, last_hidden_gpu

    @torch.no_grad()
    def decode_step_collect_layerwise(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Tuple,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ) -> Tuple[torch.Tensor, Tuple, List[torch.Tensor]]:
        """Single auto-regressive decode step with per-layer hidden collection.

        Args:
            token_ids: [B, 1]
            attention_mask: [B, total_seq_len] (including past)
            past_key_values: KV cache

        Returns:
            next_token: [B]
            past_key_values: updated
            layer_hiddens_cpu: list of [B, D] CPU float16
        """
        outputs = self.model(
            input_ids=token_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        layer_hiddens_cpu = [h[:, -1, :].detach().cpu().to(torch.float16) for h in outputs.hidden_states]
        logits = outputs.logits[:, -1, :]  # [B, vocab]

        if temperature > 0:
            logits = logits / temperature
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[remove] = float("-inf")
            probs = torch.softmax(sorted_logits, dim=-1)
            sampled_idx = torch.multinomial(probs, num_samples=1)  # [B, 1]
            next_token = sorted_indices.gather(1, sampled_idx).squeeze(-1)
        else:
            next_token = logits.argmax(dim=-1)

        return next_token, outputs.past_key_values, layer_hiddens_cpu
