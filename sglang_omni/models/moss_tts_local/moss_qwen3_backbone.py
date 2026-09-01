# SPDX-License-Identifier: Apache-2.0
"""Decode-only ``torch.compile`` indirection for the MOSS Local backbone."""

from __future__ import annotations

from typing import Optional, Union

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.qwen3 import Qwen3Model


class MossLocalQwen3Model(Qwen3Model):
    """Keep prefill eager while using compiled layers for decode batches."""

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            hidden_states = (
                self.embed_tokens(input_ids) if input_embeds is None else input_embeds
            )
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        forward_mode = getattr(forward_batch, "forward_mode", None)
        is_decode = bool(
            forward_mode is not None
            and getattr(forward_mode, "is_decode", lambda: False)()
        )
        compiled = getattr(self, "_compiled_decode_layers", None)
        max_bs = getattr(self, "_compiled_max_decode_bs", None)
        use_compiled = (
            is_decode
            and compiled is not None
            and (max_bs is None or int(hidden_states.shape[0]) <= int(max_bs))
        )
        layers = compiled if use_compiled else self.layers

        aux_hidden_states: list[torch.Tensor] = []
        for i in range(self.start_layer, self.end_layer):
            if i in self.layers_to_capture:
                aux_hidden_states.append(
                    hidden_states + residual if residual is not None else hidden_states
                )
            hidden_states, residual = layers[i](
                positions, hidden_states, forward_batch, residual
            )
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        if not aux_hidden_states:
            return hidden_states
        return hidden_states, aux_hidden_states


__all__ = ["MossLocalQwen3Model"]
