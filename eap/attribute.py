from typing import Callable, List, Union, Optional, Literal
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_attention_mask
from tqdm import tqdm
from einops import einsum
import einops
import bitsandbytes.functional as bnbF
from .graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

cache = []

def make_hooks_and_matrices(model: HookedTransformer, graph: Graph, batch_size:int , n_pos:int, scores: torch.Tensor, detach=True):
    """Makes a matrix containing activation differences, and hooks to fill it and the score matrix up

    Args:
        model (HookedTransformer): model to attribute
        graph (Graph): graph to attribute
        batch_size (int): size of the particular batch you're attributing
        n_pos (int): size of the position dimension
        scores (Tensor): The scores tensor you intend to fill

    Returns:
        Tuple[Tuple[List, List, List], Tensor]: The final tensor ([batch, pos, n_src_nodes, d_model]) stores activation differences, i.e. corrupted activation - clean activation. 
        The first set of hooks will add in the activations they are run on (run these on corrupted input), while the second set will subtract out the activations they are run on (run these on clean input). 
        The third set of hooks will take in the gradients during the backwards pass and multiply it by the activation differences, adding this value in-place to the scores matrix that you passed in. 
    """
    activation_difference = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model), device='cuda', dtype=model.cfg.dtype)

    processed_attn_layers = set()
    fwd_hooks_clean = []
    fwd_hooks_corrupted = []
    bwd_hooks = []
    n_heads = model.cfg.n_heads
    
    def activation_hook(index, activations, hook, add : bool = True):
        """Hook to add/subtract activations to/from the activation difference matrix
        Args:
            index ([type]): forward index of the node
            activations ([type]): activations to add
            hook ([type]): hook (unused)
            add (bool, optional): whether to add or subtract. Defaults to True."""
        acts = activations.detach() if detach else activations
        if not add:
            acts = -acts
        try:
            activation_difference[:, :, index] += acts
        except RuntimeError as e:
            print("Activation Hook Error", hook.name, activation_difference[:, :, index].size(), acts.size(), index)
            raise e
    
    def gradient_hook(prev_index: Union[slice, int], bwd_index: Union[slice, int], gradients:torch.Tensor, hook):
        """Hook to multiply the gradients by the activations and add them to the scores matrix

        Args:
            prev_index (Union[slice, int]): index before which all nodes contribute to the present node
            bwd_index (Union[slice, int]): backward pass index of the node
            gradients (torch.Tensor): gradients of the node
            hook ([type]): hook
        """
        grads = gradients.detach()
        try:
            if grads.ndim == 3:
                grads = grads.unsqueeze(2)
            if model.cfg.use_split_qkv_input == False:
                # 将QKV还原为q_input, k_input, v_input
                if isinstance(bwd_index, slice):
                    layer = (prev_index - 1) // (n_heads + 1)
                    head = bwd_index.start % (3 * n_heads + 1)
                    if head < n_heads:
                        W = model.blocks[layer].attn.W_Q
                    elif head in range(n_heads, 2*n_heads):
                        W = model.blocks[layer].attn.W_K
                    elif head in range(2*n_heads, 3*n_heads):
                        W = model.blocks[layer].attn.W_V

                    for layer_name, data in model.state_dict().items():
                        if layer_name == f'blocks.{layer}.ln1.w':
                            w = data
                    scale = cache[f'blocks.{layer}.ln1.hook_scale']
                    if scale.ndim == 3:
                        scale = einops.repeat(
                            scale,
                            "batch pos 1 -> batch pos head_index 1",
                            head_index=n_heads
                        )
                    norm = cache[f'blocks.{layer}.ln1.hook_normalized']
                    if norm.ndim == 3:
                        norm = einops.repeat(
                            norm,
                            "batch pos d_model -> batch pos head_index d_model",
                            head_index=n_heads
                        )

                    if model.cfg.load_in_4bit == True:
                        W = bnbF.dequantize_4bit(W.t(), W.quant_state).to(grads.dtype)
                        W = einops.rearrange(W, "d_model (head_index d_head) -> head_index d_model d_head",  head_index = n_heads)
                    ln_grad = einops.einsum(grads, W, w, "batch pos head_index d_head, head_index d_model d_head, d_model -> batch pos head_index d_model")
                    dnorm = ln_grad
                    grads = dnorm / scale - (norm * (dnorm * norm).mean(-1, keepdim=True)) / scale

            s = einsum(activation_difference[:, :, :prev_index], grads,'batch pos forward hidden, batch pos backward hidden -> forward backward')
            s = s.squeeze(1)#.to(scores.device)
            scores[:prev_index, bwd_index] += s
        except RuntimeError as e:
            print("Gradient Hook Error", hook.name, activation_difference.size(), grads.size(), prev_index, bwd_index)
            raise e

    for name, node in graph.nodes.items():
        if isinstance(node, AttentionNode):
            if node.layer in processed_attn_layers:
                continue
            else:
                processed_attn_layers.add(node.layer)

        # exclude logits from forward
        if not isinstance(node, LogitNode):
            fwd_index = graph.forward_index(node)
            # fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index))) #hsc: clean_only
            fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
        if not isinstance(node, InputNode):
            prev_index = graph.prev_index(node)
            if isinstance(node, AttentionNode):
                for i, letter in enumerate('qkv'):
                    bwd_index = graph.backward_index(node, qkv=letter)
                    bwd_hooks.append((node.qkv_inputs[i], partial(gradient_hook, prev_index, bwd_index)))
            else:
                bwd_index = graph.backward_index(node)
                bwd_hooks.append((node.in_hook, partial(gradient_hook, prev_index, bwd_index)))
            
    return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference

def tokenize_plus(model: HookedTransformer, inputs: List[str]):
    """
    Tokenizes the input strings using the provided model.

    Args:
        model (HookedTransformer): The model used for tokenization.
        inputs (List[str]): The list of input strings to be tokenized.

    Returns:
        tuple: A tuple containing the following elements:
            - tokens (torch.Tensor): The tokenized inputs.
            - attention_mask (torch.Tensor): The attention mask for the tokenized inputs.
            - input_lengths (torch.Tensor): The lengths of the tokenized inputs.
            - n_pos (int): The maximum sequence length of the tokenized inputs.
    """
    tokens = model.to_tokens(inputs, prepend_bos=True, padding_side='right')
    attention_mask = get_attention_mask(model.tokenizer, tokens, True)
    input_lengths = attention_mask.sum(1)
    n_pos = attention_mask.size(1)
    return tokens, attention_mask, input_lengths, n_pos

def get_scores_eap(model: HookedTransformer, graph: Graph, dataloader:DataLoader, metric: Callable[[Tensor], Tensor], quiet=False):
    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

        with model.hooks(fwd_hooks=fwd_hooks_corrupted):
            corrupted_logits = model(corrupted_tokens, attention_mask=attention_mask)

        with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
            logits = model(clean_tokens, attention_mask=attention_mask)
            metric_value = metric(logits, corrupted_logits, input_lengths, label)
            metric_value.backward()

    scores /= total_items

    return scores

def get_scores_eap_ig(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], steps=30, quiet=False):
    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)

    names_filter = []    
    for layer in range(model.cfg.n_layers):
        names_filter.append(f'blocks.{layer}.ln1.hook_scale')
        names_filter.append(f'blocks.{layer}.ln1.hook_normalized')

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
    # for clean, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted) #hsc: clean_only

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask) #hsc: clean_only

            input_activations_corrupted = activation_difference[:, :, graph.forward_index(graph.nodes['input'])].clone()

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                # clean_logits = model(clean_tokens, attention_mask=attention_mask)
                global cache
                clean_logits, cache = model.run_with_cache(clean_tokens, attention_mask=attention_mask, names_filter=names_filter)

            input_activations_clean = input_activations_corrupted - activation_difference[:, :, graph.forward_index(graph.nodes['input'])]

        def input_interpolation_hook(k: int):
            def hook_fn(activations, hook):
                new_input = input_activations_corrupted + (k / steps) * (input_activations_clean - input_activations_corrupted)
                new_input.requires_grad = True 
                return new_input
            return hook_fn

        total_steps = 0
        for step in range(1, steps+1):
            total_steps += 1
            with model.hooks(fwd_hooks=[(graph.nodes['input'].out_hook, input_interpolation_hook(step))], bwd_hooks=bwd_hooks):
                logits = model(clean_tokens, attention_mask=attention_mask)
                metric_value = metric(logits, clean_logits, input_lengths, label)
                metric_value.backward()

    scores /= total_items
    scores /= total_steps

    return scores

def get_scores_ig_partial_activations(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], steps=30, quiet=False):
    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_tokens, attention_mask=attention_mask)

        def output_interpolation_hook(k: int, difference: torch.Tensor):
            def hook_fn(activations: torch.Tensor, hook):
                new_output = activations + (1 - k / steps) * difference
                return new_output
            return hook_fn

        total_steps = 0
        for step in range(1, steps+1):
            total_steps += 1
            with model.hooks(fwd_hooks=[(node.out_hook, output_interpolation_hook(step, activation_difference[:, :, graph.forward_index(node)])) for node in graph.nodes.values() if not isinstance(node, LogitNode)], bwd_hooks=bwd_hooks):
                logits = model(clean_tokens, attention_mask=attention_mask)
                metric_value = metric(logits, clean_logits, input_lengths, label)
                metric_value.backward()

    scores /= total_items
    scores /= total_steps

    return scores


def get_scores_ig_activations(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], steps=30, quiet=False, ablate_all_at_once=False):
    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        detach = bool(ablate_all_at_once)
        (_, _, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, detach=detach)
        (fwd_hooks_corrupted, _, _), activations_corrupted = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, detach=detach)
        (fwd_hooks_clean, _, _), activations_clean = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, detach=detach)


        with model.hooks(fwd_hooks=fwd_hooks_corrupted):
            _ = model(corrupted_tokens, attention_mask=attention_mask)

        with model.hooks(fwd_hooks=fwd_hooks_clean):
            clean_logits = model(clean_tokens, attention_mask=attention_mask)

        activation_difference += activations_corrupted.clone().detach() - activations_clean.clone().detach()

        def output_interpolation_hook(k: int, clean: torch.Tensor, corrupted: torch.Tensor):
            def hook_fn(activations: torch.Tensor, hook):
                alpha = k/steps
                new_output = alpha * clean + (1 - alpha) * corrupted
                return new_output
            return hook_fn

        total_steps = 0

        nodeslist = [[graph.nodes['input']]]
        for layer in range(graph.cfg['n_layers']):
            nodeslist.append([graph.nodes[f'a{layer}.h{head}'] for head in range(graph.cfg['n_heads'])])
            nodeslist.append([graph.nodes[f'm{layer}']])

        if ablate_all_at_once:
            nodeslist = [node for node in graph.nodes.values() if not isinstance(node, LogitNode)]

        for nodes in nodeslist:
            for step in range(1, steps+1):
                total_steps += 1
                fwd_hooks = []
                for node in nodes:
                    clean_acts = activations_clean[:, :, graph.forward_index(node)]
                    corrupted_acts = activations_corrupted[:, :, graph.forward_index(node)]
                    fwd_hooks.append((node.out_hook, output_interpolation_hook(step, clean_acts, corrupted_acts)))

                with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
                    logits = model(clean_tokens, attention_mask=attention_mask)
                    metric_value = metric(logits, clean_logits, input_lengths, label)

                    metric_value.backward(retain_graph=True)

    scores /= total_items
    scores /= total_steps

    return scores

def get_scores_clean_corrupted(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], quiet=False):
    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_tokens, attention_mask=attention_mask)


        total_steps = 2
        with model.hooks(bwd_hooks=bwd_hooks):
            logits = model(clean_tokens, attention_mask=attention_mask)
            metric_value = metric(logits, clean_logits, input_lengths, label)
            metric_value.backward()
            #model.zero_grad()

            corrupted_logits = model(corrupted_tokens, attention_mask=attention_mask)
            corrupted_metric_value = metric(corrupted_logits, clean_logits, input_lengths, label)
            corrupted_metric_value.backward()
            #model.zero_grad()

    scores /= total_items
    scores /= total_steps

    return scores

allowed_aggregations = {'sum', 'mean', 'l2'}        
def attribute(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], aggregation='sum', method: Union[Literal['EAP', 'EAP-IG', 'EAP-IG-partial-activations', 'EAP-IG-activations', 'clean-corrupted']]='EAP-IG', ig_steps: Optional[int]=5, quiet=False):
    if aggregation not in allowed_aggregations:
        raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')
        
    if method != 'EAP' and method != 'clean-corrupted':
        assert ig_steps is not None, f"ig_steps must be set for method {method}"

    if method == 'EAP':
        scores = get_scores_eap(model, graph, dataloader, metric, quiet=quiet)
    elif method == 'EAP-IG':
        scores = get_scores_eap_ig(model, graph, dataloader, metric, steps=ig_steps, quiet=quiet)
    elif method == 'EAP-IG-partial-activations':
        scores = get_scores_ig_partial_activations(model, graph, dataloader, metric, steps=ig_steps, quiet=quiet)
    elif method == 'EAP-IG-activations':
        scores = get_scores_ig_activations(model, graph, dataloader, metric, steps=ig_steps, quiet=quiet)
    elif method  == 'clean-corrupted':
        scores = get_scores_clean_corrupted(model, graph, dataloader, metric, quiet=quiet)
    else:
        raise ValueError(f"integrated_gradients must be in ['EAP', 'EAP-IG', 'EAP-IG-partial-activations', 'EAP-IG-activations', 'clean-corrupted'], but got {method}")

    if aggregation == 'mean':
        scores /= model.cfg.d_model
    elif aggregation == 'l2':
        scores = torch.linalg.vector_norm(scores, ord=2, dim=-1)
        
    scores = scores.cpu().numpy()

    for edge in tqdm(graph.edges.values(), total=len(graph.edges)):
        edge.score = scores[graph.forward_index(edge.parent, attn_slice=False), graph.backward_index(edge.child, qkv=edge.qkv, attn_slice=False)]

def numpy(a, decimals=None):
    v = np.array(a) if isinstance(a, list) else a.detach().cpu().numpy()
    if decimals is not None: v = v.round(decimals)
    return v

def topk_md(tensor, k, largest=True, transpose=False):
    k = min(tensor.numel(), k)
    if tensor.ndim == 1:
        values, indices = tensor.topk(k, largest=largest)
        return indices.numpy(), values.numpy()
    values, indices = tensor.flatten().topk(k, largest=largest)
    values, indices = values.cpu(), indices.cpu()
    # https://stackoverflow.com/questions/64241325/top-k-indices-of-a-multi-dimensional-tensor
    rows, cols = np.unravel_index(indices.numpy(), tensor.shape)
    return (rows, cols, values.numpy()) if not transpose else list(zip(rows, cols, values.numpy()))

def _plot_attn(attn, tokens, ytokens=None, ystart=None, ystop=None, y_pos=None, x_pos=None, topk=None,
            use_imshow=False, annot=False, figsize=(10, 10), fontsize=10, transpose=False, ax=None):
    # ytokens = ytokens or tokens
    ytokens = tokens 
    if y_pos is None and topk is not None:
        attn_ = attn.clone(); attn_[:, 0] = 0  # null attn to start pos is ignored
        y_pos, x_pos, _ = topk_md(attn_, k=topk)
    if ystart is not None:
        ystop = ystop or attn.size(0)
        attn = attn[ystart: ystop]
        ytokens = ytokens[ystart: ystop]
        if y_pos is not None: y_pos = [p - ystart for p in y_pos]
    square = True # attn.size(0) == attn.size(1)
    if ax is None:
        if not square:
            figsize2 = (attn.size(1), attn.size(0))
            a = max(s1 / s2 for s1, s2 in zip(figsize, figsize2))  # min
            figsize = [s * a for s in figsize2]
    if transpose:
        attn = attn.T
        tokens, ytokens = ytokens, tokens
        x_pos, y_pos = y_pos, x_pos
        figsize = figsize[::-1]
    if ax is None: plt.figure(figsize=figsize)

    if use_imshow:
        ax.imshow(attn)#, cmap='hot')
        ax.set_xticks(np.arange(0, attn.size(1), 1)); ax.set_xticklabels(tokens[0])
        ax.set_yticks(np.arange(0, attn.size(0), 1)); ax.set_yticklabels(ytokens[0])
    else:
        kwargs = dict(linewidths=0.1, linecolor='grey') if y_pos is None else {}
        ax = sns.heatmap(numpy(attn), square=square, cbar=False, annot=annot, fmt='d',
            xticklabels=tokens, yticklabels=ytokens, ax=ax, **kwargs)
    _ = ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=fontsize, rotation=90)
    _ = ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=fontsize, rotation=0)
    if transpose: ax.tick_params(right=True, labelright=True, left=False, labelleft=False)#, top=True, labeltop=True) # cause x3 slowdown!
    kwargs = dict(linewidths=0.5 * 2, color='grey')
    if y_pos is not None: ax.hlines(y=y_pos, xmin=0, xmax=attn.size(1)-0.5*use_imshow, **kwargs)  # max-0.5 for imshow
    if x_pos is not None: ax.vlines(x=x_pos, ymin=0, ymax=attn.size(0)-0.5*use_imshow, **kwargs)
    plt.show()