# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import collections
import json
import os
import re
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
from onnx.numpy_helper import from_array, to_array

from .split_onnx import OnnxSplitter, save_model


def _target_name(name, deco_digit=True, using_qairt_workflow=False):
    name = f"_{name}" if deco_digit and name.isdigit() else name
    # name = name.replace('.', '_')
    if not using_qairt_workflow:
        name = name.replace("/", "-")
    return name


def get_onnx_input_output_names(
    onnxfile, onnxmodel=None, deco_digit=True, using_qairt_workflow=False
):
    onnxmodel = _load_model(onnxfile) if onnxmodel is None else onnxmodel
    input_names = [
        _target_name(
            i.name, deco_digit=deco_digit, using_qairt_workflow=using_qairt_workflow
        )
        for i in onnxmodel.graph.input
    ]
    output_names = [
        _target_name(
            i.name, deco_digit=deco_digit, using_qairt_workflow=using_qairt_workflow
        )
        for i in onnxmodel.graph.output
    ]
    return input_names, output_names


def get_split_tensors(onnxfile, onnxmodel=None, include_first_input=True):
    """
    Model topology
            │ ←─────────  layers[0]  ────────────→ │       │ ←─────────  layers[-1]  ─────────────→ │
            │                                      │       │                                        │
    embed ────┬──────────── add0 ─┬─────────── add1 ── ┄┄┄  ─┬─────────────── add ─┬───────────── add ─── lmhead
            ↑ └─ norm ─ attn ─┘   └─ norm ─ ffn ─┘   ↑       ↑ └─ norm ─ attn ─┘   └─ norm ─ ffn ─┘   ↑
            │                                        │       │                                        │
            │                                        │       │                                        │
            valid splitting points
    """

    def get_nodes():
        model = _load_model(onnxfile) if onnxmodel is None else onnxmodel
        nodes = {i.name: i for i in model.graph.node}
        seq = {i.name: idx for idx, i in enumerate(model.graph.node)}
        producers = collections.defaultdict(lambda: None)
        producers.update({i.output[0]: i.name for i in model.graph.node})
        return nodes, seq, producers

    nodes, seq, producers = get_nodes()

    def maybe_skip_cast(a):
        if nodes[a].op_type == "Cast":
            return producers[nodes[a].input[0]]
        else:
            return a

    def can_visit(src, dst):
        if seq[src] < seq[dst]:
            return False
        stack, visited = collections.deque([src]), set()
        while stack:
            cur = stack.pop()
            if cur == dst:
                return True
            visited.add(cur)
            next_nodes = [
                producers[tensor]
                for tensor in nodes[cur].input
                if producers[tensor] is not None
            ]
            for name in next_nodes:
                if name not in visited and seq[name] >= seq[dst]:
                    stack.append(name)
        return False

    def is_residual_add(nodename, strict):
        if nodes[nodename].op_type != "Add":
            return False
        a, b = [producers[tensor] for tensor in nodes[nodename].input]
        if a is None or b is None:
            return False
        a = maybe_skip_cast(a)
        b = maybe_skip_cast(b)
        begin, end = (a, b) if seq[a] < seq[b] else (b, a)
        if strict and nodes[begin].op_type != "Add":
            return False
        return can_visit(end, begin)

    def get_add0(add1):
        a, b = [producers[tensor] for tensor in nodes[add1].input]
        a = maybe_skip_cast(a)
        b = maybe_skip_cast(b)
        add0 = a if seq[a] < seq[b] else b
        assert is_residual_add(add0, strict=False)
        return add0

    def get_layer0_input(add0):
        a, b = [producers[tensor] for tensor in nodes[add0].input]
        return a if seq[a] < seq[b] else b

    residual_add_names = [
        name for name in nodes.keys() if is_residual_add(name, strict=True)
    ]
    if len(residual_add_names) % 2 == 1:
        # 'add0' is missing in residual_adds
        add0 = get_add0(residual_add_names[0])
        residual_add_names.insert(0, add0)

    output_tensors = []
    if include_first_input:
        layer0_input = get_layer0_input(residual_add_names[0])
        output_tensors.append(nodes[layer0_input].output[0])
    output_tensors += [
        nodes[node].output[0] for i, node in enumerate(residual_add_names) if i % 2 == 1
    ]

    return output_tensors


def _load_model(onnxfile, load_external_data=False, model_cache={}):
    if onnxfile not in model_cache:
        model_cache[onnxfile] = onnx.load(
            onnxfile, load_external_data=load_external_data
        )
    return model_cache[onnxfile]


def _load_encoding(encodingfile, no_merge=False):
    all = {}
    if encodingfile is not None:
        with open(encodingfile) as json_file:
            quant_encoding_dict = json.load(json_file)
        if no_merge:
            return quant_encoding_dict
        all.update(quant_encoding_dict["activation_encodings"])
        all.update(quant_encoding_dict["param_encodings"])
    return all


def _save_encoding(encodings, encodingfile):
    with open(encodingfile, "wt") as json_file:
        json.dump(encodings, json_file, indent=4, sort_keys=True)


def embed_forecast_token_embeddings(onnxmodel, forecast_token_embeddings, base_dir):

    (embedding_table_name,) = [
        node.input[0] for node in onnxmodel.graph.node if node.op_type == "Gather"
    ]
    (embedding_table_proto,) = [
        i for i in onnxmodel.graph.initializer if i.name == embedding_table_name
    ]
    embedding_table = to_array(embedding_table_proto, base_dir=base_dir)

    assert (
        embedding_table.shape[1] == forecast_token_embeddings.shape[1]
    ), "Mismatching token embedding size"
    new_embedding_table = np.concatenate(
        (embedding_table, forecast_token_embeddings), axis=0
    )
    onnxmodel.graph.initializer.remove(embedding_table_proto)
    onnxmodel.graph.initializer.append(
        from_array(new_embedding_table, embedding_table_proto.name)
    )


def split_onnx_by_names(
    onnxfile,
    modelname,
    pickle_filedir,
    *list_of_output_tensors,
    output_dir=".",
    onnxmodel=None,
    encoding_file=None,
    using_qairt_workflow=False,
):
    encodings = None
    uses_lists = None
    if encoding_file is not None:
        with open(encoding_file) as f:
            encodings = json.load(f)
        uses_lists = isinstance(encodings["activation_encodings"], list)
        if uses_lists:
            # Convert encodings to dictionary
            encodings["activation_encodings"] = {
                v["name"]: v for v in encodings["activation_encodings"]
            }
            encodings["param_encodings"] = {
                v["name"]: v for v in encodings["param_encodings"]
            }

    onnx_to_artifacts_map = dict()
    onnxmodel = (
        _load_model(onnxfile, load_external_data=False)
        if onnxmodel is None
        else onnxmodel
    )
    splitter = OnnxSplitter(onnxmodel, verbose=False)
    base_dir = os.path.dirname(onnxfile)
    using_external_data = OnnxSplitter.is_using_external_data(onnxmodel)

    list_of_output_tensors = [i.split(",") for i in list_of_output_tensors]
    num_splits = len(list_of_output_tensors) + 1

    # 1. split model
    new_model_info = []
    for i, subgraph in enumerate(splitter.split(list_of_output_tensors)):
        new_basename = f"{modelname}_{i+1}_of_{num_splits}"
        input_tensor_names = [i.name for i in subgraph.input]
        output_tensor_names = [i.name for i in subgraph.output]
        new_model_info.append([new_basename, input_tensor_names, output_tensor_names])

        submodel = onnx.helper.make_model(
            subgraph, opset_imports=onnxmodel.opset_import
        )
        if (
            not using_external_data
            and submodel.ByteSize() < onnx.checker.MAXIMUM_PROTOBUF
        ):
            onnx.checker.check_model(submodel)

        if using_external_data:
            onnx.load_external_data_for_model(submodel, base_dir=base_dir)

        part_root_path = Path(output_dir) / (new_basename + ".aimet")
        part_root_path.mkdir(parents=True, exist_ok=True)

        newonnxfile = part_root_path / (new_basename + ".onnx")
        save_model(submodel, newonnxfile, using_external_data)

        # Save subset of encodings
        if encodings is not None:
            new_encodings = deepcopy(encodings)

            activation_names = (
                set(o for x in submodel.graph.node for o in x.output)
                | set(x.name for x in submodel.graph.input)
                | set(x.name for x in submodel.graph.output)
            )
            param_names = set(x.name for x in submodel.graph.initializer)

            for k in encodings["activation_encodings"]:
                if k not in activation_names:
                    del new_encodings["activation_encodings"][k]

            for k in encodings["param_encodings"]:
                if k not in param_names:
                    del new_encodings["param_encodings"][k]

            if uses_lists:
                # convert back
                new_encodings["activation_encodings"] = list(
                    new_encodings["activation_encodings"].values()
                )
                new_encodings["param_encodings"] = list(
                    new_encodings["param_encodings"].values()
                )

            new_encodings_path = part_root_path / (new_basename + ".encodings")
            with open(new_encodings_path, "w") as write_file:
                json.dump(new_encodings, write_file, indent=4, sort_keys=True)

    return onnx_to_artifacts_map


def _get_lm_head_sizes(onnxmodel):
    "Get dimensions of the LM head : embedding_size, vocab_size"
    lm_head_weight_name = next(
        node.input[1]
        for node in reversed(onnxmodel.graph.node)
        if node.op_type in ("Conv", "MatMul", "Gemm")
    )
    (lm_head_weight,) = [
        i for i in onnxmodel.graph.initializer if lm_head_weight_name == i.name
    ]
    if len(lm_head_weight.dims) == 2:
        embedding_size, vocab_size = lm_head_weight.dims
    else:
        (lm_head,) = [i for i in onnxmodel.graph.node if lm_head_weight.name in i.input]
        if lm_head.op_type == "Conv":
            attr_group = [i.i for i in lm_head.attribute if i.name == "group"]
            group = attr_group[0] if len(attr_group) == 1 else 1
            grouped_vocab, group_size, _, _ = lm_head_weight.dims
            vocab_size, embedding_size = grouped_vocab // group, group * group_size
        elif lm_head.op_type == "MatMul":
            group, group_size, vocab_size = lm_head_weight.dims
            embedding_size = group * group_size
        else:
            raise RuntimeError(f"Unexpected lm_head op_type:{lm_head}")

    return embedding_size, vocab_size


def fill_input_encodings_of_split(onnxmodel, encodingfile, output_tensor_list):

    changed = False
    encodings = _load_encoding(encodingfile, no_merge=True)
    enc_act, enc_param = encodings["activation_encodings"], encodings["param_encodings"]
    producer = {tensor: node for node in onnxmodel.graph.node for tensor in node.output}
    for split_tensor in output_tensor_list:
        if split_tensor not in enc_act:
            assert split_tensor in producer
            input_tensor = producer[split_tensor].input[0]  # use only 1st input
            if input_tensor in producer:
                while input_tensor not in enc_act and input_tensor not in enc_param:
                    input_tensor = producer[input_tensor].input[0]
                input_encoding = (
                    enc_act[input_tensor]
                    if input_tensor in enc_act
                    else enc_param[input_tensor]
                )
                enc_act[split_tensor] = input_encoding
                changed = True

    if changed:
        backup = f"{encodingfile}.bak"
        if not os.path.exists(backup):
            shutil.move(encodingfile, backup)
        _save_encoding(encodings, encodingfile)


def split_onnx(
    onnxfile,
    modelname,
    pickle_filedir,
    num_splits,
    num_layers_per_split: Optional[int] = None,
    output_dir="./",
    split_embedding=False,
    encoding_file=None,
    using_qairt_workflow=False,
):
    def _is_cache(layer, name):
        return re.search(f"past_(key|value)_{layer}_", name) is not None

    num_splits = int(num_splits)

    onnxmodel = _load_model(onnxfile, load_external_data=False)
    input_names, output_names = get_onnx_input_output_names(
        onnxfile,
        onnxmodel=onnxmodel,
        deco_digit=False,
        using_qairt_workflow=using_qairt_workflow,
    )
    output_tensor_list = get_split_tensors(
        onnxfile, onnxmodel=onnxmodel, include_first_input=split_embedding
    )

    # Infer the shape of per-layer tensors
    (input_ids,) = [i for i in onnxmodel.graph.input if i.name == "input_ids"]
    batch_size, seq_length = [i.dim_value for i in input_ids.type.tensor_type.shape.dim]

    embedding_size, vocab_size = _get_lm_head_sizes(onnxmodel)

    per_layer_output_value_info = [
        onnx.helper.make_tensor_value_info(
            name, onnx.TensorProto.FLOAT, [batch_size, seq_length, embedding_size]
        )
        for name in output_tensor_list
    ]
    onnxmodel.graph.value_info.extend(per_layer_output_value_info)

    names_to_split = []
    if split_embedding:
        first_output_tensors = output_tensor_list[0].split(",")
        fill_input_encodings_of_split(onnxmodel, encoding_file, first_output_tensors)
        names_to_split.append(output_tensor_list[0])
        output_tensor_list.pop(0)

    num_layers = len(output_tensor_list)
    if num_layers_per_split is None:
        num_layers_per_split = (
            ((num_layers - 1) // num_splits)
            if split_embedding
            else (num_layers // num_splits)
        )
    past_key_values = {
        layer: [output for output in output_names if _is_cache(layer, output)]
        for layer in range(num_layers)
    }

    for layer_end in range(num_layers_per_split, num_layers, num_layers_per_split):
        outputs = [output_tensor_list[layer_end - 1]]
        for layer in range(layer_end - num_layers_per_split, layer_end):
            outputs += past_key_values[layer]
        names_to_split.append(",".join(outputs))

    names_to_split = names_to_split[: num_splits - 1]
    assert (
        num_splits == len(names_to_split) + 1
    ), f"Failed to split into {num_splits} pieces!"
    return split_onnx_by_names(
        onnxfile,
        modelname,
        pickle_filedir,
        *names_to_split,
        output_dir=output_dir,
        onnxmodel=onnxmodel,
        encoding_file=encoding_file,
        using_qairt_workflow=using_qairt_workflow,
    )
