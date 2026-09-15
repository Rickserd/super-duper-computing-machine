#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset preprocessing helpers.

The training/eval pipeline is built around ``AutoModelForSequenceClassification``
with a single text (or text-pair) input and an integer label. Multiple-choice
benchmarks such as HellaSwag don't fit that shape natively (they provide a
context plus a list of candidate endings). This module reformulates such tasks
into the standard single-sequence N-way classification format so they flow
through the existing tokenization, training, and inference code unchanged.
"""

from typing import List

from config import DatasetConfig


# Letters used to enumerate candidate answers in the rendered prompt.
_CHOICE_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def format_multiple_choice_text(context: str, endings: List[str]) -> str:
    """Render a context and its candidate endings into a single prompt string.

    Example output::

        A man is sitting on a roof. he
        A. starts pulling up roofing on a roof.
        B. is using wrap to wrap a pair of skis.
        C. is ripping level tiles off.
        D. is holding a rubik's cube.
        Answer:
    """
    context = (context or "").strip()
    lines = [context]
    for letter, ending in zip(_CHOICE_LETTERS, endings):
        lines.append(f"{letter}. {str(ending).strip()}")
    lines.append("Answer:")
    return "\n".join(lines)


def preprocess_multiple_choice_dataset(dataset, dataset_config: DatasetConfig):
    """Transform a raw multiple-choice ``DatasetDict`` into single-text classification.

    Produces two columns and drops all others:
      - ``dataset_config.text_column``: context + enumerated endings (string)
      - ``dataset_config.label_column``: gold choice index (int; -1 when unlabeled,
        as in held-out test splits)

    Works on a ``DatasetDict`` (all splits) or a single ``Dataset``.
    """
    context_col = dataset_config.context_column
    choices_col = dataset_config.choices_column
    text_col = dataset_config.text_column
    label_col = dataset_config.label_column

    if not context_col or not choices_col:
        raise ValueError(
            f"Dataset '{dataset_config.name}' is marked is_multiple_choice but is "
            f"missing context_column/choices_column configuration."
        )

    def _build(example):
        text = format_multiple_choice_text(example[context_col], example[choices_col])
        raw_label = example.get(label_col, "")
        label_str = str(raw_label).strip()
        label = int(label_str) if label_str != "" else -1
        return {text_col: text, label_col: label}

    # Determine which raw columns to drop. A DatasetDict exposes column_names as a
    # {split: [cols]} mapping; a bare Dataset exposes a flat list.
    column_names = dataset.column_names
    if isinstance(column_names, dict):
        any_split_cols = next(iter(column_names.values()))
    else:
        any_split_cols = column_names
    remove_cols = [c for c in any_split_cols if c not in (text_col, label_col)]

    return dataset.map(_build, remove_columns=remove_cols, desc="Formatting multiple-choice")
