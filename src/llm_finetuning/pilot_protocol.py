"""Answer-only pilot prompts with a fixed, reference-independent evidence budget."""

from __future__ import annotations

import re

PROTOCOL = "answer-only-v2; question-ranked evidence; fixed prompt budget; completion-only loss"
INSTRUCTION = (
    "Use the evidence to answer the financial question. "
    "Reply only with Final answer: <answer>. Do not explain."
)
_STOP_WORDS = set(
    (
        "a an and are as at be by did do for from how in is it of on or the to was what were with"
    ).split()
)


def _prompt_with_evidence(question, evidence, tokenizer):
    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": (f"{INSTRUCTION}\n\nEvidence:\n{evidence}\n\nQuestion: {question}"),
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def _encode_example(example, tokenizer, max_length, max_new_tokens=64):
    """Retain chat framing/question; select evidence without consulting the answer."""
    raw_prompt = example["prompt"]
    evidence_part, separator, question = raw_prompt.rpartition("\n\nQuestion: ")
    _, evidence_separator, evidence = evidence_part.partition("\n\nEvidence:\n")
    if not separator or not evidence_separator or not question.strip():
        raise ValueError("Pilot expects a normalized FinQA prompt with evidence and question")
    prompt_budget = max_length - max_new_tokens

    def encode(evidence_text):
        return tokenizer.encode(
            _prompt_with_evidence(question, evidence_text, tokenizer), add_special_tokens=False
        )

    if prompt_budget <= 0 or len(encode("")) > prompt_budget:
        raise ValueError("Token budget cannot retain the complete question and chat instructions")
    lines = [line for line in evidence.splitlines() if line.strip()]
    selected = {}
    if len(encode(evidence)) <= prompt_budget:
        retained = evidence
        selected = dict(enumerate(lines))
    else:
        terms = set(re.findall(r"\w+", question.casefold())) - _STOP_WORDS
        # Rank by question overlap, never by the reference answer; ties retain source order.
        ranked = sorted(
            range(len(lines)),
            key=lambda index: (
                -len(terms & set(re.findall(r"\w+", lines[index].casefold()))),
                index,
            ),
        )
        for index in ranked:
            candidate = {**selected, index: lines[index]}
            text = "\n".join(candidate[key] for key in sorted(candidate))
            if len(encode(text)) <= prompt_budget:
                selected = candidate
        # A single long paragraph may not fit; retain a bounded prefix rather than nothing.
        if not selected and ranked:
            index = ranked[0]
            low, high = 0, len(lines[index])
            while low < high:
                middle = (low + high + 1) // 2
                if len(encode(lines[index][:middle])) <= prompt_budget:
                    low = middle
                else:
                    high = middle - 1
            if low:
                selected[index] = lines[index][:low]
        retained = "\n".join(selected[key] for key in sorted(selected))
    prompt = encode(retained)
    # Only now consult the answer, after the generation prompt is fixed.
    completion = tokenizer.encode(f"Final answer: {example['answer']}", add_special_tokens=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Pilot tokenizer must define an EOS token")
    completion.append(tokenizer.eos_token_id)
    if len(completion) > max_new_tokens:
        raise ValueError(
            "Reference answer exceeds the fixed completion budget; increase it explicitly"
        )
    original_tokens = len(tokenizer.encode(evidence, add_special_tokens=False))
    retained_tokens = len(tokenizer.encode(retained, add_special_tokens=False))
    return {
        "input_ids": prompt + completion,
        "labels": [-100] * len(prompt) + completion,
        "prompt_ids": prompt,
        "id": example["id"],
        "answer": example["answer"],
        "context": {
            "prompt_tokens": len(prompt),
            "prompt_budget": prompt_budget,
            "evidence_original_tokens": original_tokens,
            "evidence_retained_tokens": retained_tokens,
            "evidence_omitted_tokens": max(0, original_tokens - retained_tokens),
            "evidence_truncated": retained != evidence,
            "selected_line_indices": sorted(selected),
            "no_source_evidence": evidence == "(no additional evidence provided)" or not lines,
        },
    }
