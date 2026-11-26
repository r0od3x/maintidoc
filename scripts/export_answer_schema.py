"""Export the AnswerV1 JSON schema for constrained decoding.

This is what replaces the hand-written `SCHEMA` dict of the prototype notebook.
The notebook and the API now constrain generation with the same object, so the
two can no longer drift apart.

    uv run python scripts/export_answer_schema.py            # print
    uv run python scripts/export_answer_schema.py -o s.json  # write

In the notebook, swap the literal dict for:

    from maintidoc.contracts import AnswerV1, answer_json_schema
    result = client.chat(model=..., messages=..., format=answer_json_schema())
    answer = AnswerV1.model_validate_json(result.message.content)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from maintidoc.contracts import answer_json_schema


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, help="Fichier de sortie JSON")
    args = parser.parse_args(argv)

    payload = json.dumps(answer_json_schema(), indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"Schéma écrit dans {args.output}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
