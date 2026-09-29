"""CLI entrypoint for Von."""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Union
import click
import uvicorn

from .engine import VON_CURRENT_ALIASES, VON_VERSION
from .api import decide as api_decide
from .api import judge as api_judge
from .api import rate as api_rate
from .api import system_one as api_system_one
from .device import _detect_device, get_device_description


@click.group()
@click.version_option(version="1.0.0", prog_name="von")
def main():
    """Von - Open Source System One Decision Model."""
    pass


@main.command()
@click.option("--host", default="0.0.0.0", help="Host interface to bind on.")
@click.option("--port", default=8000, type=int, help="Port to listen on.")
@click.option(
    "--model",
    "backend",
    default=f"von-{VON_VERSION}",
    # Derived from the engine rather than restated: a hardcoded copy had already
    # drifted, so the CLI rejected von-latest and default while the library
    # accepted them.
    type=click.Choice(sorted(VON_CURRENT_ALIASES)),
    help=f"Von model version to load (Von {VON_VERSION} is the only model).",
)
@click.option("--device", default="auto", help="Compute device: 'auto', 'cuda', 'rocm', 'mps', 'openvino', 'dml', 'cpu'.")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload.")
@click.option(
    "--max-state-tokens",
    default=None,
    type=int,
    help="Middle-truncate states longer than this many tokens (env VON_MAX_STATE_TOKENS). "
         "Default 8192 = the encoder window; lower it for a hard latency ceiling.",
)
@click.option(
    "--chains",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Directory of TOML chain-of-options definitions (env VON_CHAINS_DIR). Default: the bundled library.",
)
@click.option("--no-chains", is_flag=True, default=False, help="Disable chain-of-options (VON_CHAINS_DIR=off).")
@click.option(
    "--on-overflow",
    type=click.Choice(["truncate", "refuse"]),
    default=None,
    help="State longer than the window: middle-truncate with a warning (default) or refuse with HTTP 422 "
         "(env VON_ON_OVERFLOW). Use refuse under no-truncation benchmark rules.",
)
@click.option(
    "--noul-decision",
    type=click.Choice(["band", "raw"]),
    default=None,
    help="Noul P(yes): 'band' (default) commits every answer outside the 0.2..0.8 abstention band, "
         "'raw' returns the calibrated posterior unchanged (env VON_NOUL_DECISION).",
)
def serve(host: str, port: int, backend: str, device: str, reload: bool, max_state_tokens: Optional[int],
          chains: Optional[str], no_chains: bool, on_overflow: Optional[str], noul_decision: Optional[str]):
    """Start the Von System One HTTP server."""
    if on_overflow:
        os.environ["VON_ON_OVERFLOW"] = on_overflow
    if noul_decision:
        os.environ["VON_NOUL_DECISION"] = noul_decision
    os.environ["VON_BACKEND"] = backend
    if no_chains:
        os.environ["VON_CHAINS_DIR"] = "off"
    elif chains:
        os.environ["VON_CHAINS_DIR"] = chains
    if max_state_tokens is not None:
        os.environ["VON_MAX_STATE_TOKENS"] = str(max_state_tokens)
    if device and device != "auto":
        os.environ["VON_DEVICE"] = device
    dev_obj = _detect_device(device)
    dev_desc = get_device_description(dev_obj)
    click.echo(f"Starting Von Decision Server [{backend} on {dev_desc}] on http://{host}:{port}")
    uvicorn.run("von.server:app", host=host, port=port, reload=reload)


@main.command()
@click.argument("text")
@click.option(
    "-c",
    "--choices",
    required=True,
    help="Comma-separated choices (e.g. 'billing,bug_report,feature_request').",
)
@click.option(
    "-i",
    "--instructions",
    default="Which option best describes the input?",
    help="Instructions for classification.",
)
@click.option(
    "--device",
    default="auto",
    help="Compute device: 'auto', 'cuda', 'rocm', 'mps', 'openvino', 'dml', 'cpu'.",
)
def decide(text: str, choices: str, instructions: str, device: str):
    """Classify input text among discrete choices."""
    if device and device != "auto":
        os.environ["VON_DEVICE"] = device
    opts = [c.strip() for c in choices.split(",") if c.strip()]
    if not opts:
        click.echo("Error: At least one choice must be provided.", err=True)
        sys.exit(1)

    ans = api_decide(state=text, choices=opts, instructions=instructions)
    click.echo(
        json.dumps(
            {
                "choice": ans.choice,
                "confidence": ans.confidence,
                "probabilities": ans.probabilities,
            },
            indent=2,
        )
    )


@main.command()
@click.argument("text")
@click.option(
    "-i",
    "--instructions",
    required=True,
    help="Boolean judgment question (e.g. 'Is the server down?').",
)
@click.option(
    "--pos",
    default="",
    help="Explicit criteria description for True condition.",
)
@click.option(
    "--neg",
    default="",
    help="Explicit criteria description for False condition.",
)
@click.option(
    "--device",
    default="auto",
    help="Compute device: 'auto', 'cuda', 'rocm', 'mps', 'openvino', 'dml', 'cpu'.",
)
def judge(text: str, instructions: str, pos: str, neg: str, device: str):
    """Evaluate a yes/no judgment (Noul) and return the probability."""
    if device and device != "auto":
        os.environ["VON_DEVICE"] = device
    crit = {}
    if pos:
        crit["true"] = pos
    if neg:
        crit["false"] = neg

    prob = api_judge(state=text, instructions=instructions, criteria=crit or None)
    click.echo(
        json.dumps(
            {
                "type": "noul",
                "instructions": instructions,
                "noul": prob,
            },
            indent=2,
        )
    )


@main.command()
@click.argument("text")
@click.option(
    "-l",
    "--levels",
    required=True,
    help="Comma-separated descriptions of ordered levels from 0 to N-1.",
)
@click.option(
    "-i",
    "--instructions",
    default="Rate where the state falls on this scale:",
    help="Instructions for rating.",
)
@click.option(
    "--device",
    default="auto",
    help="Compute device: 'auto', 'cuda', 'rocm', 'mps', 'openvino', 'dml', 'cpu'.",
)
def rate(text: str, levels: str, instructions: str, device: str):
    """Rate text on an ordered multi-level scale (Score)."""
    if device and device != "auto":
        os.environ["VON_DEVICE"] = device
    lvl_list: List[Union[str, Dict[str, Any]]] = [lvl.strip() for lvl in levels.split(",") if lvl.strip()]
    if len(lvl_list) < 2:
        click.echo("Error: At least two levels must be provided.", err=True)
        sys.exit(1)

    ans = api_rate(state=text, criteria=lvl_list, instructions=instructions)
    click.echo(
        json.dumps(
            {
                "type": "score",
                "score": ans.score,
                "confidence": ans.confidence,
                "legend": ans.legend,
                "probabilities": ans.probabilities,
            },
            indent=2,
        )
    )


@main.command()
@click.argument("request_file", type=click.Path(exists=True))
def eval(request_file: str):
    """Evaluate a JSON request file containing state and questions."""
    try:
        with open(request_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError as exc:
        click.echo(f"Error: cannot read {request_file}: {exc}", err=True)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        click.echo(f"Error: {request_file} is not valid JSON: {exc}", err=True)
        sys.exit(1)

    state = data.get("state")
    questions = data.get("questions")
    model = data.get("model", "von-latest")

    if state is None or questions is None:
        click.echo("Error: JSON must contain 'state' and 'questions' fields.", err=True)
        sys.exit(1)

    resp = api_system_one(state=state, questions=questions, model=model)
    click.echo(json.dumps(resp.model_dump(), indent=2))


if __name__ == "__main__":
    main()
