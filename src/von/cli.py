"""CLI entrypoint for Von."""

import json
import sys
import click
import uvicorn

from .api import decide as api_decide
from .api import system_one as api_system_one


@click.group()
@click.version_option(version="1.0.0", prog_name="von")
def main():
    """Von - Open Source System One Decision Model."""
    pass


@main.command()
@click.option("--host", default="0.0.0.0", help="Host interface to bind on.")
@click.option("--port", default=8000, type=int, help="Port to listen on.")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload.")
def serve(host: str, port: int, reload: bool):
    """Start the Von System One HTTP server."""
    click.echo(f"Starting Von Decision Server on http://{host}:{port}")
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
def decide(text: str, choices: str, instructions: str):
    """Classify input text among discrete choices."""
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
@click.argument("request_file", type=click.Path(exists=True))
def eval(request_file: str):
    """Evaluate a JSON request file containing state and questions."""
    with open(request_file, "r", encoding="utf-8") as f:
        data = json.load(f)

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
