"""Example: Sponsor Form Triage using Hop.

Direct implementation of the scenario from the article:
Evaluating a sponsor inquiry with Noul, Choice, and Score in a single call.
"""

import json
import hop

state = {
    "opportunity": "link",
    "name": "Managed Postgres",
    "description": "We make a managed PostgreSQL hosting product and would like to sponsor the newsletter in October.",
}

questions = {
    "is_sponsor_inquiry": hop.noul(
        instructions="Does `description` ask to sponsor the site or newsletter?"
    ),
    "product_category": hop.choice(
        instructions="What kind of product is described by `name` and `description`?",
        criteria={
            "dev_tool": "Developer tools, hosting, APIs, SaaS for developers",
            "course": "Courses, books, or training",
            "unrelated": "Anything not aimed at developers",
        },
    ),
    "message_quality": hop.score(
        instructions="How specific is the request?",
        criteria=[
            "Generic template, no reference to this site",
            "Mentions the site but no concrete ask",
            "Concrete ask with a timeframe or product named",
        ],
    ),
}

# Run speculative fan-out (all 3 questions answered in one pass)
response = hop.system_one(state=state, questions=questions)

print("=== Hop Response ===")
print("Model:", response.model)
print("Usage:", response.usage)
print("\n=== Answers ===")
for q_id, answer in response.answers.items():
    print(f"[{q_id}]: {answer.model_dump()}")

# Business logic branching
answers = response.answers
if (
    answers["is_sponsor_inquiry"].noul > 0.8
    and answers["product_category"].choice == "dev_tool"
):
    print("\nAction: Automatically send rate card email!")
else:
    print("\nAction: Queue for manual human review.")
