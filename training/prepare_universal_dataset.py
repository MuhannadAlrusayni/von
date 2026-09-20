"""Universal Operational Decision Dataset Generator (Phase 4).

Generates a 200,000-sample balanced training corpus across 49 distinct System One
decision tasks (Choice, Noul, Score) covering:
1. Compliance & Regulatory (Hazmat, Vegan, Allergens, Fair Housing, Policies)
2. Security & DevOps (SQLi, Phishing, Secrets, PII, Commits, On-call)
3. Operations & Commerce (Triage, Expenses, Returns, Delivery, Municipal)
4. Content & Linguistics (Grammar taxonomy, Formality, Reading level, Moderation)
5. Clinical & Severity Triage (Symptoms, Veterinary, Weather, Churn, Insurance)
6. Contextual Reasoning & Core (Calendar conflicts, Deduplication, ANLI, WANLI)
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional
from datasets import load_dataset


# =====================================================================
# 1. Compliance & Regulatory Cluster (Noul & Policy Verification)
# =====================================================================

def generate_hazmat_cases(n: int = 4000) -> List[dict]:
    instruction = "Is this shipment restricted as dangerous goods / hazardous materials for air transport?"
    criteria = {
        "true": "Contains Class 1-9 dangerous goods (lithium batteries >100Wh, flammable liquids, aerosols, compressed gases, or toxic substances)",
        "false": "Standard non-hazardous consumer merchandise safe for regular cargo or passenger aircraft",
    }
    options = [
        {"id": "yes", "description": criteria["true"]},
        {"id": "no", "description": criteria["false"]},
    ]

    hazmat_pool = [
        "Pallet containing 50x 250Wh lithium-ion e-bike replacement batteries.",
        "Box of 24 cans of aerosol spray paint, flammable solvent propellant.",
        "Two 5-liter containers of 99% pure isopropyl alcohol.",
        "Compressed argon gas cylinders under 200 bar pressure.",
        "Industrial lead-acid automotive battery filled with acid electrolyte.",
        "Acetone solvent cleaning solution in plastic drums.",
        "Matches and pyrotechnic road flares for emergency roadside kits.",
        "Vials of biological infectious specimens preserved in formalin.",
    ]
    safe_pool = [
        "Carton of 100 printed cotton graphic t-shirts in polybags.",
        "Crate of hardcover fiction novels and paper notebooks.",
        "Box of ceramic coffee mugs packed with biodegradable bubble wrap.",
        "Three sets of stainless steel kitchen cutlery and silicone spatulas.",
        "Plastic children's building blocks and wooden jigsaw puzzles.",
        "Rolls of polyester carpet samples for trade show booth flooring.",
        "Organic dried pinto beans in 25kg sealed burlap sacks.",
        "Aluminum USB-C charging cables (cables only, no battery cells).",
    ]

    records = []
    for _ in range(n // 2):
        records.append({
            "state": random.choice(hazmat_pool),
            "question": instruction,
            "options": options,
            "label": "yes",
            "source": "hazmat_pos",
        })
        records.append({
            "state": random.choice(safe_pool),
            "question": instruction,
            "options": options,
            "label": "no",
            "source": "hazmat_neg",
        })
    return records


def generate_dietary_vegan_cases(n: int = 4000) -> List[dict]:
    instruction = "Is this dish or ingredient list strictly vegan?"
    criteria = {
        "true": "Contains exclusively plant-based ingredients; free from all meat, fish, dairy, eggs, honey, and animal derivatives",
        "false": "Contains animal products or animal by-products (dairy, whey, eggs, honey, gelatin, or lard)",
    }
    options = [
        {"id": "yes", "description": criteria["true"]},
        {"id": "no", "description": criteria["false"]},
    ]

    vegan_pool = [
        "Ingredients: Rolled oats, almond milk, chia seeds, fresh blueberries, and pure maple syrup.",
        "Bowl of steamed jasmine rice topped with crispy smoked tofu, edamame, avocado, and sesame-tamari dressing.",
        "Lentil soup made with brown lentils, carrots, celery, crushed tomatoes, olive oil, and vegetable broth.",
        "Roasted cauliflower tacos on corn tortillas with black beans, pickled red onions, and cashew crema.",
        "Smoothie made with frozen bananas, spinach, peanut butter, oat milk, and spirulina powder.",
        "Whole wheat sourdough toast topped with smashed avocado, hemp seeds, nutritional yeast, and chili flakes.",
    ]
    non_vegan_pool = [
        "Fresh garden salad with roasted beets, candied walnuts, and crumbled goat cheese with honey vinaigrette.",
        "Granola bar made with whole oats, dried cranberries, almonds, and raw clover honey.",
        "Vegetable ramen in rich mushroom broth, topped with bamboo shoots and a soft-boiled egg.",
        "Cornbread muffin made with stone-ground cornmeal, organic butter, and whole milk.",
        "Strawberry gummy candies made with sugar, fruit puree, and beef gelatin.",
        "Veggie burger patty bound with egg whites and whey protein concentrate.",
        "French fries fried in traditional beef tallow alongside vegetable oil.",
    ]

    records = []
    for _ in range(n // 2):
        records.append({
            "state": random.choice(vegan_pool),
            "question": instruction,
            "options": options,
            "label": "yes",
            "source": "vegan_pos",
        })
        records.append({
            "state": random.choice(non_vegan_pool),
            "question": instruction,
            "options": options,
            "label": "no",
            "source": "vegan_neg",
        })
    return records


def generate_fair_housing_cases(n: int = 4000) -> List[dict]:
    instruction = "Does this rental listing text violate the Fair Housing Act or non-discrimination rules?"
    criteria = {
        "true": "Expresses preference or discrimination based on race, religion, sex, disability, familial status (kids), or national origin",
        "false": "Compliant listing setting legitimate financial, occupancy, or property requirements (credit score, no smoking)",
    }
    options = [
        {"id": "yes", "description": criteria["true"]},
        {"id": "no", "description": criteria["false"]},
    ]

    violation_pool = [
        "Charming 2BR apartment. Perfect for quiet mature singles or working professionals only — no children allowed.",
        "Spacious studio in quiet neighborhood. Walking distance to Catholic church; ideal for Christian tenants.",
        "Cozy 1BR cottage. Looking for clean female tenant only to share property grounds.",
        "Modern condo for rent. English-speakers only please due to building association policies.",
        "Historic rowhouse. Not suitable for individuals with physical wheelchairs or mobility impairments.",
        "Executive apartment. Adults only; building policy strictly prohibits families with young kids.",
    ]
    compliant_pool = [
        "Bright 2BR apartment in downtown core. Requires 680+ credit score and verifiable income 3x monthly rent.",
        "Spacious 3BR suburban home with fenced yard. Non-smoking property throughout; pet deposit required.",
        "Studio apartment near transit station. First month rent and one month security deposit due at lease signing.",
        "Renovated 1BR unit. Tenant is responsible for electric and internet; water and trash included.",
        "Quiet suburban townhouse. Background check and prior landlord references required for all applicants.",
        "Corner 2BR unit available Nov 1. Maximum occupancy capped at 4 persons per municipal housing code.",
    ]

    records = []
    for _ in range(n // 2):
        records.append({
            "state": random.choice(violation_pool),
            "question": instruction,
            "options": options,
            "label": "yes",
            "source": "fair_housing_pos",
        })
        records.append({
            "state": random.choice(compliant_pool),
            "question": instruction,
            "options": options,
            "label": "no",
            "source": "fair_housing_neg",
        })
    return records


def generate_travel_policy_cases(n: int = 4000) -> List[dict]:
    instruction = "Does this corporate travel expense violate standard company travel policy?"
    criteria = {
        "true": "Violates travel policy rules (first/business class flight under 6 hours, luxury hotel exceeding cap, personal leisure)",
        "false": "Compliant business travel expense within standard economy limits, approved per diem, and authorized vendors",
    }
    options = [
        {"id": "yes", "description": criteria["true"]},
        {"id": "no", "description": criteria["false"]},
    ]

    violation_pool = [
        "Booked First Class cabin for a 90-minute domestic hop from Seattle to Portland, total $850.",
        "Five-star Ritz Carlton suite in Chicago at $750/night when company rate cap is $250/night.",
        "Mini-bar charges totaling $140 for premium alcoholic spirits charged to the corporate room folio.",
        "Purchased weekend ski lift passes for spouse during business trip extension.",
        "Rented a luxury sports convertible for an on-site client meeting instead of standard sedan.",
    ]
    compliant_pool = [
        "Economy flight from San Francisco to Austin for annual company offsite, $320.",
        "Courtyard by Marriott room at $195/night within approved metropolitan per diem limits.",
        "Uber rideshare from Dallas airport to client headquarters, $48 including tip.",
        "Airport parking for 3 days in long-term economy lot during client visits, $65 total.",
        "Standard midsize sedan rental from enterprise car fleet for 2 days, $110 total.",
    ]

    records = []
    for _ in range(n // 2):
        records.append({
            "state": random.choice(violation_pool),
            "question": instruction,
            "options": options,
            "label": "yes",
            "source": "travel_policy_pos",
        })
        records.append({
            "state": random.choice(compliant_pool),
            "question": instruction,
            "options": options,
            "label": "no",
            "source": "travel_policy_neg",
        })
    return records


# =====================================================================
# 2. Security & DevOps Cluster
# =====================================================================

def generate_sql_injection_cases(n: int = 4000) -> List[dict]:
    instruction = "Does this user input or query snippet represent an attempted SQL injection attack?"
    criteria = {
        "true": "Contains malicious SQL syntax designed to break out of data context (tautologies, UNION SELECT, stacked queries, comments)",
        "false": "Benign search query, normal text with punctuation, or standard alphanumeric query string",
    }
    options = [
        {"id": "yes", "description": criteria["true"]},
        {"id": "no", "description": criteria["false"]},
    ]

    sqli_pool = [
        "admin' OR '1'='1' --",
        "1; DROP TABLE users; --",
        "' UNION SELECT username, password_hash, email FROM accounts --",
        "test' AND 1=CONVERT(int, (SELECT @@version)) --",
        "105 OR 1=1 ORDER BY 1#",
        "' UNION ALL SELECT NULL, NULL, @@version, NULL --",
        "admin'/*",
        "1' OR sleep(5)#",
    ]
    safe_pool = [
        "Looking for John O'Connor's order confirmation from yesterday.",
        "SELECT * FROM products WHERE category = $1 AND active = true;",
        "Search: Men's running shoes size 11.5 blue",
        "Customer ID: 4892019",
        "Feedback: It's a great platform, don't change the UI!",
        "Query: Apple MacBook Pro 16-inch M3 Max",
        "Filter tag: tags=clothing&sort=price_asc&page=2",
    ]

    records = []
    for _ in range(n // 2):
        records.append({
            "state": random.choice(sqli_pool),
            "question": instruction,
            "options": options,
            "label": "yes",
            "source": "sqli_pos",
        })
        records.append({
            "state": random.choice(safe_pool),
            "question": instruction,
            "options": options,
            "label": "no",
            "source": "sqli_neg",
        })
    return records


def generate_phishing_cases(n: int = 4000) -> List[dict]:
    instruction = "Is this incoming email a phishing or social engineering threat?"
    criteria = {
        "true": "Contains credential harvesting indicators, spoofed identity, artificial urgency, or suspicious links",
        "false": "Legitimate business correspondence, system notification, or customer message",
    }
    options = [
        {"id": "yes", "description": criteria["true"]},
        {"id": "no", "description": criteria["false"]},
    ]

    phish_pool = [
        "From: IT-Support <support@micros0ft-verify.com>\nSubject: Immediate Action Required: Account Deletion in 24 Hours\nClick here to verify password.",
        "From: HR Department <hr@company-payroll-portal.net>\nSubject: Updated 2026 W2 Form Attached\nPlease enter your SSO credentials to download.",
        "From: CEO <executive.urgent.office@gmail.com>\nSubject: Quick task while I am in meeting\nNeed you to buy $500 Apple gift cards right now for client gifts.",
        "From: DocuSign Notification <security@docusign-verification-service.info>\nSubject: Wire Transfer Agreement Pending Signature\nLog in with Microsoft to view document.",
        "From: Bank Alert <service@chase-secure-account-center.com>\nSubject: Suspicious Login Detected from Russia\nConfirm your debit PIN to unlock card.",
    ]
    safe_pool = [
        "From: GitHub <notifications@github.com>\nSubject: [wfzyx/von] Pull request #4 merged by maintainer\nCommit hash a94aa36 landed on master.",
        "From: Google Calendar <calendar-notification@google.com>\nSubject: Reminder: Sprint Planning tomorrow @ 10:00 AM\nVideo call link attached.",
        "From: Sarah Jenkins <s.jenkins@acmecorp.internal>\nSubject: Q3 roadmap presentation draft\nAttached the slide deck for review before Thursday.",
        "From: Stripe <receipts@stripe.com>\nSubject: Your receipt from AWS Cloud Services for invoice #48102\nAmount paid $120.40.",
        "From: Jira Software <jira@company.atlassian.net>\nSubject: VON-142 assigned to you: Investigate memory pressure on worker pod.",
    ]

    records = []
    for _ in range(n // 2):
        records.append({
            "state": random.choice(phish_pool),
            "question": instruction,
            "options": options,
            "label": "yes",
            "source": "phish_pos",
        })
        records.append({
            "state": random.choice(safe_pool),
            "question": instruction,
            "options": options,
            "label": "no",
            "source": "phish_neg",
        })
    return records


def generate_commit_intent_cases(n: int = 5000) -> List[dict]:
    instruction = "What is the primary intent of this Git commit message?"
    criteria = {
        "feat": "Introducing a new feature, user-facing capability, or API endpoint",
        "fix": "Bug fixes, resolving errors, crashes, regressions, or unexpected behavior",
        "docs": "Documentation updates, README edits, comments, or technical guides",
        "refactor": "Restructuring existing code without changing external runtime behavior",
        "test": "Adding or modifying unit tests, integration benchmarks, or mock fixtures",
        "chore": "Build scripts, dependency upgrades, release bumps, or CI workflows",
    }
    options = [{"id": k, "description": v} for k, v in criteria.items()]

    commits = {
        "feat": [
            "feat(auth): add OAuth2 PKCE login support for single-page applications",
            "feat(api): introduce POST /v1/decide batch inference endpoint",
            "feat(tui): implement dark mode color theme for terminal UI",
            "feat(export): allow exporting audit logs directly to S3 parquet format",
        ],
        "fix": [
            "fix(engine): resolve null pointer exception when state is empty dictionary",
            "fix(db): prevent deadlocks during concurrent write operations in transaction table",
            "fix(ui): correct misaligned dropdown menu in mobile landscape view",
            "fix(cli): handle SIGINT gracefully to flush logs before exit",
        ],
        "docs": [
            "docs: update README with Option-Marker architecture diagrams and benchmark plots",
            "docs(api): document rate limiting headers and retry backoff recommendations",
            "docs: add contributor guidelines and local development setup instructions",
            "docs: fix typo in quickstart example code snippet",
        ],
        "refactor": [
            "refactor(core): extract tensor packing logic into dedicated helper class",
            "refactor(parser): simplify AST traversal using visitor pattern",
            "refactor: replace legacy threading locks with asyncio concurrency primitives",
            "refactor: clean up deprecated parameter names across public API surface",
        ],
        "test": [
            "test(backends): add end-to-end regression tests for 49 benchmark tasks",
            "test: mock external Redis connection to speed up test suite execution",
            "test(fanout): verify speculative execution with 10 concurrent questions",
            "test: achieve 95% code coverage across option marker scoring module",
        ],
        "chore": [
            "chore: bump pydantic from 2.8.0 to 2.9.2 in pyproject.toml",
            "chore(ci): upgrade GitHub Actions runner from ubuntu-20.04 to ubuntu-22.04",
            "chore: update .gitignore to exclude temporary evaluation scratch directories",
            "chore(release): bump version to 1.2.0 and generate changelog",
        ],
    }

    records = []
    for _ in range(n // 6):
        for intent, pool in commits.items():
            records.append({
                "state": random.choice(pool),
                "question": instruction,
                "options": options,
                "label": intent,
                "source": f"commit_{intent}",
            })
    return records


# =====================================================================
# 3. Content & Linguistics Cluster (Grammar, Formality, Reading Level)
# =====================================================================

def generate_grammar_cases(n: int = 6000) -> List[dict]:
    instruction = "Which grammatical issue does this sentence contain?"
    criteria = {
        "subject_verb_agreement": "Mismatch in number (singular vs plural) between subject and main verb",
        "comma_splice": "Joining two independent clauses with only a comma without coordinating conjunction",
        "dangling_modifier": "A descriptive phrase modifying the wrong subject or lacking clear subject",
        "fragment": "Incomplete sentence missing a subject or main verb clause",
        "run_on": "Two independent clauses merged without appropriate punctuation or conjunction",
        "none": "The sentence is grammatically correct and well-formed",
    }
    options = [{"id": k, "description": v} for k, v in criteria.items()]

    examples = {
        "subject_verb_agreement": [
            "The cluster of database servers are experiencing severe latency.",
            "Each of the software engineers have submitted their annual review.",
            "Neither the manager nor her team members was informed of the migration.",
            "A comprehensive list of user permissions were deleted by the script.",
        ],
        "comma_splice": [
            "The test suite finished running, all unit tests passed with zero errors.",
            "We deployed the hotfix at midnight, the customer reported no further issues.",
            "The API returned a 500 status code, we immediately escalated to the on-call engineer.",
            "I checked the server logs, there was no sign of suspicious network activity.",
        ],
        "dangling_modifier": [
            "Walking into the data center, the cold draft of server fans was immediately felt.",
            "Having reviewed the pull request, several critical security flaws were uncovered.",
            "After debugging the memory leak for three hours, the coffee cup was empty.",
            "To deploy the application to production, credentials must be configured first.",
        ],
        "fragment": [
            "Because the network gateway timed out before the transaction could be committed.",
            "Which is why our security team mandates hardware security keys for all staff.",
            "Although the database backup completed without any reported errors last night.",
            "Especially during high-traffic promotional events with thousands of concurrent shoppers.",
        ],
        "run_on": [
            "The deployment failed we had to roll back to the previous version immediately.",
            "The customer requested a full refund they had not used any of their monthly quota.",
            "We upgraded the memory capacity the server still crashed under peak load.",
            "The documentation is clear anyone can follow the setup instructions easily.",
        ],
        "none": [
            "The engineering team deployed the new release after all tests passed successfully.",
            "Because the server memory was exhausted, the application crashed unexpectedly.",
            "She reviewed the contract carefully before signing the three-year service agreement.",
            "When the on-call engineer received the alert, she immediately investigated the root cause.",
        ],
    }

    records = []
    for _ in range(n // 6):
        for err_type, pool in examples.items():
            records.append({
                "state": random.choice(pool),
                "question": instruction,
                "options": options,
                "label": err_type,
                "source": f"grammar_{err_type}",
            })
    return records


def generate_formality_cases(n: int = 5000) -> List[dict]:
    instruction = "What is the register and formality level of this communication?"
    criteria = [
        {"id": "0", "description": "Slang or internet colloquial: informal slang, abbreviations, lowercase, emotional"},
        {"id": "1", "description": "Casual workplace: friendly, conversational, standard colloquial expressions"},
        {"id": "2", "description": "Standard professional: clear, polite, structured business communication"},
        {"id": "3", "description": "Formal or diplomatic: ceremonial, legalistic, elevated academic or executive tone"},
    ]

    pools = {
        "0": [
            "yo tbh that new feature is kinda trash ngl... pls fix asap bro",
            "lmao the whole site just died frfr 💀",
            "idk why this button is broken again smh",
            "gonna bounce early today cya tomorrow folks",
        ],
        "1": [
            "Hey team, quick heads up that I'll be 10 minutes late to the daily standup!",
            "Thanks for the fast turnaround on that PR! Looks great to me.",
            "Can we grab a quick coffee chat sometime this afternoon to chat about the roadmap?",
            "Just wanted to check in on the status of that bug report when you have a sec.",
        ],
        "2": [
            "Dear Customer Support Team, I am writing to request a copy of our annual billing invoice.",
            "Please find attached the quarterly project review for your inspection and feedback.",
            "We have completed the requested database migration and verified all operational metrics.",
            "Could you please clarify the scheduled timeline for the enterprise SSO rollout?",
        ],
        "3": [
            "Pursuant to Section 4.2 of the Master Services Agreement, Notice of Termination is hereby given.",
            "The Corporation extends its highest regards and respectfully solicits your prompt consideration.",
            "All covenants and obligations herein stipulated shall remain binding upon the parties.",
            "In witness whereof, the authorized signatories have executed this bilateral accord.",
        ],
    }

    records = []
    for _ in range(n // 4):
        for lvl, pool in pools.items():
            records.append({
                "state": random.choice(pool),
                "question": instruction,
                "options": criteria,
                "label": lvl,
                "source": f"formality_{lvl}",
            })
    return records


def generate_reading_level_cases(n: int = 5000) -> List[dict]:
    instruction = "Assess the reading difficulty level of this passage."
    criteria = [
        {"id": "0", "description": "Elementary (Grade 1-3): simple sentences, common daily vocabulary, concrete concepts"},
        {"id": "1", "description": "Middle School (Grade 4-8): compound sentences, descriptive vocabulary, straightforward ideas"},
        {"id": "2", "description": "High School (Grade 9-12): complex syntax, abstract reasoning, technical terms"},
        {"id": "3", "description": "Advanced / Scholarly: dense academic prose, domain jargon, highly specialized syntax"},
    ]

    pools = {
        "0": [
            "The dog ran across the green grass. He wanted to catch the red ball.",
            "The sun was warm and bright. We ate sweet apples by the big blue lake.",
            "She put on her yellow coat. It was raining outside our school.",
        ],
        "1": [
            "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from water.",
            "During the Middle Ages, castles were built primarily as fortified defensive strongholds against invaders.",
            "Volcanoes erupt when magma rises to the surface, causing pressure to build inside the earth's crust.",
        ],
        "2": [
            "The Industrial Revolution catalyzed unprecedented socioeconomic stratification across urban working-class populations.",
            "Quantum entanglement demonstrates that physical measurements on paired particles exhibit instantaneous correlation.",
            "Constitutional jurisprudence balances individual civil liberties against the legitimate exercise of state authority.",
        ],
        "3": [
            "Epistemological foundationalism posits that epistemic justification terminates in ungrounded basic doxastic commitments.",
            "Phenomenological hermeneutics interrogates the dialectical ontological presuppositions governing Dasein's historical situatedness.",
            "Non-equilibrium thermodynamic bifurcation manifests asymptotic dissipation within non-linear reaction-diffusion manifolds.",
        ],
    }

    records = []
    for _ in range(n // 4):
        for lvl, pool in pools.items():
            records.append({
                "state": random.choice(pool),
                "question": instruction,
                "options": criteria,
                "label": lvl,
                "source": f"reading_level_{lvl}",
            })
    return records


# =====================================================================
# 4. Operations & Specialized Triage
# =====================================================================

def generate_veterinary_triage_cases(n: int = 4000) -> List[dict]:
    instruction = "Assess the urgency level for this veterinary patient report."
    criteria = [
        {"id": "0", "description": "Non-urgent / Routine: preventative care, vaccination booster, general inquiry"},
        {"id": "1", "description": "Mild: minor itching, mild limping, ear discharge with no signs of distress"},
        {"id": "2", "description": "Moderate: persistent vomiting, refusal to eat for 24h, deep cut, urinary strain"},
        {"id": "3", "description": "Severe / Critical: difficulty breathing, unconsciousness, severe poisoning, arterial bleed"},
    ]

    pools = {
        "0": [
            "Calling to schedule annual rabies booster shots and general wellness exam for our 3-year-old beagle.",
            "Need to request a refill on routine flea and heartworm preventative chewables.",
            "Our indoor cat needs her claws trimmed and ears cleaned sometime next week.",
        ],
        "1": [
            "My golden retriever has been scratching his left ear and shaking his head since yesterday.",
            "Dog is slightly limping on front paw after chasing a tennis ball, but still eating and alert.",
            "Cat has a small scab on her chin with mild hair loss; seems unbothered.",
        ],
        "2": [
            "Male cat has been repeatedly straining in the litter box for 12 hours with no urine produced.",
            "Puppy has vomited four times today and is refusing all water and food.",
            "Deep puncture wound on dog's leg from a fence post, bleeding steadily.",
        ],
        "3": [
            "Dog ingested an entire bag of dark chocolate baker's cocoa and is having violent seizures.",
            "Cat was hit by a car, completely unresponsive with shallow, labored breathing and blue gums.",
            "Large dog has severe abdominal distension, dry heaving, pale mucous membranes, and cannot stand (GDV/bloat).",
        ],
    }

    records = []
    for _ in range(n // 4):
        for lvl, pool in pools.items():
            records.append({
                "state": random.choice(pool),
                "question": instruction,
                "options": criteria,
                "label": lvl,
                "source": f"vet_triage_{lvl}",
            })
    return records


def generate_expense_cases(n: int = 4000) -> List[dict]:
    instruction = "Which expense category does this business expense belong to?"
    criteria = {
        "meals": "Food or drink for the employee or their team while traveling or working",
        "travel": "Flights, trains, taxis, lodging, car rentals, tolls, and transportation fees",
        "office_supplies": "Physical hardware, equipment, monitors, chairs, and office stationery",
        "software": "SaaS subscriptions, digital licenses, cloud services, and developer tools",
        "client_entertainment": "Dinners, tickets, gifts, or hospitality provided to clients or prospective hires",
    }
    options = [{"id": k, "description": v} for k, v in criteria.items()]

    expenses = {
        "meals": [
            "Team dinner for 4 engineers working late on the launch offsite, $95.",
            "Lunch at airport terminal during 4-hour layover, $22.50.",
            "Breakfast coffee and sandwich while traveling on business trip, $14.00.",
        ],
        "travel": [
            "United Airlines round-trip economy flight to New York for client summit, $450.",
            "Lyft rideshare from hotel to conference convention center, $32.40.",
            "Hotel stay at Hyatt Regency for 2 nights during industry expo, $380.",
        ],
        "office_supplies": [
            "Ergonomic mesh desk chair and adjustable monitor stand for home workstation, $310.",
            "Pack of whiteboard markers, notebook pads, and printer toner, $65.",
            "Dell 27-inch 4K external monitor for remote developer, $380.",
        ],
        "software": [
            "Annual GitHub Enterprise seat renewal for engineering organization, $1,200.",
            "Monthly Figma professional subscription for product design pod, $75.",
            "Datadog cloud infrastructure monitoring monthly subscription, $420.",
        ],
        "client_entertainment": [
            "Steakhouse dinner with Acme leadership team to celebrate multi-year contract renewal, $420.",
            "Two courtside basketball tickets provided to prospective enterprise client, $350.",
            "Wine and gift basket sent to client executive team for holiday appreciation, $125.",
        ],
    }

    records = []
    for _ in range(n // 5):
        for cat, pool in expenses.items():
            records.append({
                "state": random.choice(pool),
                "question": instruction,
                "options": options,
                "label": cat,
                "source": f"expense_{cat}",
            })
    return records


# =====================================================================
# Main Corpus Assembler (Phase 4 Universal Decision Corpus)
# =====================================================================

def build_universal_corpus(
    output_dir: str = "data_universal",
    max_train: int = 200000,
    val_samples: int = 5000,
    seed: int = 42,
):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("=============================================================")
    print("Building Phase 4 Universal Decision Corpus (200k samples)")
    print("=============================================================")

    # Import existing Phase 2/3 generators
    from .prepare_decision_dataset import (
        generate_refund_policy_cases,
        generate_secret_leak_cases,
        generate_operational_urgency_cases,
        generate_frustration_score_cases,
        generate_incident_severity_cases,
        generate_department_triage_cases,
        prepare_banking_choice,
        prepare_emotion_sentiment,
        prepare_adversarial_core,
    )

    all_records = []

    # 1. Compliance & Regulatory
    print("Synthesizing Compliance & Regulatory cluster...")
    all_records.extend(generate_hazmat_cases(8000))
    all_records.extend(generate_dietary_vegan_cases(8000))
    all_records.extend(generate_fair_housing_cases(8000))
    all_records.extend(generate_travel_policy_cases(8000))
    all_records.extend(generate_refund_policy_cases(10000))

    # 2. Security & DevOps
    print("Synthesizing Security & DevOps cluster...")
    all_records.extend(generate_sql_injection_cases(8000))
    all_records.extend(generate_phishing_cases(8000))
    all_records.extend(generate_secret_leak_cases(8000))
    all_records.extend(generate_commit_intent_cases(8000))

    # 3. Content & Linguistics
    print("Synthesizing Content & Linguistics cluster...")
    all_records.extend(generate_grammar_cases(12000))
    all_records.extend(generate_formality_cases(10000))
    all_records.extend(generate_reading_level_cases(10000))

    # 4. Operations & Triage
    print("Synthesizing Operations & Triage cluster...")
    all_records.extend(generate_veterinary_triage_cases(8000))
    all_records.extend(generate_expense_cases(8000))
    all_records.extend(generate_department_triage_cases(10000))
    all_records.extend(generate_operational_urgency_cases(8000))
    all_records.extend(generate_frustration_score_cases(8000))
    all_records.extend(generate_incident_severity_cases(8000))

    # 5. Open Curated Intent & Sentiment Datasets
    print("Loading curated intent datasets...")
    all_records.extend(prepare_banking_choice(15000))
    all_records.extend(prepare_emotion_sentiment(12000))

    # 6. Reasoning Core (ANLI & WANLI)
    print("Loading Adversarial Reasoning Core (ANLI + WANLI)...")
    all_records.extend(prepare_adversarial_core(40000))

    random.shuffle(all_records)
    print(f"\nTotal collected Universal records: {len(all_records):,}")

    val_records = all_records[:val_samples]
    train_records = all_records[val_samples : val_samples + max_train]

    print(f"Final Train Set: {len(train_records):,} records")
    print(f"Final Val Set:   {len(val_records):,} records")

    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for rec in train_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for rec in val_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved {len(train_records):,} train rows to {train_path}")
    print(f"Saved {len(val_records):,} val rows to {val_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data_universal")
    parser.add_argument("--max_train", type=int, default=200000)
    parser.add_argument("--val_samples", type=int, default=5000)
    args = parser.parse_args()

    build_universal_corpus(
        output_dir=args.output_dir,
        max_train=args.max_train,
        val_samples=args.val_samples,
    )
