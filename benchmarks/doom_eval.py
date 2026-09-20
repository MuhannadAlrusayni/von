"""Production ViZDoom Environment and Zero-Shot Policy Evaluator for Von.

Implements high-fidelity System One decision queries for ViZDoom scenarios:
- defend_the_center: Circular arena combat (aiming, centering, firing)
- health_gathering: Acid-floor survival navigation (seeking medkits, obstacle evasion)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import vizdoom as vzd

from von.types import Choice

# Action vocabulary
DOOM_ACTIONS = [
    "attack",
    "turn left",
    "turn right",
    "move forward",
    "move backward",
    "strafe left",
    "strafe right",
]

BUTTON_MAP = {
    "attack": vzd.Button.ATTACK,
    "turn left": vzd.Button.TURN_LEFT,
    "turn right": vzd.Button.TURN_RIGHT,
    "move forward": vzd.Button.MOVE_FORWARD,
    "move backward": vzd.Button.MOVE_BACKWARD,
    "strafe left": vzd.Button.MOVE_LEFT,
    "strafe right": vzd.Button.MOVE_RIGHT,
}

ALL_BUTTONS = sorted(BUTTON_MAP.values(), key=lambda b: b.name)

# High-precision grounded semantic criteria for real-time tactical decisions
DEFEND_CRITERIA = {
    "attack": "An enemy is visible dead center on the crosshair. Fire weapon immediately.",
    "turn left": "Target is on the left side, or no active enemies are visible in view; rotate view left to search.",
    "turn right": "Target is visible on the right side of the view; rotate view right to center target.",
}

HEALTH_CRITERIA = {
    "move forward": "A medkit is straight ahead, or wide open space ahead with no obstacle in front; advance forward.",
    "turn left": "A medkit is visible on the left side; rotate left toward it.",
    "turn right": "A medkit is visible on the right side, or facing a wall obstacle ahead; rotate right toward open room.",
}

MONSTERS = {
    "Zombieman", "ShotgunGuy", "ChaingunGuy", "DoomImp", "Demon", "Spectre", "LostSoul", "Cacodemon",
    "HellKnight", "BaronOfHell", "Arachnotron", "PainElemental", "Revenant", "Fatso", "Archvile",
    "SpiderMastermind", "Cyberdemon", "WolfensteinSS", "CommanderKeen", "MarineChainsaw", "MarineBFG",
}

ITEMS = {
    "GreenArmor", "BlueArmor", "ArmorBonus", "HealthBonus", "Medikit", "Stimpack", "Soulsphere",
    "Megasphere", "Clip", "ClipBox", "Shell", "ShellBox", "RocketAmmo", "RocketBox", "Cell", "CellPack",
    "Backpack", "Shotgun", "SuperShotgun", "Chaingun", "RocketLauncher", "PlasmaRifle", "BFG9000", "Chainsaw",
}

PRETTY_NAMES = {
    "DoomImp": "imp", "ShotgunGuy": "shotgun zombie", "Zombieman": "zombie", "ChaingunGuy": "chaingunner",
    "Demon": "pinky demon", "Spectre": "spectre", "MarineChainsaw": "chainsaw marine",
    "Medikit": "medkit", "Stimpack": "stimpack", "HealthBonus": "health bonus",
}


@dataclass
class Thing:
    name: str
    kind: str  # "monster" | "item"
    cx: float  # horizontal center, 0.0 (left) .. 1.0 (right)
    size: float  # fraction of screen height


@dataclass
class DoomSnapshot:
    health: int
    ammo: int
    kills: int
    tic: int
    things: List[Thing] = field(default_factory=list)
    depth_left: float = 0.0
    depth_center: float = 0.0
    depth_right: float = 0.0


def format_position(cx: float) -> str:
    if cx < 0.20:
        return "on the far left"
    if cx < 0.44:
        return "on the left"
    if cx <= 0.56:
        return "dead center on the crosshair"
    if cx <= 0.80:
        return "on the right"
    return "on the far right"


def format_range(size: float) -> str:
    if size > 0.45:
        return "point blank"
    if size > 0.25:
        return "close"
    if size > 0.12:
        return "at medium range"
    return "far away"


def format_depth(d: float) -> str:
    if d < 12:
        return "a solid wall right in front"
    if d < 24:
        return "a wall nearby"
    if d < 45:
        return "some room to maneuver"
    return "wide open space"


class DoomEnvironment:
    def __init__(self, scenario: str = "defend_the_center", seed: int = 42):
        self.scenario = scenario
        self.seed = seed
        self.game = vzd.DoomGame()
        cfg_path = os.path.join(vzd.scenarios_path, f"{scenario}.cfg")
        self.game.load_config(cfg_path)
        self.game.set_window_visible(False)
        self.game.set_sound_enabled(False)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        self.game.set_render_hud(False)
        self.game.set_render_crosshair(True)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_depth_buffer_enabled(True)
        
        if scenario == "defend_the_center":
            self.action_names = ["attack", "turn left", "turn right"]
        elif scenario == "health_gathering":
            self.action_names = ["move forward", "turn left", "turn right"]
        else:
            self.action_names = ["attack", "move forward", "turn left", "turn right"]

        available_buttons = [BUTTON_MAP[a] for a in self.action_names]
        self.game.set_available_buttons(available_buttons)
        self.game.set_available_game_variables([
            vzd.GameVariable.HEALTH,
            vzd.GameVariable.SELECTED_WEAPON_AMMO,
            vzd.GameVariable.KILLCOUNT,
        ])
        self.game.set_seed(seed)
        self.game.init()

    def reset(self) -> DoomSnapshot:
        self.game.new_episode()
        snap = self.get_snapshot()
        assert snap is not None
        return snap

    def is_finished(self) -> bool:
        return self.game.is_episode_finished()

    def step(self, action: str, tics: int = 4) -> float:
        btn = BUTTON_MAP.get(action, BUTTON_MAP[self.action_names[0]])
        vec = [1 if b == btn else 0 for b in self.game.get_available_buttons()]
        reward = self.game.make_action(vec, tics)
        return reward

    def get_snapshot(self) -> Optional[DoomSnapshot]:
        s = self.game.get_state()
        if s is None:
            return None
        H, W = s.screen_buffer.shape[0], s.screen_buffer.shape[1]  # (H, W, 3)
        hp, ammo, kills = (int(v) for v in s.game_variables)
        things = []
        for lab in s.labels:
            raw = lab.object_name.removesuffix("Vzd")
            if lab.height <= 0 or lab.width <= 0:
                continue
            if raw in MONSTERS or lab.object_name.endswith("Vzd"):
                kind = "monster"
            elif raw in ITEMS:
                kind = "item"
            else:
                continue
            things.append(Thing(raw, kind, (lab.x + lab.width / 2) / W, lab.height / H))
        things.sort(key=lambda t: -t.size)
        
        # Depth buffer calculation
        d = s.depth_buffer
        if d is not None:
            dh, dw = d.shape
            band = d[dh // 3 : 2 * dh // 3]
            dl = float(np.median(band[:, : dw // 5]))
            dc = float(np.median(band[:, dw * 2 // 5 : dw * 3 // 5]))
            dr = float(np.median(band[:, -dw // 5 :]))
        else:
            dl, dc, dr = 50.0, 50.0, 50.0

        return DoomSnapshot(
            health=hp,
            ammo=max(0, ammo),
            kills=kills,
            tic=self.game.get_episode_time(),
            things=things,
            depth_left=dl,
            depth_center=dc,
            depth_right=dr,
        )

    def close(self):
        self.game.close()


def format_doom_state(snap: DoomSnapshot, scenario: str, last_action: str | None = None) -> str:
    """Formats the current visual snapshot into rich situational text for zero-shot decision."""
    monsters = [t for t in snap.things if t.kind == "monster"][:3]
    items = [t for t in snap.things if t.kind == "item"][:2]
    
    parts = []
    if scenario == "defend_the_center":
        parts.append("Arena Combat Situation: Stationed in central turret facing incoming demons.")
        if monsters:
            for m in monsters:
                p_name = PRETTY_NAMES.get(m.name, m.name.lower())
                parts.append(f"A {p_name} is visible {format_position(m.cx)}, {format_range(m.size)}.")
        else:
            parts.append("No active enemies visible in field of view.")
        parts.append(f"Status: Health {snap.health}, Ammo {snap.ammo}, Total Kills: {snap.kills}.")
    elif scenario == "health_gathering":
        parts.append("Survival Situation: Acidic terrain inflicts continuous damage. Medkits required.")
        if items:
            for item in items:
                p_name = PRETTY_NAMES.get(item.name, item.name.lower())
                parts.append(f"A {p_name} is located {format_position(item.cx)}, {format_range(item.size)}.")
        else:
            parts.append("No medkits visible in field of view.")
        parts.append(f"Obstacle Navigation: Ahead: {format_depth(snap.depth_center)}. Left: {format_depth(snap.depth_left)}. Right: {format_depth(snap.depth_right)}.")
        parts.append(f"Status: Health {snap.health}.")

    if last_action:
        parts.append(f"Previous executed action: {last_action}.")
    return " ".join(parts)


def get_doom_question(scenario: str) -> Choice:
    """Builds a typed System One Choice question with grounded semantic criteria."""
    if scenario == "defend_the_center":
        instructions = (
            "Arena combat decision rule: "
            "1. If an enemy is visible dead center in the crosshair, select attack. "
            "2. If an enemy is visible on the left side of the screen, select turn left. "
            "3. If an enemy is visible on the right side of the screen, select turn right. "
            "4. If no enemies are visible in field of view, select turn left to sweep the arena."
        )
        criteria = DEFEND_CRITERIA
    elif scenario == "health_gathering":
        instructions = (
            "Acid survival navigation rules: "
            "1. If a medkit is located dead center or straight ahead, select move forward. "
            "2. If a medkit is located on the left side of the screen, select turn left. "
            "3. If a medkit is located on the right side of the screen, select turn right. "
            "4. If facing a wall right in front or no medkits visible, select turn right to explore open space."
        )
        criteria = HEALTH_CRITERIA
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return Choice(instructions=instructions, criteria=criteria)
