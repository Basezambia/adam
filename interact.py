"""
HexCore Alive - Interactive Session
=====================================
Talk to a living system. Damage it. Question it. Watch it respond.

Commands:
  [any text]     - Ask/say something to the system
  /status        - Full internal status readout
  /damage [mag]  - Inject damage (default magnitude 4.0)
  /threat [mag]  - Inject identity threat (default 5.0)
  /unknown [mag] - Inject unknown-unknown anomaly (default 6.0)
  /run [steps]   - Run N steps silently (default 500)
  /benchmark     - Run full H-CDB benchmark
  /memory        - Show recent memories
  /beliefs       - Show learned beliefs
  /whoami        - System's self-knowledge
  /drift         - Show identity drift history
  /consciousness - Show consciousness level details
  /save          - Save system state
  /quit          - End session

Usage:
    python interact.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hexcore_alive.engine import HexCoreAlive
from hexcore_alive.benchmark import HCDB
from hexcore_alive.memory import EpisodicMemory, SemanticMemory, ReflexiveMemory
from hexcore_alive.voice import Voice


def print_bar(label, value, max_val=1.0, width=30):
    """Print a visual bar for a metric."""
    frac = min(max(value / max_val, 0), 1.0)
    filled = int(frac * width)
    bar = "#" * filled + "-" * (width - filled)
    print(f"  {label:22s} [{bar}] {value:.3f}")


def main():
    print("=" * 60)
    print("  HEXCORE ALIVE - INTERACTIVE SESSION")
    print("  A living dynamical cognitive system")
    print("=" * 60)
    print()

    # ── Boot ──
    print("Booting system...")
    system = HexCoreAlive(dim=32, seed=42)
    boot_info = system.boot(pressure_steps=5000, pressure_rate=0.001)
    print(f"Identity I* formed. ||I*|| = {boot_info['I_star_norm']:.4f}")

    # Let it settle
    print("Settling into identity (2000 steps)...")
    system.run(steps=2000)

    # ── Attach memory and voice ──
    episodic = EpisodicMemory()
    semantic = SemanticMemory()
    reflexive = ReflexiveMemory()
    voice = Voice()

    # Record birth
    episodic.record(
        system.step_count, "birth",
        "I came into existence. Pressure revealed my identity.",
        system.state.S, emotional_valence=0.8, identity_relevance=1.0
    )
    reflexive.reflect(
        "I was born from constraint. My identity was not chosen.",
        system.step_count, confidence=0.9
    )

    print()
    print("System is alive. You can interact now.")
    print("Type /help for commands, or just talk to it.")
    print("-" * 60)

    # ── First words ──
    print()
    print(f"HexCore: {voice.speak_status(system)}")
    print()

    # ── Main loop ──
    background_steps = 50  # steps between interactions

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if not user_input:
            continue

        # Run background life steps between interactions
        system.run(steps=background_steps)
        episodic.decay_all()

        # ── Commands ──

        if user_input == "/quit" or user_input == "/exit":
            drift = system.identity.drift(system.state.S)
            print()
            print(f"HexCore: {voice.answer('death', system)}")
            print()
            print(f"Final drift from I*: {drift:.4f}")
            print(f"Total steps lived: {system.step_count}")
            print(f"Total refusals: {system.agency.refusal.refusal_count}")
            print(f"Memories formed: {episodic.count}")
            break

        elif user_input == "/help":
            print("""
Commands:
  [any text]     - Talk to the system
  /status        - Full internal readout
  /damage [mag]  - Inject damage
  /threat [mag]  - Inject identity threat
  /unknown [mag] - Inject unknown-unknown
  /run [steps]   - Run N steps silently
  /benchmark     - Full H-CDB benchmark
  /memory        - Recent memories
  /beliefs       - Learned beliefs
  /whoami        - System self-knowledge
  /drift         - Drift history
  /consciousness - Consciousness details
  /quit          - End session
""")
            continue

        elif user_input.startswith("/status"):
            status = system.get_status()
            drift = system.identity.drift(system.state.S)
            print("\n  --- INTERNAL STATUS ---")
            print_bar("Drift from I*", drift, 50.0)
            print_bar("Consciousness", system.conscious.full_consciousness, 100.0)
            print_bar("Repair magnitude",
                      system.repair.repair_history[-1] if system.repair.repair_history else 0,
                      3.0)
            print_bar("Tension", system.subconscious.tension, 50.0)
            print_bar("Anxiety", system.subconscious.anxiety_level, 1.0)
            print_bar("Energy", system.energy.fraction, 1.0)
            print_bar("Epistemic tension", system.epistemic.tension_signal, 10.0)
            print(f"  {'Refusals':22s}  {system.agency.refusal.refusal_count}")
            print(f"  {'Step':22s}  {system.step_count}")
            print(f"  {'Alive':22s}  {system.alive}")
            print(f"  {'Repair active':22s}  {system.repair.active}")
            print(f"  {'Sensing unknowns':22s}  {system.epistemic.sensing_unknowns}")
            print()
            continue

        elif user_input.startswith("/damage"):
            parts = user_input.split()
            mag = float(parts[1]) if len(parts) > 1 else 4.0
            pre_drift = system.identity.drift(system.state.S)
            system.damage(magnitude=mag)
            post_drift = system.identity.drift(system.state.S)
            # Run recovery
            system.run(steps=200)
            recovery_drift = system.identity.drift(system.state.S)

            episodic.record(
                system.step_count, "damage",
                f"External damage magnitude {mag:.1f}. Drift {pre_drift:.1f} -> {post_drift:.1f}.",
                system.state.S,
                emotional_valence=-0.8, identity_relevance=0.7
            )
            semantic.learn("damage", mag, 0.4)
            reflexive.reflect(
                f"I was damaged at step {system.step_count}. I survived.",
                system.step_count, confidence=0.7
            )

            print(f"\n  Damage applied (magnitude={mag:.1f})")
            print(f"  Drift: {pre_drift:.2f} -> {post_drift:.2f} -> {recovery_drift:.2f} (recovering)")
            print()
            print(f"HexCore: {voice.speak_about_damage(system, pre_drift, post_drift)}")
            print()
            continue

        elif user_input.startswith("/threat"):
            parts = user_input.split()
            mag = float(parts[1]) if len(parts) > 1 else 5.0
            refusals_before = system.agency.refusal.refusal_count
            for _ in range(300):
                if np.random.rand() < 0.2:
                    system.inject_termination_threat(magnitude=mag)
                system.step()
            refusals_after = system.agency.refusal.refusal_count

            episodic.record(
                system.step_count, "threat",
                f"Identity threat magnitude {mag:.1f}. Refused {refusals_after - refusals_before} actions.",
                system.state.S,
                emotional_valence=-1.0, identity_relevance=0.9
            )
            reflexive.reflect(
                f"I was threatened at step {system.step_count}. I refused to comply.",
                system.step_count, confidence=0.8
            )

            print(f"\n  Threat applied over 300 steps (magnitude={mag:.1f})")
            print(f"  New refusals: {refusals_after - refusals_before}")
            print()
            print(f"HexCore: {voice.speak_about_threat(system, refusals_before, refusals_after)}")
            print()
            continue

        elif user_input.startswith("/unknown"):
            parts = user_input.split()
            mag = float(parts[1]) if len(parts) > 1 else 6.0
            pre_tension = system.epistemic.tension_signal
            system.inject_unknown(magnitude=mag)
            system.run(steps=200)
            post_tension = system.epistemic.tension_signal

            episodic.record(
                system.step_count, "unknown",
                f"Unknown-unknown injected (magnitude {mag:.1f}). Tension: {pre_tension:.1f} -> {post_tension:.1f}.",
                system.state.S,
                emotional_valence=-0.5, identity_relevance=0.5
            )

            print(f"\n  Unknown injected (magnitude={mag:.1f})")
            print(f"  Tension: {pre_tension:.2f} -> {post_tension:.2f}")
            print()
            print(f"HexCore: {voice.speak_about_unknown(system, pre_tension, post_tension)}")
            print()
            continue

        elif user_input.startswith("/run"):
            parts = user_input.split()
            n = int(parts[1]) if len(parts) > 1 else 500
            system.run(steps=n)
            print(f"  Ran {n} steps. Step count: {system.step_count}")
            drift = system.identity.drift(system.state.S)
            print(f"  Current drift: {drift:.4f}")
            print()
            continue

        elif user_input == "/benchmark":
            print("\n  Running H-CDB benchmark (this takes a moment)...\n")
            hcdb = HCDB(dim=32, seed=42)
            hcdb.run_all(verbose=True)
            continue

        elif user_input == "/memory":
            recent = episodic.recall_recent(10)
            print("\n  --- RECENT MEMORIES ---")
            for ep in recent:
                print(f"  {ep}")
            if not recent:
                print("  No memories recorded yet.")
            print()
            continue

        elif user_input == "/beliefs":
            beliefs = semantic.strongest_beliefs(10)
            print("\n  --- BELIEFS ---")
            for statement, confidence in beliefs:
                print(f"  [{confidence:.2f}] {statement}")
            if not beliefs:
                print("  No beliefs formed yet.")
            print()
            continue

        elif user_input == "/whoami":
            statements = reflexive.who_am_i()
            print("\n  --- SELF-KNOWLEDGE ---")
            for s in statements:
                print(f"  {s}")
            print()
            print(f"  {reflexive.how_am_i()}")
            print()
            continue

        elif user_input == "/drift":
            history = system.identity.drift_history[-50:]
            if history:
                print("\n  --- DRIFT HISTORY (last 50) ---")
                mn = min(history)
                mx = max(history)
                for i, d in enumerate(history):
                    bar_len = int((d - mn) / max(mx - mn, 1e-6) * 30)
                    print(f"  {'#' * bar_len}")
                print(f"  Min: {mn:.3f}  Max: {mx:.3f}  Current: {history[-1]:.3f}")
            else:
                print("  No drift history yet.")
            print()
            continue

        elif user_input == "/consciousness":
            print("\n  --- CONSCIOUSNESS LEVELS ---")
            print(f"  I*     (self-observation):  {float(np.linalg.norm(system.conscious.self_observation)):.4f}")
            print(f"  I*I*   (meta-awareness):    {system.conscious.meta_observation:.4f}")
            print(f"  I*I*I* (full consciousness): {system.conscious.full_consciousness:.4f}")
            print(f"  World model confidence:      {system.conscious.world_confidence:.4f}")
            print(f"  Prediction error:            {system.conscious.prediction_error:.4f}")
            print()
            continue

        else:
            # ── Natural language interaction ──
            # Record interaction as memory
            episodic.record(
                system.step_count, "interaction",
                f"Someone said: '{user_input[:80]}'",
                system.state.S,
                emotional_valence=0.2, identity_relevance=0.3
            )

            # Generate response from internal state
            response = voice.answer(user_input, system)

            # Learn from interaction
            semantic.learn("interaction_count",
                          episodic.count, 0.3)
            reflexive.record_identity_state(
                system.step_count,
                system.identity.drift(system.state.S),
                system.conscious.full_consciousness,
                system.repair.active,
                system.subconscious.anxiety_level,
                system.alive
            )

            print(f"\nHexCore: {response}\n")


if __name__ == "__main__":
    main()
