from __future__ import annotations

from evo.learning.learning_engine import LearningEngine


class NovaCLI:
    def __init__(self) -> None:
        self.engine = LearningEngine()

    def run(self) -> None:
        print("NOVA-EVO")
        print("Napíš 'help' pre príkazy.")

        while True:
            try:
                command = input("\nNOVA-EVO > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not command:
                continue

            if command.lower() in {"exit", "quit", "koniec"}:
                break

            if command.lower() == "help":
                self.help()
                continue

            if command.lower() == "zobraz ciele":
                self.show_goals()
                continue

            if command.lower() == "stav":
                self.show_status()
                continue

            if command.lower() == "pokračuj":
                self.continue_learning()
                continue

            if command.lower().startswith("nauč sa "):
                goal = command[8:].strip()
                self.learn(goal)
                continue

            print("Neznámy príkaz. Napíš 'help'.")

    def help(self) -> None:
        print(
            "\nPríkazy:"
            "\n  nauč sa <cieľ>"
            "\n  zobraz ciele"
            "\n  exit"
        )

    def learn(self, goal: str) -> None:
        if not goal:
            print("Chýba cieľ učenia.")
            return

        print(f"\nCieľ: {goal}")
        print("Vytváram učebný plán...")

        result = self.engine.learn(goal)

        print(f"Stav: {result.status}")

        if result.plan:
            print("\nPlán:")
            for index, step in enumerate(result.plan, 1):
                print(f"  {index}. {step}")

    def show_status(self) -> None:
        goals = self.engine.goals.list_goals()

        if not goals:
            print("Žiadne uložené ciele.")
            return

        for goal in goals:
            print(f"\nCieľ: {goal.get('goal')}")
            print(f"Stav: {goal.get('status')}")
            print(f"Fáza: {goal.get('current_phase')}")
            print(
                f"Dokončené úlohy: "
                f"{len(goal.get('completed_tasks', []))}"
            )
            print(
                f"Neúspešné úlohy: "
                f"{len(goal.get('failed_tasks', []))}"
            )
            print(
                f"Ďalšia úloha: "
                f"{goal.get('next_task')}"
            )

    def continue_learning(self) -> None:
        goals = self.engine.goals.list_goals()

        if not goals:
            print("Žiadny cieľ na pokračovanie.")
            return

        goal = goals[0]

        print(f"\nPokračujem v cieli: {goal.get('goal')}")
        print(f"Stav: {goal.get('status')}")
        print(f"Fáza: {goal.get('current_phase')}")
        print(f"Ďalšia úloha: {goal.get('next_task')}")

    def show_goals(self) -> None:
        goals = self.engine.goals.list_goals()

        if not goals:
            print("Žiadne uložené ciele.")
            return

        print("\nUložené ciele:")

        for goal in goals:
            print(
                f"  - {goal.get('goal')} "
                f"[{goal.get('status')}]"
            )


if __name__ == "__main__":
    NovaCLI().run()
