from inference import TASK_IDS, run_episode
from env.environment import OpenEnv


def main() -> None:
    env = OpenEnv()
    print("[START - MOCK BASELINE]")

    total = 0.0
    for task_id in TASK_IDS:
        score, trace = run_episode(env, task_id, client=None)
        total += score

        print(f"\n[Task] {task_id}")
        for step in trace:
            print(f"- phase={step['phase']} reward={step['reward']:.4f} feedback={step['feedback']}")
        print(f"[Task Score] {score:.4f}")

    print(f"\n[Average Score] {total / len(TASK_IDS):.4f}")
    print("[END - MOCK BASELINE]")


if __name__ == "__main__":
    main()
