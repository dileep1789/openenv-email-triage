from env.environment import OpenEnv


def test_environment() -> None:
    print("Testing OpenEnv environment...")
    env = OpenEnv()

    obs = env.reset("medium_response")
    print(f"Phase 1: {obs.phase_name}")

    _, reward_1, done_1, _ = env.step(
        {
            "type": "classify",
            "category": "support",
            "reasoning": "Account access issue maps to support.",
        }
    )

    _, reward_2, done_2, _ = env.step(
        {
            "type": "decide",
            "decision": "reply",
            "reasoning": "Direct response can resolve account issue.",
        }
    )

    _, reward_3, done_3, _ = env.step(
        {
            "type": "respond",
            "response": (
                "We sent a reset link. Please verify identity before use. "
                "The link is valid for 24 hours."
            ),
            "reasoning": "Includes required security policy language.",
        }
    )

    final_state = env.state()
    print(f"Rewards: {reward_1:.4f}, {reward_2:.4f}, {reward_3:.4f}")
    print(f"Done flags: {done_1}, {done_2}, {done_3}")
    print(f"Final score: {final_state.cumulative_reward:.4f}")

    if done_3 and final_state.cumulative_reward > 0.8:
        print("Test passed!")
    else:
        print("Test failed.")


if __name__ == "__main__":
    test_environment()
