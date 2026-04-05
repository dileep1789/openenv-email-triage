from env.environment import OpenEnv

def test_environment():
    print("Testing environment setup...")
    env = OpenEnv()
    
    # Test reset
    obs = env.reset("easy_classification")
    print(f"Observation: {obs.email.subject}")
    
    # Test step
    action = {
        "type": "classify",
        "category": "complaint",
        "reasoning": "Test action"
    }
    
    next_obs, reward, done, info = env.step(action)
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    
    if reward > 0 and done:
        print("Test passed!")
    else:
        print(f"Test failed with reward {reward}")

if __name__ == "__main__":
    test_environment()
