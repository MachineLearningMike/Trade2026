class Trajectory:
    """Simple container for trajectory data"""
    def __init__(self):
        self.rewards = []
        self.values = []
        self.dones = []
        self.log_probs = []
        self.states = []
        self.actions = []

def correct_ppo_update(trajectory, network, optimizer, 
                       gamma=0.99, lambda_=0.95, 
                       entropy_coef=1e-4, clip_epsilon=0.2):
    """
    Complete correct PPO update
    """
    
    # Convert to tensors
    states = torch.FloatTensor(np.array(trajectory.states))
    actions = torch.LongTensor(trajectory.actions)
    old_log_probs = torch.FloatTensor(trajectory.log_probs)
    old_values = torch.FloatTensor(trajectory.values)
    
    # Compute returns and advantages
    returns = []
    running_add = 0
    
    # Backwards for returns
    for t in reversed(range(len(trajectory.rewards))):
        if trajectory.dones[t]:
            running_add = 0
        running_add = trajectory.rewards[t] + gamma * running_add
        returns.insert(0, running_add)
    
    returns = torch.FloatTensor(returns)
    
    # Compute advantages using GAE
    advantages = []
    gae = 0
    next_value = 0 if trajectory.dones[-1] else old_values[-1]
    
    for t in reversed(range(len(trajectory.rewards))):
        if trajectory.dones[t]:
            next_value = 0
            gae = 0
        
        delta = trajectory.rewards[t] + gamma * next_value - old_values[t]
        gae = delta + gamma * lambda_ * gae
        advantages.insert(0, gae)
        next_value = old_values[t]
    
    advantages = torch.FloatTensor(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # PPO update loop (multiple epochs)
    for _ in range(ppo_epochs):
        # Get current policy outputs
        log_probs, state_values, entropy = network.evaluate(states, actions)
        
        # Calculate ratio
        ratios = torch.exp(log_probs - old_log_probs)
        
        # PPO loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Critic loss
        critic_loss = torch.nn.MSELoss()(state_values, returns)
        
        # Total loss (correct sign)
        total_loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy.mean()
        
        # Update network
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
        optimizer.step()
    
    return total_loss.item()