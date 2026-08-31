# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    """创建并返回一个环境构建函数（thunk），用于延迟初始化环境。"""
    def thunk():
        # 如果启用了视频录制且当前是第0个环境，则创建带渲染模式的环境并包装为视频录制环境
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            # 否则，正常创建环境（无渲染）
            env = gym.make(env_id)
        # 将观察空间展平为一维向量（处理如 dm_control 的 Dict 观察空间）
        env = gym.wrappers.FlattenObservation(env)
        # 记录每个 episode 的统计信息（如回报、长度）
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # 将动作裁剪到合法范围
        env = gym.wrappers.ClipAction(env)
        # 对观察值进行归一化（运行均值和方差）
        env = gym.wrappers.NormalizeObservation(env)
        # 将归一化后的观察值裁剪到 [-10, 10] 范围内，防止极端值
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # 对奖励进行归一化，使用折扣因子 gamma
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        # 将归一化后的奖励裁剪到 [-10, 10] 范围内，防止极端奖励
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """对神经网络层进行正交初始化（Orthogonal Initialization）并设置偏置。

    正交初始化通过使权重矩阵的行/列向量两两正交且模长为1，保证深层网络前向传播
    与反向传播过程中信号范数稳定，从而缓解梯度消失与梯度爆炸问题。标准差参数 std
    用于缩放正交矩阵，以匹配后续激活函数的统计特性（如 ReLU/Tanh 的期望输出方差）。

    适用对象：具备 .weight 与 .bias 属性的层，如 nn.Linear、nn.Conv2d 等。
    不适用对象：无参数的层（如 nn.Tanh）、仅有 weight 无 bias 的层（如 nn.Embedding）。

    Args:
        layer: 待初始化的网络层。
        std (float): 正交矩阵的缩放标准差。默认值 np.sqrt(2) 适用于 Tanh/ReLU 后的隐藏层；
            输出层常取 1.0（价值估计）或 0.01（策略均值，使初始策略接近确定性）。
        bias_const (float): 偏置初始化的常数值，默认 0.0。

    Returns:
        初始化后的 layer。
    """
    # 使用正交分布初始化权重矩阵，并以 std 进行缩放
    torch.nn.init.orthogonal_(layer.weight, std)
    # 将偏置向量所有元素初始化为指定常数 bias_const
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """PPO 智能体网络，包含价值网络（critic）与策略网络（actor）。

    网络架构说明：
    1. 本网络采用多层感知机（MLP, Multi-Layer Perceptron）结构，即全连接前馈网络。
       MLP 由交替的线性层（nn.Linear）与激活函数（nn.Tanh）串联而成，信息单向流动，
       适用于向量形式的观察输入（如机器人关节角度、速度等连续状态）。

    2. nn.Sequential 是 PyTorch 提供的容器，用于按顺序串联各层，数据流严格遵循
       定义的顺序：input → layer1 → layer2 → ... → output。
       若需更复杂的连接方式（如并联分支、跳跃连接/残差连接、多输入多输出等），
       需继承 nn.Module 并自定义 forward 方法。

    3. nn.Linear(in_features, out_features) 定义全连接层，数学形式为 y = xW^T + b。
       参数含义：
       - in_features: 输入特征维度（前一层的神经元数）。
       - out_features: 输出特征维度（当前层的神经元数）。
       例如 nn.Linear(64, 64) 表示输入 64 维、输出 64 维，权重矩阵形状为 (64, 64)。

    4. 网络设计并非完全随意，需遵循维度匹配原则：前一层的 out_features 必须等于
       后一层的 in_features。此外，隐藏层维度、激活函数选择、层数等属于超参数，
       通常依据任务复杂度、经验惯例及实验验证确定。本代码中 64×64 的隐藏层与
       Tanh 激活函数是连续控制任务中的常见配置，兼顾表达能力与训练稳定性。

    5. 正交初始化（layer_init）中 std 的选取依据：
       - 隐藏层使用默认值 sqrt(2)：补偿 Tanh/ReLU 激活函数造成的信号方差衰减。
       - critic 输出层 std=1.0：保持价值估计的初始尺度适中。
       - actor 均值输出层 std=0.01：使初始策略接近零均值，增强训练初期探索的稳定性。
    """

    def __init__(self, envs):
        """初始化智能体网络。

        Args:
            envs: 向量化环境对象，用于获取观察和动作空间的形状信息。
        """
        super().__init__()
        # 构建价值网络（critic）：输入为环境观察，输出为标量状态价值估计 V(s)
        self.critic = nn.Sequential(
            # 输入层 → 第一隐藏层：将观察空间展平后的维度映射至 64 维
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),  # 双曲正切激活函数，输出范围 (-1, 1)，引入非线性并抑制极端值
            # 第一隐藏层 → 第二隐藏层：维持 64 维特征表示
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            # 第二隐藏层 → 输出层：输出单个标量值，std=1.0 保持初始输出尺度
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # 构建策略网络（actor）的均值部分：输入为观察，输出为连续动作的均值 μ(s)
        self.actor_mean = nn.Sequential(
            # 输入层 → 第一隐藏层：维度同 critic，共享观察输入结构
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            # 第一隐藏层 → 第二隐藏层
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            # 第二隐藏层 → 输出层：输出动作均值，std=0.01 使初始策略接近零均值（确定性）
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        # 可学习的对数标准差参数 log(σ)，独立于状态，初始化为全零（即 σ=1）
        # 形状为 (1, action_dim)，通过广播机制适配 batch 维度
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        """根据输入观察 x 计算状态价值估计。

        Args:
            x: 输入观察张量。

        Returns:
            状态价值估计。
        """
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """根据输入观察 x 计算动作、动作的对数概率、策略熵和状态价值。

        Args:
            x: 输入观察张量。
            action: 可选，如果提供则计算该动作的对数概率；否则采样新动作。

        Returns:
            action: 动作张量。
            log_prob: 动作的对数概率（按维度求和后）。
            entropy: 策略的熵（按维度求和后）。
            value: 状态价值估计。
        """
        # 通过 actor_mean 网络计算动作的均值
        action_mean = self.actor_mean(x)
        # 将可学习的对数标准差扩展到与 action_mean 相同的形状
        action_logstd = self.actor_logstd.expand_as(action_mean)
        # 对数标准差取指数得到标准差
        action_std = torch.exp(action_logstd)
        # 构造正态分布对象
        probs = Normal(action_mean, action_std)
        # 如果没有提供 action，则从分布中采样一个动作
        if action is None:
            action = probs.sample()
        # 返回动作、对数概率（sum(1) 表示对动作各维度求和）、熵（sum(1)）和状态价值
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()