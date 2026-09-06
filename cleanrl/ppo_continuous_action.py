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
    """创建并返回一个环境构建函数（thunk），用于延迟初始化环境。

    说明：
    1. Gymnasium 是上层环境接口标准，底层物理仿真由具体后端（如 MuJoCo、Isaac Sim、
       PyBullet 等）实现。用户可通过自定义 Wrapper 或继承环境类修改奖励函数。
    2. 奖励由环境根据当前状态和动作计算返回，非 PPO 算法定义。不同环境奖励函数各异，
       可通过 gym.wrappers.TransformReward 进行简单变换，或通过自定义 Wrapper 实现
       基于状态的动态奖励（如鼓励平滑动作、保持直立等）。
    """
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
        # 对奖励进行归一化，使用折扣因子 gamma；不改变奖励语义，仅稳定训练
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        # 将归一化后的奖励裁剪到 [-10, 10] 范围内，防止极端奖励导致梯度不稳定
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

    6. 强化学习 Agent 通常继承 nn.Module，因为：
       - 需要可学习的参数（nn.Parameter）和子模块（如 nn.Linear）；
       - 需要 .to(device) 将网络移至 GPU/CPU；
       - 需要 .state_dict() 保存/加载模型；
       - 需要与 PyTorch 自动求导机制集成（backward/update）。
       不同算法的 Agent 结构各异：PPO 使用 Actor-Critic，DQN 仅用 Q 网络，
       SAC 使用 Actor + 双 Critic 等，但均基于 nn.Module 构建。
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
        """根据输入观察 x 计算动作、动作的对数概率密度、策略熵和状态价值。

        Args:
            x: 输入观察张量。
            action: 可选，如果提供则计算该动作的对数概率密度；否则采样新动作。

        Returns:
            action: 动作张量。
            log_prob: 给定状态 s 下策略选择动作 a 的对数概率密度 log π(a|s)。
                连续动作空间中使用概率密度（非概率），因单点概率为零。
                多维独立高斯分布下，联合概率 = 各维度概率之积，取对数后变为求和，
                因此 .sum(1) 在动作维度上求和得到总的 log_prob。
                在 PPO 中用于计算新旧策略比率 r(θ) = exp(log π_new - log π_old)。
            entropy: 策略的熵（按维度求和后），衡量策略的随机性/不确定性。
            value: 状态价值估计。
        """
        # 通过 actor_mean 网络计算动作的均值
        action_mean = self.actor_mean(x)
        # 将可学习的对数标准差扩展到与 action_mean 相同的形状
        action_logstd = self.actor_logstd.expand_as(action_mean)
        # 对数标准差取指数得到标准差
        action_std = torch.exp(action_logstd)
        # 构造正态分布对象：各动作维度独立服从一维高斯分布 N(μ_d, σ_d²)
        probs = Normal(action_mean, action_std)
        # 如果没有提供 action，则从分布中采样一个动作
        if action is None:
            action = probs.sample()
        # 返回动作、对数概率密度、熵和状态价值
        # .sum(1) 在动作维度上求和：log π(a|s) = Σ_d log π(a_d|s)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    # 使用 tyro 库解析命令行参数，将 Args 数据类中的字段映射为命令行选项
    args = tyro.cli(Args)
    # 计算每个训练迭代的总样本数：并行环境数 × 每环境每轮次步数
    args.batch_size = int(args.num_envs * args.num_steps)
    # 计算每个小批量（mini-batch）的样本数：总批量大小 // 小批量数量
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # 计算总训练迭代次数：总时间步数 // 每迭代批量大小
    args.num_iterations = args.total_timesteps // args.batch_size
    # 构造运行名称，包含环境ID、实验名、随机种子和时间戳，用于唯一标识本次实验
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # 若启用跟踪（track），则初始化 Weights & Biases（wandb）进行实验跟踪
    if args.track:
        import wandb

        # 初始化 wandb 项目，同步 TensorBoard 日志，记录超参数和代码
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # 创建 TensorBoard 的 SummaryWriter，日志保存至 runs/{run_name} 目录
    writer = SummaryWriter(f"runs/{run_name}")
    # 将所有超参数以 Markdown 表格形式写入 TensorBoard，便于后续查看
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    # 设置 Python 内置 random、NumPy 和 PyTorch 的随机种子，确保实验可复现
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # 设置 CuDNN 的确定性模式，保证在 GPU 上的计算结果可复现（可能牺牲部分性能）
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # 根据 CUDA 是否可用及用户配置，选择计算设备（GPU 或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # 使用 gym.vector.SyncVectorEnv 创建同步向量化环境，并行运行多个环境实例以提高采样效率
    # Gymnasium 是上层环境接口标准，底层物理仿真由具体后端（如 MuJoCo、Isaac Sim 等）实现
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    # 断言动作空间为连续型（gym.spaces.Box），本算法仅支持连续动作空间
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # 实例化 Agent（策略网络与价值网络），并将其移动到指定计算设备（GPU/CPU）
    agent = Agent(envs).to(device)
    # 创建 Adam 优化器，用于更新 agent 的参数；设置学习率及数值稳定性常数 eps
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    # 初始化数据缓冲区，用于存储每个 rollout 收集的观察、动作、对数概率、奖励、终止标志和价值估计
    # 缓冲区为多维张量（Tensor），形状 (num_steps, num_envs, ...)，表示每步每个环境的数据
    # 例如 obs 形状为 (2048, 4, 17)：2048 步 × 4 个并行环境 × 17 维观察
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    # 初始化全局步数计数器和训练开始时间戳
    global_step = 0
    start_time = time.time()
    # 重置所有环境，获取初始观察；gymnasium 返回 (observation, info)
    # 环境返回的观察为 NumPy 数组（CPU 内存），需转换为 PyTorch Tensor 并移至 GPU
    next_obs, _ = envs.reset(seed=args.seed)
    # 将 NumPy 数组转换为 PyTorch Tensor 并移至计算设备（GPU/CPU）
    # 此转换必要原因：1) Gymnasium 环境使用 NumPy，PyTorch 网络需要 Tensor；2) 数据需在 GPU 上加速计算
    next_obs = torch.Tensor(next_obs).to(device)
    # 初始化下一时刻的终止标志，0 表示所有环境均未结束
    next_done = torch.zeros(args.num_envs).to(device)

    # 外层循环：训练迭代。每次迭代收集一批数据并进行多轮网络更新
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        # 若启用学习率退火（anneal_lr），则根据当前迭代进度线性衰减学习率
        if args.anneal_lr:
            # 计算剩余训练比例，从 1.0 线性降至接近 0
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            # 当前学习率 = 剩余比例 × 初始学习率
            lrnow = frac * args.learning_rate
            # 更新优化器中的学习率参数
            optimizer.param_groups[0]["lr"] = lrnow

        # 内层循环：每个 rollout 的步数循环，收集 num_steps 步的环境交互数据
        for step in range(0, args.num_steps):
            # 累计全局步数（每步增加 num_envs，因为是并行环境）
            global_step += args.num_envs
            # 将当前观察存入缓冲区对应位置
            # next_obs 形状为 (num_envs, obs_dim)，同时包含所有并行环境的观察
            obs[step] = next_obs
            # 将当前终止标志存入缓冲区
            dones[step] = next_done

            # ALGO LOGIC: action logic
            # 在数据收集阶段（Rollout）使用 torch.no_grad() 上下文，避免构建计算图：
            # 1. 前向传播不保存中间激活值与梯度信息，显著降低显存占用并加速推理；
            # 2. 明确区分"数据收集"与"参数更新"阶段，此阶段仅执行策略推理与环境交互，
            #    不更新网络参数，因此无需计算梯度；
            # 3. 若误在此阶段保留梯度，将导致显存浪费且无法释放，直至后续更新阶段。
            with torch.no_grad():
                # 通过策略网络采样动作，并获取对应的对数概率与价值估计
                # 返回的 action 形状为 (num_envs, action_dim)，同时包含所有并行环境的动作
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                # 将价值估计展平为 (num_envs,) 以匹配缓冲区 values[step] 的形状 (num_envs,)
                values[step] = value.flatten()
            # 将动作与对数概率存入对应步的缓冲区，供后续 PPO 更新阶段使用
            # 赋值操作本身不涉及梯度计算，放在 with 块内外效果相同
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            # 在环境中执行动作，获取下一观察、奖励、终止标志、截断标志和额外信息
            # action 为 GPU 上的 PyTorch Tensor，需先 .cpu() 移至 CPU，再 .numpy() 转为 NumPy 数组
            # 原因：Gymnasium 环境底层使用 NumPy，且运行在 CPU 上
            # 注意：SyncVectorEnv 在 episode 结束时会自动调用 reset() 重置该环境，
            # 因此 next_obs 中已结束环境对应的是新 episode 的初始观察，循环无需中断。
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            # 计算下一时刻的终止标志：终止（termination）或截断（truncation）满足其一即视为 episode 结束
            # termination：自然终止（任务完成或失败）；truncation：人为截断（达到最大步数限制）
            # next_done 不控制循环是否继续，而是作为 GAE 计算中的掩码：
            # 当 done=True 时，nextnonterminal=0，截断优势的跨 episode 累积，并阻止价值 bootstrap。
            next_done = np.logical_or(terminations, truncations)
            # 将奖励转换为 Tensor 并调整形状后存入缓冲区
            # 奖励由环境根据当前状态和动作计算返回，非 PPO 算法定义；不同环境奖励函数各异
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            # 将下一观察和终止标志转换为 Tensor 并移至设备
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # 若 infos 中包含 "final_info"，说明有环境完成了一个 episode，记录其统计信息
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        # 打印并记录 episode 的累计回报（return）和长度（length）到 TensorBoard
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        # 使用广义优势估计（GAE）计算优势函数和回报（returns）
        with torch.no_grad():
            # 获取最后一步的状态价值，用于 bootstrap（引导/自举）
            next_value = agent.get_value(next_obs).reshape(1, -1)
            # 初始化优势函数缓冲区，形状与 rewards 相同
            advantages = torch.zeros_like(rewards).to(device)
            # 初始化 GAE 的累积优势变量
            lastgaelam = 0
            # 逆序遍历时间步，从最后一步向前计算优势
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    # 最后一步：使用 next_done 和 next_value（来自环境重置后的状态）
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    # 非最后一步：使用下一时间步的 done 标志和价值估计
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                # 计算 TD 残差（Temporal Difference residual）
                # delta = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                # 计算 GAE 优势：A_t = delta_t + γ * λ * (1 - done) * A_{t+1}
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            # 回报 = 优势 + 价值估计（即 GAE-Lambda 形式的 Q 值估计）
            returns = advantages + values

        # flatten the batch
        # 将缓冲区数据展平，合并前两个维度 (num_steps, num_envs) 为 batch_size，
        # 以便后续随机打乱并构造小批量训练。
        # reshape 语法说明：
        #   (-1,) 是单元素元组，-1 表示自动推断该维度大小 = num_steps × num_envs；
        #   + 是 Python 元组拼接运算符，如 (-1,) + (17,) = (-1, 17)；
        #   因此 obs 从 (2048, 4, 17) 展平为 (8192, 17)，保留观察维度结构。
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        # 创建索引数组 [0, 1, 2, ..., batch_size-1]，用于后续随机打乱并构造小批量
        # np.arange(N) 生成从 0 到 N-1 的整数数组，类似 Python 的 range(N) 但返回 NumPy 数组
        b_inds = np.arange(args.batch_size)
        # 初始化列表，记录每次更新中被裁剪（clipped）的样本比例
        clipfracs = []
        # 内层循环：对收集的数据进行多轮（update_epochs）策略和价值网络更新
        for epoch in range(args.update_epochs):
            # 原地随机打乱索引数组，使每个 epoch 中小批量的样本顺序不同，增加训练随机性
            # np.random.shuffle(arr) 对数组进行 Fisher-Yates 原地洗牌，无返回值
            np.random.shuffle(b_inds)
            # 按 minibatch_size 将 batch_size 个样本划分为多个小批量
            # range(start, stop, step) 生成从 start 到 stop（不含）步长为 step 的序列
            # 例如 batch_size=8192, minibatch_size=2048 → start 依次为 0, 2048, 4096, 6144
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # 使用当前网络参数，重新计算小批量样本的动作对数概率、熵和价值
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                # 计算新旧策略的对数概率比率（log ratio）
                logratio = newlogprob - b_logprobs[mb_inds]
                # 比率 ratio = exp(logratio)，用于 PPO 的裁剪目标函数
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # 计算近似的 KL 散度，用于监控策略更新幅度和早停判断
                    # old_approx_kl: 一阶近似（-logratio 的均值）
                    old_approx_kl = (-logratio).mean()
                    # approx_kl: 更精确的二阶近似（(ratio - 1) - logratio 的均值）
                    approx_kl = ((ratio - 1) - logratio).mean()
                    # 记录被裁剪的样本比例：ratio 超出 [1-clip_coef, 1+clip_coef] 的样本占比
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # 提取当前小批量的优势值
                mb_advantages = b_advantages[mb_inds]
                # 若启用优势归一化，则对小批量优势进行零均值单位方差标准化（加 1e-8 防止除零）
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                # PPO 裁剪目标函数：取未裁剪和裁剪后目标的最大值，限制策略更新幅度
                # pg_loss1: 未裁剪的策略梯度损失（标准策略梯度形式）
                pg_loss1 = -mb_advantages * ratio
                # pg_loss2: 裁剪后的策略梯度损失，ratio 被限制在 [1-clip_coef, 1+clip_coef]
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                # 最终策略损失取两者最大值，防止策略更新过大
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                # 将价值估计展平为一维，以匹配目标回报的形状
                newvalue = newvalue.view(-1)
                # 若启用价值函数裁剪（clip_vloss），则使用裁剪后的价值损失
                if args.clip_vloss:
                    # 未裁剪的价值损失：新价值与目标回报的均方误差
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    # 裁剪后的价值：在旧价值基础上限制变化幅度不超过 clip_coef
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    # 裁剪后的价值损失
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    # 取未裁剪和裁剪后损失的较大值，防止价值函数更新过快
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    # 若不裁剪，使用标准均方误差损失
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # 计算策略熵的均值，用于鼓励探索（熵奖励）
                entropy_loss = entropy.mean()
                # 综合损失函数：策略损失 - 熵系数 × 熵损失 + 价值系数 × 价值损失
                # 负号表示最大化熵（鼓励探索），价值损失最小化
                # loss 数学上是参数 θ 的函数 L(θ)，代码中代入当前 θ 和数据后为标量，
                # 但携带计算图（grad_fn），记录了对 θ 的依赖关系，使反向传播成为可能。
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # 梯度清零，防止梯度累积
                optimizer.zero_grad()
                # 反向传播：沿计算图动态应用链式法则，计算 ∂loss/∂θ
                # 不采用"提前推导符号梯度表达式"的方式，原因：
                # 1. 符号微分未简化时表达式规模随层数指数/双指数膨胀，存储不可行；
                # 2. 简化过程的代价 ≥ 自动微分的代价，且简化结果等价于反向传播；
                # 3. PPO 的 clip/min 使梯度分段定义，符号表达式不唯一。
                # 自动微分本质上是隐式地完成符号简化 + 数值求值，计算量仅约 2× 前向传播。
                loss.backward()
                # 梯度裁剪：限制全局梯度范数不超过 max_grad_norm，防止梯度爆炸
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                # 执行参数更新（Adam 优化器的一步）：θ = θ - lr × (∂loss/∂θ 的自适应修正)
                optimizer.step()

            # 若设置了目标 KL 散度阈值且当前近似 KL 超过阈值，则提前终止当前迭代的数据更新
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # 计算价值函数的解释方差（explained variance），衡量价值网络对回报的拟合程度
        # 将 Tensor 转换为 NumPy 数组以利用 NumPy 的统计函数
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        # 计算目标回报（y_true）的方差
        var_y = np.var(y_true)
        # explained_var = 1 - Var(y_true - y_pred) / Var(y_true)，越接近 1 表示拟合越好
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # 将训练指标写入 TensorBoard，用于可视化训练过程
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # 计算并打印每秒步数（Steps Per Second, SPS），衡量训练效率
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    # 若启用模型保存，则将训练好的模型参数保存至指定路径
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        # 保存 agent 的状态字典（state_dict），包含所有可学习参数
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        # 导入评估模块，对保存的模型进行策略评估
        from cleanrl_utils.evals.ppo_eval import evaluate

        # 运行评估，计算 10 个 episode 的累计回报
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
        # 将每个评估 episode 的回报记录到 TensorBoard
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        # 若启用模型上传，则将模型和评估结果推送至 Hugging Face Hub
        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            # 构造模型仓库名称和 ID
            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    # 关闭向量化环境，释放资源
    envs.close()
    # 关闭 TensorBoard 的 SummaryWriter，确保所有日志写入磁盘
    writer.close()