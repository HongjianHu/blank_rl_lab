from isaaclab.utils import configclass

from blank_rl_lab.rsl_rl import RslRlCtsActorCriticCfg, RslRlCtsAlgorithmCfg
from blank_rl_lab.rsl_rl import RslRlMoENGCTSActorCriticCfg, RslRlMoENGCTSAlgorithmCfg
from blank_rl_lab.tasks.manager_based.locomotion.legged.velocity.agents.go2_rsl_rl_cts_cfg import Go2CtsRunnerCfg

@configclass
class Go2MoENGCtsRunnerCfg(Go2CtsRunnerCfg):
    experiment_name = "go2_moe_no_goal_cts"

    policy = RslRlMoENGCTSActorCriticCfg(
        class_name="ActorCriticMoENGCTS",
        history_length=5,
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        teacher_encoder_hidden_dims=[512, 256],
        student_encoder_hidden_dims=[512, 256],
        student_expert_num=8,
        student_expert_hidden_dim=256,
        activation="elu",
        latent_dim=32,
        norm_type="l2norm",
        obs_no_goal_mask=(
            [True] * 6
            + [False] * 3
            + [True] * 36
        ),
    )

    algorithm = RslRlMoENGCTSAlgorithmCfg(
        class_name="MoENGCTS",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        student_encoder_learning_rate=1.0e-3,
        load_balance_coef=0.01,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        teacher_env_ratio=0.75,
    )