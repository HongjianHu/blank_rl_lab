from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from blank_rl_lab.rsl_rl import RslRlCtsActorCriticCfg, RslRlCtsAlgorithmCfg
from blank_rl_lab.rsl_rl import RslRlMoENGCTSActorCriticCfg, RslRlMoENGCTSAlgorithmCfg

@configclass
class Go2CtsRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyRunnerCTS"
    load_optimizer: bool = True
    seed = 0
    num_steps_per_env = 24
    max_iterations = 150000
    save_interval = 500
    experiment_name = "go2_cts"
    empirical_normalization = False
    clip_actions = 100.0
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }

    policy = RslRlCtsActorCriticCfg(
        class_name="ActorCriticCTS",
        history_length=5,
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        teacher_encoder_hidden_dims=[512, 256],
        student_encoder_hidden_dims=[512, 256],
        activation="elu",
        latent_dim=32,
        norm_type="l2norm",
    )

    algorithm = RslRlCtsAlgorithmCfg(
        class_name="CTS",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        student_encoder_learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        teacher_env_ratio=0.75,
    )
