from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from blank_rl_lab.rsl_rl import ExtremeParkourTeacherPolicyCfg, ExtremeParkourPPOAlgorithmCfg

@configclass
class Go2ExtremeParkourTeacherRunnerCfg(RslRlOnPolicyRunnerCfg):

    class_name = "ExtremeParkourRunner"

    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 100
    experiment_name = "go2_extreme_parkour_teacher"
    run_name = "no_action_delay"
    load_optimizer: bool = True
    observation_contract_version: str = "go2_extreme_parkour_teacher_v1"
    clip_actions = 100.0
    empirical_normalization = False
    obs_groups = {
        "policy": ["proprio", "terrain_scan", "priv_explicit", "priv_latent"],
        "critic": ["proprio", "terrain_scan", "priv_explicit", "priv_latent", "proprio_history"],
        "adaptation": ["proprio_history"],
        "estimator": ["proprio"],
        "estimator_target": ["priv_explicit"],
    }
    policy = ExtremeParkourTeacherPolicyCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        state_dependent_std=False,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = ExtremeParkourPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        optimizer="adam",
        normalize_advantage_per_mini_batch=False,
        rnd_cfg=None,
        symmetry_cfg=None,
    )
