from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from blank_rl_lab.rsl_rl import RslRlPpoAmpAlgorithmCfg, RslRlAmpCfg

@configclass
class Go2RslRlOnPolicyRunnerAmpCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AMPRunner"
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 200
    experiment_name = "go2_amp"
    obs_groups = {
        "policy": ["policy"], 
        "critic": ["critic"], 
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"]
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
    )
    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_cfg=RslRlAmpCfg(
            # 256 * 4096 envs ~= the original 1e6 AMP replay transitions.
            disc_obs_buffer_size=256,
            grad_penalty_scale=10.0,
            disc_trunk_weight_decay=1.0e-4,
            disc_linear_weight_decay=1.0e-2,
            disc_learning_rate=5.0e-5,
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[512, 256],
                activation="lrelu",
                style_reward_scale=0.3,
                task_style_lerp=0.8
            ),
            loss_type="LSGAN"
        ),
        # symmetry_cfg=RslRlSymmetryCfg(
        #     use_data_augmentation=True, data_augmentation_func=g1.compute_symmetric_states,
        #     use_mirror_loss=True, mirror_loss_coeff=0.1,
        # )
    )
