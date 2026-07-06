from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg, RslRlPpoActorCriticCfg

from .amp_cfg import RslRlAmpCfg

@configclass
class RslRlPpoAmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the AMP algorithm."""

    class_name: str = "PPOAmp"
    """The algorithm class name. Default is PPOAmp."""

    amp_cfg: RslRlAmpCfg = RslRlAmpCfg()
    """Configuration for the AMP (Adversarial Motion Priors) in the training."""


@configclass
class RslRlBaseRunnerCfg:
    """Base configuration of the runner."""

    seed: int = 42
    """The seed for the experiment. Defaults to 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Defaults to cuda:0."""

    num_steps_per_env: int = MISSING # type:ignore
    """The number of steps per environment per update."""

    max_iterations: int = MISSING # type:ignore
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING # type:ignore
    """This parameter is deprecated and will be removed in the future.

    For rsl-rl < 4.0.0, use `actor_obs_normalization` and `critic_obs_normalization` of the policy instead.
    For rsl-rl >= 4.0.0, use `obs_normalization` of the model instead.
    """

    obs_groups: dict[str, list[str]] = MISSING # type:ignore
    """A mapping from observation groups to observation sets.

    The keys of the dictionary are predefined observation sets used by the underlying algorithm
    and values are lists of observation groups provided by the environment.

    For instance, if the environment provides a dictionary of observations with groups "policy", "images",
    and "privileged", these can be mapped to algorithmic observation sets as follows:

    .. code-block:: python

        obs_groups = {
            "actor": ["policy", "images"],
            "critic": ["policy", "privileged"],
        }

    This way, the actor will receive the "policy" and "images" observations, and the critic will
    receive the "policy" and "privileged" observations.

    For more details, please check ``vec_env.py`` in the rsl_rl library.
    """

    clip_actions: float | None = None
    """The clipping value for actions. If None, then no clipping is done. Defaults to None.

    .. note::
        This clipping is performed inside the :class:`RslRlVecEnvWrapper` wrapper.
    """

    check_for_nan: bool = True
    """Whether to check for NaN values coming from the environment."""

    save_interval: int = MISSING # type:ignore
    """The number of iterations between saves."""

    experiment_name: str = MISSING # type:ignore
    """The experiment name."""

    run_name: str = ""
    """The run name. Defaults to empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Defaults to tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Defaults to "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Defaults to "isaaclab"."""

    resume: bool = False
    """Whether to resume a previous training. Defaults to False.

    This flag will be ignored for distillation.
    """

    load_run: str = ".*"
    """The run directory to load. Defaults to ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Defaults to ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """

@configclass
class RslRlDistillRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration of the runner for on-policy algorithms."""

    class_name: str = "DistillationRunner"
    """The runner class name. Defaults to OnPolicyRunner."""

    algorithm: RslRlTsDistillationAlgorithmCfg = MISSING # type:ignore
    """The algorithm configuration."""

    policy: RslRlStudentTeacherCfg  = MISSING # type:ignore
    """The policy configuration.

    For rsl-rl >= 4.0.0, this configuration is is deprecated. Please use `actor` and `critic` model configurations
    instead.
    """

@configclass
class RslRlStudentTeacherCfg():
    """Configuration for the distillation(recurrent) algorithm."""
    
    class_name: str = "StudentTeacherRecurrent"
    """The algorithm class name. Default is StudentTeacherRecurrent."""

    student_obs_normalization: bool = False
    """Whether to normalize the observations for the student policy."""

    teacher_obs_normalization: bool = False
    """Whether to normalize the observations for the teacher policy."""

    student_hidden_dims: tuple[int] | list[int] = [256, 256, 256]
    """The hidden dimensions for the student policy network."""

    teacher_hidden_dims: tuple[int] | list[int] = [256, 256, 256]
    """The hidden dimensions for the teacher policy network."""

    activation: str = "elu"
    """The activation function for the policy networks."""

    init_noise_std: float = 0.1
    """The initial noise standard deviation for the policy networks."""

    noise_std_type: str = "scalar"
    """The type of noise standard deviation for the policy networks. Options are 'scalar' or 'log'."""

    rnn_type: str = "lstm"
    """The type of RNN to use for the recurrent policy. Options are 'lstm' or 'gru'."""

    rnn_hidden_dim: int = 256
    """The hidden dimension for the RNN in the recurrent policy."""

    rnn_num_layers: int = 1
    """The number of layers for the RNN in the recurrent policy."""

    teacher_recurrent: bool = False
    """Whether the teacher policy is recurrent or not."""

@configclass
class RslRlTsDistillationAlgorithmCfg():
    """Configuration for the AMP algorithm."""
    
    class_name: str = "Distillation"
    """The algorithm class name. Default is Distillation."""

    num_learning_epochs: int = 1
    """The number of learning epochs for the distillation algorithm."""

    gradient_length: int = 15
    """The length of the gradient for the distillation algorithm."""

    learning_rate: float = 1e-3
    """The learning rate for the distillation algorithm."""

    max_grad_norm: float | None = None
    """The maximum gradient norm for the distillation algorithm. If None, no clipping is applied."""

    loss_type: str = "mse"
    """The loss type for the distillation algorithm. Options are 'mse' or 'kl'."""

    optimizer: str = "adam"
    """The optimizer for the distillation algorithm. Options are 'adam' or 'sgd'."""
