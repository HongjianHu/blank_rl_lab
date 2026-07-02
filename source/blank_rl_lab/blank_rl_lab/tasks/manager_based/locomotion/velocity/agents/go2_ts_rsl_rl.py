from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from blank_rl_lab.rsl_rl import RslRlStudentTeacherCfg, RslRlTsDistillationAlgorithmCfg, RslRlDistillRunnerCfg

@configclass
class Go2FlatDistillRunnerCfg(RslRlDistillRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 200
    experiment_name = "go2_demo"
    load_run = "rough_resume"
    load_checkpoint = "model_8998.pt"
    obs_groups = {
        "policy": ["policy"], 
        "teacher": ["critic"], 
    }
    policy = RslRlStudentTeacherCfg(
        class_name="StudentTeacherRecurrent",               
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[512, 256, 128],       
        teacher_hidden_dims=[512, 256, 128],       
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        teacher_recurrent=False,
    )
    algorithm = RslRlTsDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=2,
        gradient_length=15,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        loss_type="mse",
        optimizer="adam",
    )
