__app_name__ = "mimicbot_cli"
__version__ = "1.0.0"

# must init config file here

(
    SUCCESS,
    UNKNOWN_ERROR,
    DIR_ERROR,
    FILE_ERROR,
    API_KEY_ERROR,
    PRIVILAGES_ERROR,
    BOT_ERROR,
    MISSING_GUILD_ERROR,
    USER_NAME_ERROR,
    CHANGE_VALUE,
    GPU_ERROR,
    ABORT,
) = range(12)

ERROR = {
    SUCCESS: "SUCCESS",
    UNKNOWN_ERROR: "UNKNOWN_ERROR",
    DIR_ERROR: "DIR_ERROR",
    FILE_ERROR: "FILE_ERROR",
    API_KEY_ERROR: "API_KEY_ERROR",
    PRIVILAGES_ERROR: "PRIVILAGES_ERROR",
    BOT_ERROR: "BOT_ERROR",
    MISSING_GUILD_ERROR: "MISSING_GUILD_ERROR",
    USER_NAME_ERROR: "USER_NAME_ERROR",
    CHANGE_VALUE: "CHANGE_VALUE",
    GPU_ERROR: "GPU_ERROR",
    ABORT: "ABORT",
}

class Args():
    def __init__(self):
        self.output_dir = None
        self.device = None
        self.model_type = 'gpt2'
        self.model_path = None
        self.model_name = None
        self.config_name = "microsoft/DialoGPT-small"
        self.tokenizer_name = None
        self.save_to = None
        self.repo = None
        self.cache_dir = None
        self.block_size = 512
        self.do_train = True
        self.do_eval = True
        self.evaluate_during_training = False
        self.per_gpu_train_batch_size = 4
        self.per_gpu_eval_batch_size = 4
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 1
        self.max_steps = -1
        self.warmup_steps = 0
        self.logging_steps = 1000
        self.save_steps = 3500
        self.save_total_limit = None
        self.eval_all_checkpoints = False
        self.no_cuda = None
        self.overwrite_output_dir = True
        self.overwrite_cache = True
        self.should_continue = False
        self.seed = 42
        self.local_rank = -1
        self.fp16 = False
        self.fp16_opt_level = 'O1'
