import os
from os.path import join

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.utils.utils import str2bool


def add_doom_env_args(parser):
    p = parser

    p.add_argument(
        "--num_agents",
        default=-1,
        type=int,
        help="Allows to set number of agents less than number of players, to allow humans to join the match. Default value (-1) means default number defined by the environment",
    )
    p.add_argument("--num_humans", default=0, type=int, help="Meatbags want to play?")
    p.add_argument(
        "--num_bots",
        default=-1,
        type=int,
        help="Add classic (non-neural) bots to the match. If default (-1) then use number of bots specified in env cfg",
    )
    p.add_argument(
        "--start_bot_difficulty", default=None, type=int, help="Adjust bot difficulty, useful for evaluation"
    )
    p.add_argument(
        "--timelimit", default=None, type=float, help="Allows to override default match timelimit in minutes"
    )
    p.add_argument("--res_w", default=128, type=int, help="Game frame width after resize")
    p.add_argument("--res_h", default=72, type=int, help="Game frame height after resize")
    p.add_argument(
        "--wide_aspect_ratio",
        default=False,
        type=str2bool,
        help="If true render wide aspect ratio (slower but gives better FOV to the agent)",
    )
    p.add_argument('--level', type=int, default=1, choices=[1, 2, 3], help='Difficulty level')
    p.add_argument('--constraint', type=str, default='soft', choices=['soft', 'hard'], help='Soft/Hard safety constraint')
    p.add_argument('--render_mode', type=str, default='rgb_array', help='Rendering mode')
    p.add_argument("--video_dir", default='videos', type=str, help="Record episodes to this folder after an interval.")
    p.add_argument("--video_length", default=2100, type=int, help="Length of recorded video.")
    p.add_argument("--record_every", default=5000, type=int, help="Interval after how many steps to record a video.")
    p.add_argument("--record", default=True, type=str2bool, help="Whether to record gameplay.")
    p.add_argument('--resolution', type=str, default='160x120', choices=['1600x1200', '1280x720', '800x600', '640x480', '320x240', '160x120'], help='Screen resolution of the game')
    p.add_argument('--resolution_eval', type=str, default='1280x720', help='Screen resolution of the evaluation video')


def add_doom_env_eval_args(parser):
    """Arguments used only during evaluation."""
    parser.add_argument(
        "--record_to",
        # default=join(os.getcwd(), "..", "recs"),
        default=None,
        type=str,
        help="Record episodes to this folder. This records a demo that can be replayed at full resolution. "
             "Currently, this does not work for bot environments so it is recommended to use --save_video to "
             "record episodes at lower resolution instead for such environments",
    )


def doom_override_defaults(parser):
    """RL params specific to Doom envs."""
    parser.set_defaults(
        ppo_clip_value=0.2,
        obs_subtract_mean=0.0,
        obs_scale=255.0,
        exploration_loss="symmetric_kl",
        exploration_loss_coeff=0.001,
        normalize_returns=True,
        normalize_input=True,
        env_frameskip=4,
        eval_env_frameskip=1,  # this is for smoother rendering during evaluation
        fps=35,  # for evaluation only
        heartbeat_reporting_interval=600,
    )


def default_doom_cfg(algo="APPO", env="env", experiment="test"):
    """Useful in tests."""
    argv = [f"--algo={algo}", f"--env={env}", f"--experiment={experiment}"]
    parser, args = parse_sf_args(argv)
    add_doom_env_args(parser)
    doom_override_defaults(parser)
    args = parse_full_cfg(parser, argv)
    return args
