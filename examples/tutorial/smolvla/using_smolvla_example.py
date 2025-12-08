import torch
import time

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower

MAX_EPISODES = 20
MAX_STEPS_PER_EPISODE = 20

def main():
    # device = torch.device("mps")  # or "cuda" or "cpu"
    device = torch.device("cuda")
    model_id = "yz31/smolvla_v1"

    model = SmolVLAPolicy.from_pretrained(model_id)

    inference_time = 0
    steps = 0

    preprocess, postprocess = make_pre_post_processors(
        model.config,
        model_id,
        # This overrides allows to run on MPS, otherwise defaults to CUDA (if available)
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # find ports using lerobot-find-port
    follower_port = "/dev/ttyACM0"  # something like "/dev/tty.usbmodem58760431631"

    # the robot ids are used the load the right calibration files
    follower_id = "R07254136"  # something like "follower_so100"

    # Robot and environment configuration
    # Camera keys must match the name and resolutions of the ones used for training!
    # You can check the camera keys expected by a model in the info.json card on the model card on the Hub
    camera_config = {
        "camera1": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
        "camera2": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30),
    }

    robot_cfg = SO101FollowerConfig(port=follower_port, id=follower_id, cameras=camera_config)
    robot = SO101Follower(robot_cfg)
    robot.connect()

    task = "Pick the white block and drop it in the black box"  # something like "pick the red block"
    robot_type = "so101_follower"  # something like "so100_follower" for multi-embodiment datasets

    # This is used to match the raw observation keys to the keys expected by the policy
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    for _ in range(MAX_EPISODES):
        for _ in range(MAX_STEPS_PER_EPISODE):
            obs = robot.get_observation()
            obs_frame = build_inference_frame(
                observation=obs, ds_features=dataset_features, device=device, task=task, robot_type=robot_type
            )

            obs = preprocess(obs_frame)

            start_time = time.perf_counter()
            
            action = model.select_action(obs)
            action = postprocess(action)
            action = make_robot_action(action, dataset_features)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            inference_time += elapsed_time
            steps += 1
            
            robot.send_action(action)

        print("Episode finished! Starting new episode...")
        
    average_inference_time = inference_time / steps
    print(f"Average Inference time: {average_inference_time*1000:.2f} milliseconds")


if __name__ == "__main__":
    main()
