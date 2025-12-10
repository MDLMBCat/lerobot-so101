import torch
import time

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.utils.constants import ACTION
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

MAX_STEPS = 1000

def main():
    # device = torch.device("mps")  # or "cuda" or "cpu"
    device = torch.device("cuda")
    model_id = "yz31/smolvla_v1"

    model = SmolVLAPolicy.from_pretrained(model_id)
    model.eval()

    model.to(device)
    
    preprocess_time = 0
    postprocess_time = 0
    chunk_generation_count = 0
    total_chunk_generation_time = 0
    total_action_generation_time = 0
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
        "camera1": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30),
        "camera2": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
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


    init_rerun(session_name="smolvla_example")

    for _ in range(MAX_STEPS):
        preprocess_start_time = time.perf_counter()
        
        obs = robot.get_observation()
        
        log_rerun_data(observation=obs)
        
        obs_frame = build_inference_frame(
            observation=obs, ds_features=dataset_features, device=device, task=task, robot_type=robot_type
        )

        obs = preprocess(obs_frame)

        preprocess_elapsed_time = time.perf_counter() - preprocess_start_time
        preprocess_time += preprocess_elapsed_time
        
        will_generate_chunk = len(model._queues[ACTION]) == 0
        
        inference_start_time = time.perf_counter()
        
        action = model.select_action(obs)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
            
        inference_elapsed_time = time.perf_counter() - inference_start_time
        
        postprocess_start_time = time.perf_counter()
        action = postprocess(action)
        action = make_robot_action(action, dataset_features)
            
        if will_generate_chunk:
            chunk_generation_count += 1
            total_chunk_generation_time += inference_elapsed_time
        
        steps += 1
        
        robot.send_action(action)
        
        postprocess_elapsed_time = time.perf_counter() - postprocess_start_time
        postprocess_time += postprocess_elapsed_time
        total_action_generation_time += preprocess_elapsed_time + inference_elapsed_time + postprocess_elapsed_time

    if chunk_generation_count > 0:    
        average_preprocess_time = preprocess_time / steps
        average_postprocess_time = postprocess_time / steps
        average_step_generation_time = total_chunk_generation_time / steps
        average_chunk_generation_time = total_chunk_generation_time / chunk_generation_count
        average_action_generation_time = total_action_generation_time / steps
        average_theoretical_fps = 1 / average_action_generation_time
        
        print(f"Average chunk generation time: {average_chunk_generation_time:.4f} seconds ({average_chunk_generation_time*1000:.2f} miliseconds)")
        print(f"Average preprocess time: {average_preprocess_time*1000:.2f} milliseconds")
        print(f"Average postprocess time: {average_postprocess_time*1000:.2f} milliseconds")
        print(f"Average step generation time: {average_step_generation_time:.4f} seconds ({average_step_generation_time*1000:.2f} milliseconds)")
        print(f"Average action generation time: {average_action_generation_time:.4f} seconds ({average_action_generation_time*1000:.2f} milliseconds)")
        print(f"Average theoretical FPS: {average_theoretical_fps:.2f}")
        
if __name__ == "__main__":
    main()
