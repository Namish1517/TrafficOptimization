
# train_2x2.py
import os
import inspect
import sumo_rl
from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# locate package root
base = os.path.dirname(inspect.getfile(sumo_rl))
print("sumo_rl package base:", base)

# candidate net locations (includes the path you mentioned)
net_candidates = [
    os.path.join(base, "data", "networks", "2x2", "2x2.net.xml"),
    os.path.join(base, "data", "networks", "2x2grid", "2x2.net.xml"),
    os.path.join(base, "nets", "2x2grid", "2x2.net.xml"),
    os.path.join(base, "nets", "2x2", "2x2.net.xml"),
]

net = next((p for p in net_candidates if os.path.isfile(p)), None)
if net is None:
    raise FileNotFoundError("Couldn't find 2x2 network. Searched:\n" + "\n".join(net_candidates))

# try to find a route file in the same directory as net, or fallback to package routes
net_dir = os.path.dirname(net)
rou = None
# pick first .rou.xml in same dir
for f in os.listdir(net_dir):
    if f.endswith(".rou.xml"):
        rou = os.path.join(net_dir, f)
        break

# fallback route locations
if rou is None:
    rou_candidates = [
        os.path.join(base, "data", "routes", "2x2", "2x2.rou.xml"),
        os.path.join(base, "data", "routes", "2x2grid", "2x2.rou.xml"),
    ]
    rou = next((p for p in rou_candidates if os.path.isfile(p)), None)

if rou is None:
    raise FileNotFoundError("Couldn't find matching .rou.xml route file. Looked near the net and in common package routes.")

print("Using network:", net)
print("Using route:", rou)

# Create env (headless)
env = SumoEnvironment(
    net_file=net,
    route_file=rou,
    single_agent=True,
    use_gui=False,
    out_csv_name="train_2x2_log"
)

# checkpoint callback
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./checkpoints", name_prefix="ppo_2x2")

try:
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=6000, callback=checkpoint_callback)
    model.save("ppo_2x2_final")
finally:
    # ensure env closes even on exception
    try:
        env.close()
    except Exception:
        pass

print("Training finished. Model saved as ppo_2x2_final.zip")
