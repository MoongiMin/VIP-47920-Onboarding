import time
from pathlib import Path
import mujoco as mj
from mujoco import viewer
import numpy as np


def view_dog():
    """
    Load floor2.xml (includes dog) and visualize the dog box.
    """
    model = mj.MjModel.from_xml_path("xml/floor2.xml")
    data = mj.MjData(model)

    dt = model.opt.timestep
    viewer_handle = viewer.launch_passive(model, data)

    while viewer_handle.is_running():
        mj.mj_step(model, data)
        time.sleep(dt)
        viewer_handle.sync()


if __name__ == "__main__":
    view_dog()
