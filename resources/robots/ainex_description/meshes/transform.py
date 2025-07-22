from urdf2mjcf.convert import convert_urdf_to_mjcf
import mujoco

# 增加 constraint 上限
mujoco.set_default_max_con(maxcon=1000, maxefc=128)

convert_urdf_to_mjcf("ainex.urdf", output_dir="mjcf_output")
