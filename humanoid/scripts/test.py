import mujoco
import os
from humanoid import LEGGED_GYM_ROOT_DIR

def print_body_tree(model, body_id=0, indent=0):
    """
    递归打印MuJoCo模型的body树结构

    Args:
        model: mujoco.MjModel对象
        body_id: 当前body的id，0是root body
        indent: 缩进，用来显示层级
    """
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    print("  " * indent + f"Body id={body_id}, name='{body_name}'")

    # 遍历当前body的子body
    for child_id in get_child_bodies(model, body_id):
        print_body_tree(model, child_id, indent + 1)

def get_child_bodies(model, parent_id):
    """
    获取指定body的子body列表

    Args:
        model: mujoco.MjModel对象
        parent_id: 当前body的id

    Returns:
        子body id列表
    """
    child_bodies = []
    for i in range(model.nbody):
        if model.body_parentid[i] == parent_id:
            child_bodies.append(i)
    return child_bodies

if __name__ == "__main__":
    mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/ainex_description/mjcf/ainex.xml'
    xml_path = mujoco_model_path
    model = mujoco.MjModel.from_xml_path(xml_path)
    print_body_tree(model)
