import ast
from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]
CONTROLLER_PATH = (
    REPO_ROOT / "paradex" / "io" / "robot_controller" / "xarm_controller_ros.py"
)
CAPTURE_ROBOT_PATH = REPO_ROOT / "src" / "dataset_acquisition" / "hri" / "capture_robot.py"
CONTROLLER_SOURCE = CONTROLLER_PATH.read_text(encoding="utf-8")


def _function_node(module: ast.Module, class_name: str, function_name: str) -> ast.FunctionDef:
    class_node = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )


def _source_segment(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    assert segment is not None
    return segment


def _is_self_lock(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "lock"
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def test_record_loop_checks_save_state_and_appends_under_one_lock():
    module = ast.parse(CONTROLLER_SOURCE)
    record_loop = _function_node(module, "XArmControllerROS", "record_loop")
    lock_blocks = [
        node
        for node in ast.walk(record_loop)
        if isinstance(node, ast.With)
        and any(_is_self_lock(item.context_expr) for item in node.items)
    ]

    assert len(lock_blocks) == 1
    locked_source = _source_segment(CONTROLLER_SOURCE, lock_blocks[0])
    assert "self.save_event.is_set() and self.data is not None" in locked_source
    assert 'self.data["action"].append(action_homo.copy())' in locked_source


def test_start_rejects_a_dead_record_thread():
    module = ast.parse(CONTROLLER_SOURCE)
    start = _function_node(module, "XArmControllerROS", "start")

    assert "if not self.record_thread.is_alive()" in _source_segment(
        CONTROLLER_SOURCE, start
    )


def test_controller_can_stop_servo_output_without_ending_ros_lifecycle():
    module = ast.parse(CONTROLLER_SOURCE)
    stop_motion = _function_node(
        module, "XArmControllerROS", "stop_motion_commands"
    )
    source = _source_segment(CONTROLLER_SOURCE, stop_motion)

    assert "with self.lock" in source
    assert "self._has_motion_command = False" in source
    assert "self.exit_event.set()" not in source


def test_capture_session_ends_ros_hands_before_arms():
    capture_source = (
        REPO_ROOT / "paradex/dataset_acqusition/capture.py"
    ).read_text(encoding="utf-8")
    module = ast.parse(capture_source)
    end = _function_node(module, "CaptureSession", "end")
    source = _source_segment(capture_source, end)

    assert source.index("self.hand.end()") < source.index("self.arm.end()")
    assert source.index("self.hand_right.end()") < source.index(
        "self.arm_right.end()"
    )


def test_capture_preview_default_is_rolled_back_to_five_hz():
    module = ast.parse(CAPTURE_ROBOT_PATH.read_text(encoding="utf-8"))
    matching_calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--camera-preview-refresh-interval"
    ]

    assert len(matching_calls) == 1
    default = next(
        keyword.value
        for keyword in matching_calls[0].keywords
        if keyword.arg == "default"
    )
    assert ast.literal_eval(default) == 0.2
