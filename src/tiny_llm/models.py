from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2


def shortcut_name_to_full_name(shortcut_name: str, week: int | None = None):
    lower_shortcut_name = shortcut_name.lower()
    if week in {1, 2} and lower_shortcut_name == "qwen/qwen2-7b-instruct":
        return "Qwen/Qwen2-7B-Instruct-AWQ"
    elif week in {1, 2} and lower_shortcut_name == "qwen/qwen2-1.5b-instruct":
        return "Qwen/Qwen2-1.5B-Instruct-AWQ"
    elif week in {1, 2} and lower_shortcut_name == "qwen/qwen2-0.5b-instruct":
        return "Qwen/Qwen2-0.5B-Instruct-AWQ"
    if week in {1, 2} and lower_shortcut_name == "qwen2-7b":
        return "Qwen/Qwen2-7B-Instruct-AWQ"
    elif week in {1, 2} and lower_shortcut_name == "qwen2-1.5b":
        return "Qwen/Qwen2-1.5B-Instruct-AWQ"
    elif week in {1, 2} and lower_shortcut_name == "qwen2-0.5b":
        return "Qwen/Qwen2-0.5B-Instruct-AWQ"
    elif lower_shortcut_name == "qwen2-7b":
        return "Qwen/Qwen2-7B-Instruct"
    elif lower_shortcut_name == "qwen2-0.5b":
        return "Qwen/Qwen2-0.5B-Instruct"
    elif lower_shortcut_name == "qwen2-1.5b":
        return "Qwen/Qwen2-1.5B-Instruct"
    else:
        return shortcut_name


def dispatch_model(model_name: str, torch_model, week: int, **kwargs):
    model_name = shortcut_name_to_full_name(model_name, week=week)
    if week == 1 and model_name.startswith("Qwen/Qwen2"):
        return Qwen2ModelWeek1(torch_model, **kwargs)
    if week == 2 and model_name.startswith("Qwen/Qwen2"):
        return Qwen2ModelWeek2(torch_model, **kwargs)
    raise ValueError(f"{model_name} for week {week} not supported")
