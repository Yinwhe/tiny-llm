from .qwen2_week1 import Qwen2ModelWeek1


def shortcut_name_to_full_name(shortcut_name: str):
    lower_shortcut_name = shortcut_name.lower()
    if lower_shortcut_name == "qwen2-7b":
        return "Qwen/Qwen2-7B-Instruct"
    elif lower_shortcut_name == "qwen2-0.5b":
        return "Qwen/Qwen2-0.5B-Instruct"
    elif lower_shortcut_name == "qwen2-1.5b":
        return "Qwen/Qwen2-1.5B-Instruct"
    else:
        return shortcut_name


def dispatch_model(model_name: str, torch_model, week: int, **kwargs):
    model_name = shortcut_name_to_full_name(model_name)
    if week == 1 and model_name.startswith("Qwen/Qwen2"):
        return Qwen2ModelWeek1(torch_model, **kwargs)
    raise ValueError(f"{model_name} for week {week} not supported")
