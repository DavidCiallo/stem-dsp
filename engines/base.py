"""模块注册表。

每个引擎把自己实现的 Module 注册进来，HTTP 层只认模块名和参数字典，
不关心底下是 numpy、torch 还是外部服务。照 RX 的模块名命名，产品侧好对。
"""

REGISTRY = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def get(name):
    return REGISTRY.get(name)


def names():
    return list(REGISTRY)


def describe_all():
    return [cls.describe() for cls in REGISTRY.values()]


class Module:
    """一个处理模块。

    params 里每项：{"name", "type", "default", 可选的 min/max/choices}
    type 取 number / boolean / string。

    run(audio, sr, params) 返回 {"audio", "sr", "metrics", "subtype"}；
    audio 形状是 (frames, channels) 的 float64。
    """

    name = ""
    params = []
    needs_input = True

    @classmethod
    def describe(cls):
        return {"module": cls.name, "kind": "whole_file",
                "needsInput": cls.needs_input, "parameters": cls.params}

    @classmethod
    def defaults(cls):
        return {p["name"]: p["default"] for p in cls.params}

    @classmethod
    def resolve(cls, given):
        """按声明补齐参数，并做一次类型转换。"""
        out = cls.defaults()
        for p in cls.params:
            key = p["name"]
            if key not in given:
                continue
            raw = given[key]
            kind = p["type"]
            if kind == "number":
                out[key] = float(raw)
                if "min" in p:
                    out[key] = max(p["min"], out[key])
                if "max" in p:
                    out[key] = min(p["max"], out[key])
            elif kind == "boolean":
                out[key] = bool(raw)
            elif kind == "string":
                out[key] = str(raw)
                if "choices" in p and out[key] not in p["choices"]:
                    raise ValueError(f"{cls.name}.{key} 只能取 {p['choices']}，收到 {raw!r}")
        return out
