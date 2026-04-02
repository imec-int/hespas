# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT

SI_PREFIXES = [
    ("Q", 1e30),
    ("R", 1e27),
    ("Y", 1e24),
    ("Z", 1e21),
    ("E", 1e18),
    ("P", 1e15),
    ("T", 1e12),
    ("G", 1e9),
    ("M", 1e6),
    ("k", 1e3),
    ("", 1),
    ("m", 1e-3),
    ("u", 1e-6),
    ("n", 1e-9),
    ("p", 1e-12),
    ("f", 1e-15),
    ("a", 1e-18),
    ("z", 1e-21),
    ("y", 1e-24),
    ("r", 1e-27),
    ("q", 1e-30)
]

SI_PREFIXES_MAP = {k: v for k, v in SI_PREFIXES}

class Statistic:
    def __init__(self, print_name, unit=None, prefix="", value_type=float, value_mult=None, rounding_decimals=3, default_sampled=False, normalise_unit=False):
        self.print_name = print_name
        self.unit = unit
        self.prefix = prefix
        self.value_type = value_type
        self.value_mult = (1 / value_mult) if value_mult is not None else SI_PREFIXES_MAP[prefix] if prefix is not None else 1
        self.rounding_decimals = rounding_decimals
        self.default_sampled = default_sampled
        self.sampled = default_sampled
        self.normalise_unit = normalise_unit

    def calculate(self):
        raise NotImplementedError()

    def sample(self, *args, **kwargs):
        self.sampled = True

    def reset(self):
        self.sampled = self.default_sampled

    def print(self):
        value = self.print_value()
        return "{}:{}{}".format(self.print_name, "" if value[0] == '\n' else " ", value)

    def __str__(self):
        return self.print()

    def __float__(self):
        return float(self.calculate())

    def __int__(self):
        return int(self.calculate())

    def to_dict(self):
        return {
            "type": self.__class__.__name__,
            "print_name": self.print_name,
            "unit": self.unit,
            "value_type": self.value_type.__name__,
            "value_mult": self.value_mult,
            "rounding_decimals": self.rounding_decimals,
            "default_sampled": self.default_sampled
        }

    def other_to_dict(self, other):
        if other is None or isinstance(other, (int, str, float, bool)):
            return other
        elif isinstance(other, list):
            return [self.other_to_dict(x) for x in other]
        elif isinstance(other, dict):
            return {self.other_to_dict(k): self.other_to_dict(v) for k, v in other.items()}
        elif hasattr(other, "to_dict"):
            return other.to_dict()
        raise ValueError("Cannot convert '{}' to dict")

    def round_or_int_value(self, value=None, value_mult=None):
        if value is None:
            value = self.calculate()
        if value is None:
            return None
        if value_mult is None:
            value_mult = self.value_mult
        if issubclass(self.value_type, float):
            return round(self.value_type(value * value_mult), self.rounding_decimals)
        else:
            return self.value_type(value * value_mult)

    def unit_print(self):
        return str(self.unit) if self.unit is not None else ""

    def print_value(self):
        value = self.calculate()
        unit_str = ""
        mul = self.value_mult
        if self.unit:
            new_value = value / mul
            unit_prefix = self.prefix
            if self.normalise_unit:
                current_unit_index = SI_PREFIXES.index((unit_prefix, mul))
                abs_value = abs(new_value)
                while current_unit_index < len(SI_PREFIXES)-1 and current_unit_index > 0 and (abs_value >= 1000 or abs_value <= 1e-3):
                    if abs_value >= 1000:
                        current_unit_index -= 1
                    elif abs_value <= 1e-3:
                        current_unit_index += 1
                    unit_prefix, mul = SI_PREFIXES[current_unit_index]
                    new_value = value / mul
                    abs_value = abs(new_value)
            unit_str = " " + unit_prefix + self.unit_print()

        return "{}{}".format(self.round_or_int_value(value, 1 / mul), unit_str)

    @staticmethod
    def unwrap_val(val):
        if isinstance(val, (int, float)):
            return val
        elif isinstance(val, Statistic):
            return val.calculate()
        else:
            raise TypeError("Unknown type {} for val".format(type(val).__name__))

    def print_val(self, val):
        if isinstance(val, (int, float)):
            return self.round_or_int_value(value=val)
        elif isinstance(val, Statistic):
            return val.print_value()

    def get_copy_args(self):
        return {k: v for k, v in self.to_dict() if k not in ("type", "print_name")}

    def copy(self):
        return self.__class__(self.print_name, **self.get_copy_args())

    def to_stats_out(self):
        return self.calculate()

    @classmethod
    def other_from_dict(cls, other):
        if other is None or isinstance(other, (int, str, float, bool)):
            return other
        elif isinstance(other, list):
            return [cls.other_from_dict(x) for x in other]
        elif isinstance(other, dict):
            return Statistic.from_dict(other)
        raise ValueError("Unknown from_dict item '{}'".format(other))

    @classmethod
    def get_args_from_dict(cls, d):
        return {
            "print_name": cls.other_from_dict(d["print_name"]),
            "unit": cls.other_from_dict(d["unit"]),
            "value_type": globals().get(d["value_type"], globals()["__builtins__"][d["value_type"]]), # Hmmm... think about this one
            "value_mult": cls.other_from_dict(d["value_mult"]),
            "rounding_decimals": cls.other_from_dict(d["rounding_decimals"]),
            "default_sampled": cls.other_from_dict(d["default_sampled"])
        }

    @classmethod
    def get_subclasses(cls):
        ret_dict = {}
        for subclass in cls.__subclasses__():
            name = subclass.__name__
            ret_dict[name] = subclass
            ret_dict = {**ret_dict, **subclass.get_subclasses()}
        return ret_dict

    @classmethod
    def from_dict(cls, d):
        if d["type"] != cls.__name__:
            subclass = cls.get_subclasses().get(d["type"])
            if subclass is None:
                raise ValueError("No known {} subclass '{}'".format(cls.__name__, d["type"]))
            return subclass.from_dict(d)
        return cls(**cls.get_args_from_dict(d))

class IntervalStatistic(Statistic):
    def __init__(self, print_name, start=None, end=None, **kwargs):
        super().__init__(print_name, **kwargs)
        self.start = start
        self.end = end
        self.sample(self.start, self.end)

    def sample(self, start=None, end=None, reset=False):
        if start is not None or end is not None:
            self.sampled = True
        self.start = self.value_type(start) if start is not None or reset else self.start
        self.end = self.value_type(end) if end is not None or reset else self.end

    def reset(self):
        super().reset()
        self.start = None
        self.end = None

    def calculate(self):
        if self.start is None or self.end is None:
            return None
        return self.value_type(Statistic.unwrap_val(self.end) - Statistic.unwrap_val(self.start))

    def to_dict(self):
        return {
            **super().to_dict(),
            "start": self.other_to_dict(self.start),
            "end": self.other_to_dict(self.end)
        }

    @classmethod
    def get_args_from_dict(cls, d):
        return {
            **cls.__base__.get_args_from_dict(d),
            "start": cls.other_from_dict(d["start"]),
            "end": cls.other_from_dict(d["end"])
        }

class RatioStatistic(Statistic):
    def __init__(self, print_name, nom=None, denom=None, show_division=True, sub_from_one=False, **kwargs):
        super().__init__(print_name, **kwargs)
        self.nom = nom
        self.denom = denom
        self.show_division = show_division
        self.sub_from_one = sub_from_one
        self.sample(self.nom, self.denom)

    def sample(self, nom=None, denom=None, reset=False):
        if nom is not None or denom is not None:
            self.sampled = True
        self.nom = nom if nom is not None or reset else self.nom
        self.denom = denom if denom is not None or reset else self.denom

    def reset(self):
        super().reset()
        self.nom = None
        self.denom = None

    def calculate(self):
        if self.nom is None or self.denom is None:
            return None
        denom = Statistic.unwrap_val(self.denom)
        if float(denom) == 0:
            return self.value_type(0)
        nom = Statistic.unwrap_val(self.nom)
        val = nom / denom
        if self.sub_from_one:
            val = 1 - val
        return self.value_type(val)

    def to_dict(self):
        return {
            **super().to_dict(),
            "nom": self.other_to_dict(self.nom),
            "denom": self.other_to_dict(self.denom),
            "show_division": self.other_to_dict(self.show_division),
            "sub_from_one": self.other_to_dict(self.sub_from_one)
        }

    def print_value(self):
        return "{}{}".format(super().print_value(), " ({}{}/{})".format("1 - " if self.sub_from_one else "", self.print_val(self.nom), self.print_val(self.denom)) if self.show_division is True else "")

    @classmethod
    def get_args_from_dict(cls, d):
        return {
            **cls.__base__.get_args_from_dict(d),
            "nom": cls.other_from_dict(d["nom"]),
            "denom": cls.other_from_dict(d["denom"]),
            "show_division": cls.other_from_dict(d["show_division"]),
            "sub_from_one": cls.other_from_dict(d["sub_from_one"])
        }

class AggregateStatistic(Statistic):
    def __init__(self, print_name, samples=None, **kwargs):
        self.samples = []
        super().__init__(print_name, **kwargs)
        if samples is not None:
            self.sampled = True
            self.samples = samples

    def reset(self):
        super().reset()
        self.samples = []

    def calculate(self):
        return self.value_type(sum([Statistic.unwrap_val(x) for x in self.samples if x is not None]))

    def to_dict(self):
        return {
            **super().to_dict(),
            "samples": self.other_to_dict(self.samples)
        }

    def sample(self, sample):
        self.sampled = True
        if self.unit is None and hasattr(sample, "unit"):
            self.unit = sample.unit
        self.samples.append(sample)

    @classmethod
    def get_args_from_dict(cls, d):
        return {
            **cls.__base__.get_args_from_dict(d),
            "samples": cls.other_from_dict(d["samples"])
        }

class ValueStatistic(Statistic):
    def __init__(self, print_name, value=None, **kwargs):
        super().__init__(print_name, **kwargs)
        self.value = 0
        if value is not None:
            self.sample(value)

    def reset(self):
        super().reset()
        self.value = 0

    def calculate(self):
        return self.value

    def to_dict(self):
        return {
            **super().to_dict(),
            "value": self.value
        }

    def sample(self, sample):
        self.sampled = True
        self.value = self.value_type(sample)

    @classmethod
    def get_args_from_dict(cls, d):
        return {
            **cls.__base__.get_args_from_dict(d),
            "value": cls.other_from_dict(d["value"])
        }

class SummingStatistic(ValueStatistic):
    def sample(self, sample):
        super().sample(self.value + self.value_type(sample))

class ContainerStatistic(Statistic):
    def __init__(self, print_name, sort=False, members=None, justify=True, print_string=None, **kwargs):
        if members is None:
            members = dict()
        self.members = members
        self.sort = sort
        self.justify = justify
        self.print_string = print_string
        super().__init__(print_name, **kwargs)

    def sample(self, member_name, *args, **kwargs):
        self.sampled = True
        self.members[member_name].sample(*args, **kwargs)

    def add_member(self, member_name, new_member, check_exists=False):
        if check_exists is True:
            if member_name in self.members:
                return
        self.members[member_name] = new_member

    def add_members(self, member_dict=None, /, **members):
        if member_dict is not None:
            for member_name, member in member_dict.items():
                self.add_member(member_name, member)
        for member_name, member in members.items():
            self.add_member(member_name, member)

    def has_member(self, member_name):
        return member_name in self.members

    def get_member(self, member_name, *submembers):
        ret_member = self.members[member_name]
        for submember in submembers:
            ret_member = ret_member.get_member(submember)
        return ret_member

    def calculate(self, member_name=None, *args):
        if member_name is None:
            return {k: v.calculate() for k, v in self.members.items()}
        else:
            return self.members[member_name].calculate(*args)

    def to_dict(self, allow_filter=None, disallow_filter=None):
        if allow_filter is None:
            allow_filter = self.members.keys()
        if disallow_filter is None:
            disallow_filter = list()

        return {
            **super().to_dict(),
            "members": self.other_to_dict({k: v for k, v in self.members.items() if k in allow_filter and k not in disallow_filter}),
            "sort": self.other_to_dict(self.sort),
            "justify": self.other_to_dict(self.justify),
            "print_string": self.other_to_dict(self.print_string)
        }

    def items(self):
        return self.members.items()

    def values(self):
        return self.members.values()

    def keys(self):
        return self.members.keys()

    def print_value(self):
        if self.print_string is not None:
            return self.print_string.format(**{k: v.print_value() for k, v in self.items()})

        values = [(k, [x for x in v.print_value().splitlines() if len(x) > 0], v.calculate()) for k, v in self.items()]
        if self.sort:
            value_values = []
            other_values = []
            for value in values:
                if isinstance(value[2], (int, float)):
                    value_values.append(value)
                else:
                    other_values.append(value)
            value_values = sorted(value_values, key=lambda x: x[2], reverse=True)
            other_values = sorted(other_values, key=lambda x: x[0])
            values = value_values + other_values

        pad = 1
        indent = "    "
        max_len = None
        if self.justify and len(values) > 0:
            max_len = max([len(x[0]) for x in values])
        lines = []
        for key, ele_lines, _ in values:
            justify_padding = 0 if max_len is None else max_len - len(key)
            padding = " " * (pad + justify_padding)
            key_str = "{}{}:{}".format(indent, key, padding)
            if len(ele_lines) == 1:
                lines.append("{}{}".format(key_str, ele_lines[0]))
            else:
                lines.append(key_str)
                for ele_line in ele_lines:
                    lines.append("{}{}".format(indent, ele_line))

        return "\n{}".format("\n".join([x.rstrip() for x in lines]))

    def to_stats_out(self, allow_filter=None, disallow_filter=None):
        if allow_filter is None:
            allow_filter = self.members.keys()
        if disallow_filter is None:
            disallow_filter = list()

        return {k: v.to_stats_out() for k, v in self.items() if k in allow_filter and k not in disallow_filter}

    @classmethod
    def get_args_from_dict(cls, d):
        return {
            **cls.__base__.get_args_from_dict(d),
            "members": {cls.other_from_dict(k): cls.other_from_dict(v) for k,v in d["members"].items()},
            "sort": cls.other_from_dict(d["sort"]),
            "justify": cls.other_from_dict(d["justify"]),
            "print_string": cls.other_from_dict(d["print_string"])
        }
