# File: modules/knowledge.py

class Data:
    def __init__(self, id, content, source=None, tags=None):
        self.id = id
        self.content = content
        self.source = source
        self.tags = tags or []
        self.acquired_by = {}  # {agent_name: {"mode": "direct" or "shared", "from": "AgentX" or None}}

    def __repr__(self):
        return f"<Data {self.id}: {self.content}>"


class Information:
    def __init__(self, id, content, source=None, related_data=None, tags=None):
        self.id = id
        self.content = content
        self.source = source
        self.related_data = related_data or []
        self.tags = tags or []
        self.acquired_by = {}  # {agent_name: {"mode": "direct" or "shared", "from": "AgentX" or None}}

    def __repr__(self):
        return f"<Info {self.id}: {self.content}>"


class Knowledge:
    def __init__(self):
        self.rules = []  # List of rule strings
        self.built_from = {}  # {rule_str: [info_ids]}
        self.inferred_by = {}  # {rule_str: [agent_names]}

    def add_rule(self, rule_str, from_info_ids, inferred_by_agents=None):
        self.rules.append(rule_str)
        self.built_from[rule_str] = from_info_ids
        if inferred_by_agents:
            self.inferred_by[rule_str] = inferred_by_agents
        else:
            self.inferred_by[rule_str] = []

    def try_infer_rules(self, info_set, agent_name=None):
        tag_groups = {}
        for info in info_set:
            for tag in info.tags:
                tag_groups.setdefault(tag, []).append(info)

        for tag, group in tag_groups.items():
            if len(group) >= 2:
                rule_text = f"Inferred rule for [{tag}]: " + " + ".join(i.content for i in group)
                info_ids = [i.id for i in group]
                if rule_text not in self.rules:
                    inferred_by = [agent_name] if agent_name else []
                    self.add_rule(rule_text, info_ids, inferred_by)

    def __repr__(self):
        return f"<Knowledge with {len(self.rules)} rules>"


def init_dik_packets():
    def with_source(items, source):
        for item in items:
            item.source = source
        return items

    team_data = with_source([
        Data("D001", "Each water generator can support only 2 structures", tags=["Water", "Limit"]),
        Data("D002", "Water generator output is evenly split between connected structures", tags=["Water", "Distribution"]),
        Data("D003", "Each water generator provides 60 units of water", tags=["Water", "Output"]),
        Data("D004", "Colonists require both food and water", tags=["Colonists", "Needs"]),
        Data("D005", "Plants require water", tags=["Plants", "Needs"]),
        Data("D006", "Water generator must be capped with a uniform color", tags=["Water", "Construction"]),
    ], "Team_Info")

    architect_data = with_source([
        Data("D007", "Floor space determines house capacity", tags=["Housing", "Capacity"]),
        Data("D008", "VIPs require pink flooring", tags=["VIP", "Housing"]),
        Data("D009", "House structure can vary in size", tags=["Housing", "Flexibility"]),
        Data("D010", "Only completed houses support colonists", tags=["Housing", "Completion"]),
    ], "Architect_Info")

    engineer_data = with_source([
        Data("D011", "Water generators must be built on gray foundation", tags=["Water", "Foundation"]),
        Data("D012", "Water generators are 2x2 wide and 2 bricks high", tags=["Water", "Size"]),
        Data("D013", "Water bricks are translucent blue", tags=["Water", "Material"]),
    ], "Engineer_Info")

    botanist_data = with_source([
        Data("D014", "1 correctly constructed soil brick supports 5 civilians or 2 VIPs", tags=["Soil", "Support"]),
        Data("D015", "Plants need water to grow", tags=["Plants", "Water"]),
        Data("D016", "Greenhouses require water but not food", tags=["Greenhouse", "Input"]),
    ], "Botanist_Info")

    team_info = with_source([
        Information("I001", "Each water generator delivers 60 units of water, split evenly across 2 structures", tags=["Water", "Output", "Split"]),
        Information("I002", "Colonists need both food and water; plants only need water", tags=["Needs", "Colonists", "Plants"]),
    ], "Team_Info")

    architect_info = with_source([
        Information("I003", "A 4x5 pink floor house supports 10 VIPs", tags=["VIP", "Housing", "Capacity"]),
        Information("I004", "A 5x5 house supports 50 civilians if fully built", tags=["Housing", "Capacity"]),
    ], "Architect_Info")

    engineer_info = with_source([
        Information("I005", "A correctly built water generator has a gray foundation and blue translucent bricks stacked 2 bricks high (2x2)", tags=["Water", "Construction"]),
    ], "Engineer_Info")

    botanist_info = with_source([
        Information("I006", "One soil brick supports 5 civilians or 2 VIPs and must be placed beneath green bricks to grow plants", tags=["Soil", "Support", "Greenhouse"]),
    ], "Botanist_Info")

    return {
        "Team_Info": {
            "data": team_data,
            "information": team_info
        },
        "Architect_Info": {
            "data": architect_data,
            "information": architect_info
        },
        "Engineer_Info": {
            "data": engineer_data,
            "information": engineer_info
        },
        "Botanist_Info": {
            "data": botanist_data,
            "information": botanist_info
        }
    }
