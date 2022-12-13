from typing import Callable, Dict, Iterable, List, Optional

from hwinarion.dispatcher import ActResult, BaseAction

ActionFunctionType = Callable[[str], ActResult]


class StringLiteralActor(BaseAction):
    def __init__(self, name: str, string_action_mapper: Optional[Dict[str, ActionFunctionType]] = None):
        super().__init__(name)
        self.string_action_mapper = string_action_mapper or {}  # type: Dict[str, ActionFunctionType]

    def transform_trigger(self, trigger: str) -> str:
        return trigger

    def add_action(self, trigger: str, action_fn: ActionFunctionType) -> None:
        trigger = self.transform_trigger(trigger)
        self.string_action_mapper[trigger] = action_fn

    def get_action(self, trigger: str) -> ActionFunctionType:
        trigger = self.transform_trigger(trigger)
        return self.string_action_mapper[trigger]

    def del_action(self, trigger: str) -> None:
        trigger = self.transform_trigger(trigger)
        del self.string_action_mapper[trigger]

    def triggers(self) -> Iterable[str]:
        return self.string_action_mapper.keys()

    keys = triggers

    def items(self):
        return self.string_action_mapper.items()

    @property
    def recognized_words(self) -> List[str]:
        words = set()
        words.update(*(trigger.split() for trigger in self.triggers()))
        return list(words)

    def act(self, text: str) -> ActResult:
        text = self.transform_trigger(text)
        action = self.string_action_mapper.get(text)
        if action is not None:
            return action(text)
        return ActResult.TEXT_NOT_PROCESSED
