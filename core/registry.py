from typing import Dict, Type, Optional, List
from .store import AbstractVectorStore

class VectorStoreRegistry:
    _stores: Dict[str, Type[AbstractVectorStore]] = {}
    _display_names: Dict[str, str] = {}

    @classmethod
    def register(cls, name: str, display_name: str):
        def decorator(store_cls: Type[AbstractVectorStore]):
            cls._stores[name] = store_cls
            cls._display_names[name] = display_name
            return store_cls
        return decorator

    @classmethod
    def get_store_class(cls, name: str) -> Optional[Type[AbstractVectorStore]]:
        return cls._stores.get(name)

    @classmethod
    def get_all_names(cls) -> List[str]:
        return list(cls._stores.keys())

    @classmethod
    def get_display_name(cls, name: str) -> str:
        return cls._display_names.get(name, name)

    @classmethod
    def get_display_names_map(cls) -> Dict[str, str]:
        """Return a copy of the {key: display_name} mapping."""
        return dict(cls._display_names)

