"""
Mock for schema.py (Alexandria protocol-side types).

Import and call setup() BEFORE importing spl or spl_gateway in any test.

Usage:
    import tests._mock_schema as mock_schema
    mock_schema.setup()

    from spl import SemanticUnit, ...
    from spl_gateway import SPLGateway, ...
"""

import sys
import types


def setup() -> None:
    """Inject mock schema into sys.modules so spl.py can do `from .schema import ...`"""
    if "_spl_pkg.schema" in sys.modules:
        return  # Already set up

    _mock = types.ModuleType("_spl_pkg_schema_mock")

    class _Category:
        EMPIRICAL  = "EMPIRICAL"
        MODEL      = "MODEL"
        NORMATIVE  = "NORMATIVE"
        def __getitem__(self, item): return item

    class _Modality:
        HYPOTHESIS = "hypothesis"
        def __init__(self, v): self.value = v

    class _BuilderOrigin:
        ALPHA = "alpha"
        BETA  = "beta"

    class _ClaimNode:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
        @classmethod
        def new(cls, subject, predicate, object, category, assumptions, source_refs):
            return cls(
                subject=subject,
                predicate=predicate,
                object=object,
                category=category,
                assumptions=assumptions,
                source_refs=source_refs,
                modality=None,
                builder_origin=None,
                timestamp=None,
            )
        def __repr__(self):
            return (f"ClaimNode({self.subject!r} --[{self.predicate}]--> "
                    f"{self.object!r})")

    _mock.ClaimNode     = _ClaimNode
    _mock.Category      = _Category()
    _mock.Modality      = _Modality
    _mock.BuilderOrigin = _BuilderOrigin()
    _mock.EpistemicStatus = type("EpistemicStatus", (), {"UNVALIDATED": "unvalidated"})()

    import spl as _spl_module
    _spl_module.__package__ = "_spl_pkg"
    sys.modules["_spl_pkg"]            = types.ModuleType("_spl_pkg")
    sys.modules["_spl_pkg.schema"]     = _mock   # for spl.py relative imports
    sys.modules["spl_gateway.schema"]  = _mock   # for spl_gateway._converter relative imports
