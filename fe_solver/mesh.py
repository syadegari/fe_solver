from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .shape import HEX20_PARENT_NODES, HEX8_PARENT_NODES
from .types import ModelError


@dataclass(frozen=True)
class MeshElement:
    tag: int
    topology: str
    connectivity: np.ndarray


@dataclass(frozen=True)
class PeriodicMap:
    slave_region: str
    master_entity: int
    slave_nodes: np.ndarray
    master_nodes: np.ndarray
    translation: np.ndarray


@dataclass
class Mesh:
    X: np.ndarray
    node_tags: np.ndarray
    tag_to_index: dict[int, int]
    elements: dict[int, MeshElement]
    volume_groups: dict[str, np.ndarray]
    node_groups: dict[str, np.ndarray]
    physical_dimensions: dict[str, int]
    periodic_maps: dict[str, list[PeriodicMap]]

    @property
    def ndof(self) -> int:
        return 3 * len(self.X)


def _validate_local_order(gmsh, element_type: int, topology: str) -> None:
    props = gmsh.model.mesh.getElementProperties(element_type)
    local = np.asarray(props[4], dtype=float).reshape(-1, 3)
    expected = HEX8_PARENT_NODES if topology == "hex8" else HEX20_PARENT_NODES
    if local.shape != expected.shape or not np.allclose(local, expected, atol=1.0e-14):
        raise ModelError(f"Gmsh {topology} local ordering does not match the element kernel")


def read_gmsh(path: str | Path) -> Mesh:
    try:
        import gmsh
    except ImportError as exc:
        raise ModelError("the official gmsh Python package is required") from exc
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(Path(path)))
        tags_raw, coords_raw, _ = gmsh.model.mesh.getNodes()
        node_tags = np.asarray(tags_raw, dtype=np.int64)
        order = np.argsort(node_tags)
        node_tags = node_tags[order]
        X = np.asarray(coords_raw, dtype=float).reshape(-1, 3)[order]
        tag_to_index = {int(tag): i for i, tag in enumerate(node_tags)}

        physical_dimensions: dict[str, int] = {}
        node_groups: dict[str, np.ndarray] = {}
        volume_groups: dict[str, np.ndarray] = {}
        physical_entities: dict[str, np.ndarray] = {}
        for dim_raw, ptag_raw in gmsh.model.getPhysicalGroups():
            dim, ptag = int(dim_raw), int(ptag_raw)
            name = gmsh.model.getPhysicalName(dim, ptag)
            if not name:
                raise ModelError(f"unnamed Physical Group ({dim}, {ptag})")
            if name in physical_dimensions:
                raise ModelError(f"duplicate Physical Group name {name!r}")
            physical_dimensions[name] = dim
            physical_entities[name] = np.asarray(
                gmsh.model.getEntitiesForPhysicalGroup(dim, ptag), dtype=np.int64
            )
            group_tags = np.asarray(gmsh.model.mesh.getNodesForPhysicalGroup(dim, ptag)[0], dtype=np.int64)
            node_groups[name] = np.asarray([tag_to_index[int(t)] for t in np.unique(group_tags)], dtype=np.int64)

        elements: dict[int, MeshElement] = {}
        entity_element_tags: dict[tuple[int, int], list[int]] = {}
        type_map = {5: ("hex8", 8), 17: ("hex20", 20)}
        for _, entity_raw in gmsh.model.getEntities(3):
            entity = int(entity_raw)
            types, tag_blocks, conn_blocks = gmsh.model.mesh.getElements(3, entity)
            collected: list[int] = []
            for etype_raw, etags_raw, conn_raw in zip(types, tag_blocks, conn_blocks):
                etype = int(etype_raw)
                if etype not in type_map:
                    raise ModelError(f"unsupported 3D Gmsh element type {etype}")
                topology, nnode = type_map[etype]
                _validate_local_order(gmsh, etype, topology)
                etags = np.asarray(etags_raw, dtype=np.int64)
                conns = np.asarray(conn_raw, dtype=np.int64).reshape(-1, nnode)
                for tag, conn_tags in zip(etags, conns):
                    tag_int = int(tag)
                    conn = np.asarray([tag_to_index[int(x)] for x in conn_tags], dtype=np.int64)
                    elements[tag_int] = MeshElement(tag_int, topology, conn)
                    collected.append(tag_int)
            entity_element_tags[(3, entity)] = collected
        for name, dim in physical_dimensions.items():
            if dim == 3:
                tags: list[int] = []
                for entity in physical_entities[name]:
                    tags.extend(entity_element_tags.get((3, int(entity)), []))
                volume_groups[name] = np.asarray(tags, dtype=np.int64)

        periodic_maps: dict[str, list[PeriodicMap]] = {}
        for name, dim in physical_dimensions.items():
            if dim != 2:
                continue
            maps: list[PeriodicMap] = []
            for entity in physical_entities[name]:
                master, slave_tags, master_tags, affine_raw = gmsh.model.mesh.getPeriodicNodes(2, int(entity), False)
                if int(master) == 0 or len(slave_tags) == 0:
                    continue
                affine = np.asarray(affine_raw, dtype=float).reshape(4, 4)
                if not np.allclose(affine[:3, :3], np.eye(3), atol=1.0e-10) or not np.allclose(
                    affine[3], [0, 0, 0, 1], atol=1.0e-10
                ):
                    raise ModelError(f"periodic region {name!r} has a non-translational map")
                slave = np.asarray([tag_to_index[int(t)] for t in slave_tags], dtype=np.int64)
                master_nodes = np.asarray([tag_to_index[int(t)] for t in master_tags], dtype=np.int64)
                translation = affine[:3, 3].copy()
                if not np.allclose(X[slave] - X[master_nodes], translation, atol=1.0e-8):
                    raise ModelError(f"periodic region {name!r} has inconsistent node correspondence")
                maps.append(PeriodicMap(name, int(master), slave, master_nodes, translation))
            if maps:
                periodic_maps[name] = maps
        if not elements:
            raise ModelError("mesh contains no supported solid elements")
        return Mesh(X, node_tags, tag_to_index, elements, volume_groups, node_groups, physical_dimensions, periodic_maps)
    finally:
        gmsh.finalize()
