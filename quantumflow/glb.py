"""Minimal glTF 2.0 binary writer for exhibition geometry.

The grove hangs a ``model`` exhibit as a single ``.glb`` on a plinth and reads
it with three's ``GLTFLoader``. Nothing here needs a scene graph, skinning,
textures or animation: an exhibition model is a handful of coloured triangle
soups. Writing those directly keeps the export dependency-free and keeps the
byte count under the bundler's budget, where a general-purpose exporter would
spend it on features the grove never reads.

Vertex colour is the load-bearing feature. Every surface this module writes
carries a scalar physical quantity in ``COLOR_0`` -- the sign of a
wavefunction, a local kinetic-energy density, the gap between two surfaces --
so the geometry shows a measurement rather than decorating one.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# glTF component types (glTF 2.0, table 3.2).
_FLOAT = 5126
_UNSIGNED_INT = 5125
_UNSIGNED_BYTE = 5121

# glTF buffer view targets.
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963

_GLB_MAGIC = b"glTF"
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942

#: Every chunk and buffer view is padded to this alignment (glTF 2.0, 4.4.2).
_ALIGNMENT = 4


@dataclass
class Surface:
    """One triangle mesh with a per-vertex colour and a single material.

    ``positions`` is ``(vertices, 3)`` in metres, ``indices`` is
    ``(triangles, 3)``. ``colors`` is ``(vertices, 4)`` RGBA in ``[0, 1]``;
    when absent the surface takes ``base_color`` uniformly. ``normals`` are
    computed from the triangles when not supplied.
    """

    positions: np.ndarray
    indices: np.ndarray
    colors: np.ndarray | None = None
    normals: np.ndarray | None = None
    name: str = "surface"
    base_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    metallic: float = 0.0
    roughness: float = 0.65
    #: Emissive strength applied as a fraction of the vertex colour. The grove
    #: lights a plinth dimly; a wavefunction that emits reads as made of light.
    emissive: float = 0.0
    double_sided: bool = True

    def __post_init__(self) -> None:
        self.positions = np.ascontiguousarray(self.positions, dtype=np.float32)
        self.indices = np.ascontiguousarray(self.indices, dtype=np.uint32)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(f"{self.name}: positions must have shape (vertices, 3)")
        if self.indices.ndim != 2 or self.indices.shape[1] != 3:
            raise ValueError(f"{self.name}: indices must have shape (triangles, 3)")
        if self.positions.shape[0] == 0 or self.indices.shape[0] == 0:
            raise ValueError(f"{self.name}: surface is empty")
        if int(self.indices.max(initial=0)) >= self.positions.shape[0]:
            raise ValueError(f"{self.name}: an index addresses a vertex that does not exist")
        if not np.isfinite(self.positions).all():
            raise ValueError(f"{self.name}: positions contain a non-finite value")

        if self.colors is not None:
            colors = np.ascontiguousarray(self.colors, dtype=np.float32)
            if colors.shape == (self.positions.shape[0], 3):
                opaque = np.ones((colors.shape[0], 1), dtype=np.float32)
                colors = np.concatenate([colors, opaque], axis=1)
            if colors.shape != (self.positions.shape[0], 4):
                raise ValueError(f"{self.name}: colors must have shape (vertices, 3 or 4)")
            if not np.isfinite(colors).all():
                raise ValueError(f"{self.name}: colors contain a non-finite value")
            self.colors = np.clip(colors, 0.0, 1.0)

        if self.normals is None:
            self.normals = vertex_normals(self.positions, self.indices)
        else:
            normals = np.ascontiguousarray(self.normals, dtype=np.float32)
            if normals.shape != self.positions.shape:
                raise ValueError(f"{self.name}: normals must match positions")
            self.normals = normals

    @property
    def triangles(self) -> int:
        return int(self.indices.shape[0])

    @property
    def is_translucent(self) -> bool:
        return self.base_color[3] < 1.0 or (
            self.colors is not None and float(self.colors[:, 3].min()) < 1.0
        )


@dataclass
class Model:
    """A named collection of surfaces written as one ``.glb``."""

    name: str = "model"
    surfaces: list[Surface] = field(default_factory=list)

    def add(self, surface: Surface) -> Surface:
        self.surfaces.append(surface)
        return surface

    @property
    def triangles(self) -> int:
        return sum(surface.triangles for surface in self.surfaces)


def vertex_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Return area-weighted smooth vertex normals for a triangle mesh."""
    positions = np.asarray(positions, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)

    corners = positions[indices]
    face = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])

    normals = np.zeros_like(positions)
    # Accumulating the unnormalized face normal weights each face by its own
    # area, which is what makes a marching-cubes surface look smooth.
    for corner in range(3):
        np.add.at(normals, indices[:, corner], face)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    # A vertex whose faces cancel exactly has no defined normal; point it up
    # rather than dividing by zero.
    degenerate = lengths[:, 0] <= 0.0
    normals[degenerate] = (0.0, 1.0, 0.0)
    lengths[degenerate] = 1.0
    return np.ascontiguousarray(normals / lengths, dtype=np.float32)


def _pad(buffer: bytearray, fill: int = 0) -> None:
    while len(buffer) % _ALIGNMENT:
        buffer.append(fill)


def write_glb(path: Path | str, model: Model) -> Path:
    """Write ``model`` as a binary glTF file and return the path.

    The result uses only core glTF 2.0: no extension is declared, so the
    grove's loader never needs a decoder it does not host.
    """
    if not model.surfaces:
        raise ValueError(f"{model.name}: a model needs at least one surface")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    binary = bytearray()
    buffer_views: list[dict] = []
    accessors: list[dict] = []
    materials: list[dict] = []
    meshes: list[dict] = []
    nodes: list[dict] = []

    def add_view(data: bytes, target: int | None) -> int:
        _pad(binary)
        view = {"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        binary.extend(data)
        buffer_views.append(view)
        return len(buffer_views) - 1

    def add_accessor(accessor: dict) -> int:
        accessors.append(accessor)
        return len(accessors) - 1

    for surface in model.surfaces:
        position_view = add_view(surface.positions.tobytes(), _ARRAY_BUFFER)
        position = add_accessor(
            {
                "bufferView": position_view,
                "componentType": _FLOAT,
                "count": int(surface.positions.shape[0]),
                "type": "VEC3",
                # The spec requires bounds on POSITION, and the bundler reads
                # them to place the model before the glb has downloaded.
                "min": [float(v) for v in surface.positions.min(axis=0)],
                "max": [float(v) for v in surface.positions.max(axis=0)],
            }
        )

        normal_view = add_view(surface.normals.tobytes(), _ARRAY_BUFFER)
        normal = add_accessor(
            {
                "bufferView": normal_view,
                "componentType": _FLOAT,
                "count": int(surface.normals.shape[0]),
                "type": "VEC3",
            }
        )

        attributes = {"POSITION": position, "NORMAL": normal}

        if surface.colors is not None:
            # Normalized unsigned bytes: a quarter of the bytes of float
            # colours, and 8 bits per channel is past what the eye resolves
            # on a lit surface.
            quantized = np.round(surface.colors * 255.0).astype(np.uint8)
            color_view = add_view(np.ascontiguousarray(quantized).tobytes(), _ARRAY_BUFFER)
            attributes["COLOR_0"] = add_accessor(
                {
                    "bufferView": color_view,
                    "componentType": _UNSIGNED_BYTE,
                    "normalized": True,
                    "count": int(quantized.shape[0]),
                    "type": "VEC4",
                }
            )

        index_view = add_view(surface.indices.tobytes(), _ELEMENT_ARRAY_BUFFER)
        index_accessor = add_accessor(
            {
                "bufferView": index_view,
                "componentType": _UNSIGNED_INT,
                "count": int(surface.indices.size),
                "type": "SCALAR",
            }
        )

        material: dict = {
            "name": f"{surface.name}-material",
            "pbrMetallicRoughness": {
                "baseColorFactor": list(surface.base_color),
                "metallicFactor": float(surface.metallic),
                "roughnessFactor": float(surface.roughness),
            },
            "doubleSided": bool(surface.double_sided),
        }
        if surface.is_translucent:
            material["alphaMode"] = "BLEND"
        if surface.emissive > 0.0:
            strength = float(np.clip(surface.emissive, 0.0, 1.0))
            material["emissiveFactor"] = [strength, strength, strength]
        materials.append(material)

        meshes.append(
            {
                "name": surface.name,
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": index_accessor,
                        "material": len(materials) - 1,
                        "mode": 4,
                    }
                ],
            }
        )
        nodes.append({"mesh": len(meshes) - 1, "name": surface.name})

    _pad(binary)

    gltf = {
        "asset": {"version": "2.0", "generator": "quantumflow.glb"},
        "scene": 0,
        "scenes": [{"name": model.name, "nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(binary)}],
    }

    json_chunk = bytearray(json.dumps(gltf, separators=(",", ":")).encode("utf-8"))
    _pad(json_chunk, fill=0x20)

    total = 12 + 8 + len(json_chunk) + 8 + len(binary)
    with open(path, "wb") as handle:
        handle.write(_GLB_MAGIC)
        handle.write(struct.pack("<II", 2, total))
        handle.write(struct.pack("<II", len(json_chunk), _CHUNK_JSON))
        handle.write(json_chunk)
        handle.write(struct.pack("<II", len(binary), _CHUNK_BIN))
        handle.write(binary)
    return path
