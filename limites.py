"""Límites administrativos de un país (de OpenStreetMap) para que la app
sepa sin conexión en qué municipio, provincia y comunidad está un punto.

  python3 limites.py límites.geojsonseq [más.geojsonseq…] salida.bin.gz
  python3 limites.py --paises ne_admin0.geojson paises.bin.gz

Formato (todo comprimido con gzip, enteros en little endian):
  'OTL1', u32 número de zonas, y por cada zona:
    u8 nivel (admin_level de OSM; 2 = país en paises.bin)
    u16 + utf-8 nombres: «nombre\\tes=Nombre\\ten=Name…» (solo los que cambian)
    4 × i32 caja (lon y lat mínimas y máximas, en diezmilésimas de grado)
    u32 bytes de la geometría, y la geometría: número de anillos y, por
    anillo, número de puntos y los puntos (lon, lat en diezmilésimas,
    cada uno como diferencia con el anterior, en varint zigzag).
Un punto está dentro si cruza un número impar de bordes contando todos los
anillos (así los huecos salen solos).
"""
import gzip
import json
import struct
import sys

from shapely.geometry import shape

LANGS = 'es ar de en fr hi it ja ko nl pl pt ru tr zh'.split()
# Cuánto se simplifica el borde (grados): ~30 m en municipios.
TOL = {2: 0.01, 3: 0.002, 4: 0.0015, 5: 0.001, 6: 0.0008, 7: 0.0004, 8: 0.0003}
Q = 10000


def varint(n, out):
    n = (n << 1) ^ (n >> 63)  # zigzag
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return


def rings(geom):
    polys = [geom] if geom.geom_type == 'Polygon' else list(getattr(geom, 'geoms', []))
    for p in polys:
        if p.geom_type != 'Polygon' or p.is_empty:
            continue
        yield p.exterior.coords
        for r in p.interiors:
            yield r.coords


def encode(level, names, geom):
    tol = TOL.get(level, 0.0003)
    geom = geom.simplify(tol, preserve_topology=True)
    if geom.is_empty:
        return None
    body = bytearray()
    rs = []
    for ring in rings(geom):
        pts = []
        for x, y in ring:
            q = (round(x * Q), round(y * Q))
            if not pts or pts[-1] != q:
                pts.append(q)
        if len(pts) >= 4:
            rs.append(pts)
    if not rs:
        return None
    varint(len(rs), body)
    px = py = 0
    for pts in rs:
        varint(len(pts), body)
        for x, y in pts:
            varint(x - px, body)
            varint(y - py, body)
            px, py = x, y
    x0, y0, x1, y1 = geom.bounds
    text = '\t'.join([names[''], *(f'{k}={v}' for k, v in names.items() if k)])
    t = text.encode()[:65535]
    head = struct.pack('<BH', level, len(t)) + t + struct.pack(
        '<iiiiI', int(x0 * Q) - 1, int(y0 * Q) - 1, int(x1 * Q) + 1, int(y1 * Q) + 1, len(body))
    return head + bytes(body)


def names_of(tags, name):
    out = {'': name}
    for lang in LANGS:
        v = (tags.get(f'name:{lang}') or '').strip()
        if v and v != name:
            out[lang] = v
    return out


def write(path, items):
    with gzip.open(path, 'wb', compresslevel=9) as f:
        f.write(b'OTL1' + struct.pack('<I', len(items)))
        for it in items:
            f.write(it)


def lines(srcs):
    for src in srcs:
        with open(src) as f:
            yield from f


def osm(srcs, dst):
    items, levels, seen = [], {}, set()
    if True:
        for line in lines(srcs):
            line = line.strip().lstrip('\x1e')
            if not line:
                continue
            feat = json.loads(line)
            tags = feat.get('properties') or {}
            if tags.get('boundary') != 'administrative':
                continue
            try:
                level = int(tags.get('admin_level', ''))
            except ValueError:
                continue
            name = (tags.get('name') or '').strip()
            if not 3 <= level <= 8 or not name:
                continue
            try:
                geom = shape(feat['geometry'])
            except Exception:
                continue
            item = encode(level, names_of(tags, name), geom)
            # La misma zona puede venir en dos extractos.
            if item and item not in seen:
                seen.add(item)
                items.append(item)
                levels[level] = levels.get(level, 0) + 1
    write(dst, items)
    print(dst, len(items), 'zonas', dict(sorted(levels.items())))


def paises(src, dst):
    items = []
    for feat in json.load(open(src))['features']:
        p = feat['properties']
        code = next((c for c in (p.get('ISO_A2_EH'), p.get('ISO_A2'), p.get('WB_A2'))
                     if c and c != '-99'), None)
        if not code or not feat.get('geometry'):
            continue
        item = encode(2, {'': code}, shape(feat['geometry']))
        if item:
            items.append(item)
    write(dst, items)
    print(dst, len(items), 'países')


if __name__ == '__main__':
    if sys.argv[1] == '--paises':
        paises(sys.argv[2], sys.argv[3])
    else:
        osm(sys.argv[1:-1], sys.argv[-1])
