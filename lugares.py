"""Prueba: cuánto ocuparían los lugares de un país (los mismos que pide la
app a Overpass) en un fichero para bajar una vez.

  python3 lugares.py lugares.geojsonseq salida.json.gz
"""
import gzip
import json
import sys

from shapely.geometry import shape

LANGS = 'es ar de en fr hi it ja ko nl pl pt ru tr zh'.split()
EXTRA = ('ele population wikidata wikipedia access sport drinking_water '
         'shelter_type protect_class religion heritage capacity').split()
KEYS = set('aeroway amenity archaeological_site boundary building club emergency healthcare '
           'highway historic industrial landuse leisure man_made natural place railway shop '
           'station tourism water waterway'.split())

def main(src, dst):
    out, n, kinds = [], 0, {}
    for line in open(src):
        line = line.strip().lstrip('\x1e')
        if not line:
            continue
        f = json.loads(line)
        t = f.get('properties') or {}
        name = t.get('name')
        if not name and t.get('amenity') not in ('drinking_water', 'shelter') and t.get('natural') != 'spring':
            continue
        try:
            g = shape(f['geometry'])
            p = g if g.geom_type == 'Point' else g.representative_point()
        except Exception:
            continue
        rec = {'p': [round(p.x, 5), round(p.y, 5)]}
        for k, v in t.items():
            if k in KEYS or k in EXTRA or k == 'name' or (k.startswith('name:') and k[5:] in LANGS and v != name):
                rec[k] = v
        out.append(rec)
        for k in KEYS:
            if k in t:
                kinds[f'{k}={t[k]}'] = kinds.get(f'{k}={t[k]}', 0) + 1
    with gzip.open(dst, 'wt', compresslevel=9) as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(dst, len(out), 'lugares')
    for k, c in sorted(kinds.items(), key=lambda x: -x[1])[:40]:
        print(f'  {c:8} {k}')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
