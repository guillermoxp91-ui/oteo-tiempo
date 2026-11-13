"""Los lugares de un país (los mismos que la app pedía a Overpass) en un
fichero compacto que la app baja una vez y consulta sin conexión.

  python3 lugares.py --filtro          (lo que hay que pedir a osmium)
  python3 lugares.py lugares.geojsonseq [más…] salida
      (→ salida.cerca.gz, salida.lejos.gz y salida.servicios.gz)

«cerca» lleva lo de las categorías del mapa, para el país donde estás;
«lejos», solo sus picos y poblaciones, para los países que se ven a lo
lejos; «servicios», el resto (tiendas, aparcamientos…), solo para el
buscador, y se baja la primera vez que se busca algo así.

Formato (gzip; enteros little endian; varint = LEB128, zigzag si con signo):
  'OTP1'
  u32 número de textos, y cada uno: u16 bytes + utf-8. Son etiquetas
      «clave=valor» repetidas; las demás van dentro de cada lugar.
  u32 número de capas, y por capa: u16 bytes + utf-8 de su clave principal
      (natural, place, amenity…), u32 número de celdas y por celda: i16
      lat·10 y i16 lon·10 (esquina suroeste de su cuadro de 0,1°), u32
      posición en el bloque de lugares, u32 número de lugares.
  u32 bytes del bloque de lugares, y el bloque. Cada lugar: i32 lat·1e5,
      i32 lon·1e5, u8 tipo (n, w, r), varint id, varint número de
      etiquetas y por etiqueta varint: k > 0 → texto k-1; 0 → varint bytes
      + utf-8 «clave=valor».
Un lugar sale en cada capa a la que pertenece (una iglesia que es también
edificio, en amenity y en building).
"""
import gzip
import json
import re
import struct
import sys
from collections import Counter, defaultdict

from shapely.geometry import shape

# Lo que pide la app (osm.dart: buildOverpassQuery y osmSelectors). Si se
# añade una clase de lugar allí, hay que añadirla aquí.
SEL = {
    'aeroway': 'aerodrome',
    'amenity': 'bar bicycle_rental bus_station cafe cinema clinic college dive_centre drinking_water '
               'ferry_terminal fire_station fountain fuel grave_yard hospital library monastery pharmacy '
               'place_of_worship police pub restaurant sanitary_dump_station school shelter taxi theatre university '
               # y lo del buscador «cerca de mí» (genericSearches)
               'atm bank bbq bicycle_parking car_wash charging_station dentist doctors fast_food ice_cream '
               'kindergarten marketplace nightclub parking post_box post_office recycling toilets townhall veterinary',
    'cuisine': '*',
    'office': 'estate_agent insurance',
    'archaeological_site': 'petroglyph rock_art rock_painting',
    'boundary': 'national_park protected_area',
    'building': 'cathedral monastery',
    'club': 'sport',
    'emergency': 'defibrillator',
    'healthcare': 'centre pharmacy',
    'highway': 'rest_area services bus_stop',
    'historic': 'archaeological_site castle fort memorial monastery monument ruins',
    'industrial': 'port',
    'landuse': 'cemetery winter_sports',
    'leisure': 'fitness_centre marina nature_reserve park pitch skatepark sports_centre sports_hall stadium '
               'swimming_area swimming_pool water_park bowling_alley dog_park fitness_station golf_course '
               'horse_riding miniature_golf playground',
    'man_made': 'lighthouse water_tap water_well',
    'natural': 'beach cave_entrance spring peak volcano hill',
    'railway': 'station',
    'shop': '*',
    'sport': '*',
    'station': 'subway',
    'tourism': 'alpine_hut artwork attraction camp_site caravan_site guest_house hostel hotel motel museum '
               'picnic_site viewpoint wilderness_hut chalet',
    'water': 'lake reservoir',
    'waterway': 'waterfall',
    'place': 'city town village hamlet isolated_dwelling',
}
SEL = {k: set(v.split()) for k, v in SEL.items()}
FAR = {'natural', 'place'}


def wanted(k, v):
    vals = SEL.get(k)
    return vals is not None and isinstance(v, str) and ('*' in vals or v in vals)


def filtro():
    return ' '.join(f"nwr/{k}" if '*' in v else f"nwr/{k}={','.join(sorted(v))}" for k, v in SEL.items())


# Lo que piden las categorías del mapa (osm.dart: osmSelectors y
# buildOverpassQuery), con los mismos filtros que la app pone al servidor o
# aplica después. Lo que lo cumple va en «cerca»; el resto de SEL (lo que
# solo sirve al buscador «cerca de mí» o por nombre) va en «servicios», que
# se baja solo cuando hace falta.
def named(t):
    return bool(t.get('name'))


def always(t):
    return True


SPORTS = ('tennis', 'padel', 'basketball', 'soccer', 'skateboard')
SPORT_NAMES = re.compile(r'tenis|tennis|p[aá]d[eé]l|baloncesto|basket|f[uú]tbol|football|skate', re.I)


def plays(t):
    s = t.get('sport') or ''
    return any(x in s for x in SPORTS)


def sport_named(t):
    return bool(SPORT_NAMES.search(t.get('name') or ''))


def drinkable(t):
    return t.get('drinking_water') in ('yes', 'treated')


def notable(t):
    return named(t) and bool(t.get('wikipedia') or t.get('wikidata'))


def court(t):
    return plays(t) or sport_named(t)


CORE = {
    'tourism': {'viewpoint': always, 'alpine_hut': always, 'wilderness_hut': always,
                'caravan_site': always, 'picnic_site': always, 'camp_site': always, 'hotel': always,
                'hostel': always, 'guest_house': always, 'motel': always, 'attraction': named,
                'artwork': notable, 'museum': named},
    'amenity': {'drinking_water': lambda t: t.get('drinking_water') not in ('no', 'conditional'),
                'fountain': drinkable,
                'shelter': lambda t: named(t) and t.get('shelter_type') in ('basic_hut', 'lean_to'),
                'grave_yard': always, 'sanitary_dump_station': always, 'fuel': always,
                'restaurant': named, 'bar': named, 'cafe': named, 'pub': named,
                'place_of_worship': lambda t: named(t) and t.get('religion') == 'christian',
                'monastery': named, 'bus_station': named, 'taxi': always, 'hospital': named,
                'clinic': named, 'pharmacy': always, 'bicycle_rental': always, 'police': always,
                'fire_station': always, 'cinema': named, 'theatre': named, 'library': named,
                'school': named, 'college': named, 'university': named, 'ferry_terminal': named,
                'dive_centre': named},
    'man_made': {'water_tap': drinkable, 'water_well': drinkable, 'lighthouse': always},
    'natural': {'spring': drinkable, 'beach': named, 'cave_entrance': named,
                'peak': named, 'volcano': named, 'hill': named},
    'water': {'lake': named, 'reservoir': named},
    'waterway': {'waterfall': always},
    'leisure': {'swimming_area': always,
                'swimming_pool': lambda t: named(t) and not re.search(
                    'private|customers', t.get('access') or ''),
                'sports_centre': lambda t: named(t) or plays(t),
                'water_park': named, 'stadium': named,
                'fitness_centre': named,
                'pitch': lambda t: plays(t) and t.get('access') != 'private',
                'sports_hall': court, 'skatepark': always, 'nature_reserve': named,
                'park': named, 'marina': named},
    'club': {'sport': court},
    'shop': {'sports': sport_named, 'supermarket': always, 'car_repair': always, 'tyres': always,
             'scuba_diving': named},
    'boundary': {'national_park': named,
                 'protected_area': lambda t: named(t) and re.fullmatch('[1-6]', t.get('protect_class') or '')},
    'landuse': {'cemetery': always, 'winter_sports': named},
    'highway': {'rest_area': always, 'services': always},
    'historic': {'monument': named, 'memorial': notable, 'monastery': named,
                 'archaeological_site': named, 'castle': named, 'fort': named, 'ruins': named},
    'building': {'cathedral': named, 'monastery': named},
    'railway': {'station': named},
    'station': {'subway': named},
    'aeroway': {'aerodrome': lambda t: named(t) and bool(t.get('iata') or t.get('icao'))},
    'healthcare': {'centre': named, 'pharmacy': always},
    'emergency': {'defibrillator': always},
    'archaeological_site': {'petroglyph': always, 'rock_painting': always, 'rock_art': always},
    'industrial': {'port': named},
    'sport': {'scuba_diving': always, 'surfing': always},
    'place': {v: named for v in ('city', 'town', 'village', 'hamlet', 'isolated_dwelling')},
}


def is_core(t):
    for k, vals in CORE.items():
        keep = vals.get(t.get(k)) if isinstance(t.get(k), str) else None
        if keep and keep(t):
            return True
    return False

LANGS = 'es ar de en fr hi it ja ko nl pl pt ru tr zh'.split()
# Las etiquetas que lee la app (osm.dart, place.dart); el resto no viaja.
KEEP = set(SEL) | set((
    'name ele population wikidata wikipedia start_date stars school:type school:level isced:level '
    'religion denomination construction_date brand operator operator:type drinking_water site_type '
    'shelter_type protect_class police network icao iata historic:civilization heritage castle_type '
    'bicycle_rental artwork_type access sport addr:street addr:place addr:housenumber addr:city '
    'opening_hours phone contact:phone website contact:website url cuisine').split())


def uvarint(n, out):
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return


def lines(srcs):
    for src in srcs:
        with open(src) as f:
            yield from f


def main(srcs, dst):
    core, services = [], []
    seen = set()
    for line in lines(srcs):
        line = line.strip().lstrip('\x1e')
        if not line:
            continue
        f = json.loads(line)
        t = f.get('properties') or {}
        layers = [k for k in SEL if wanted(k, t.get(k))]
        if not layers:
            continue
        fid = str(f.get('id') or '')
        kind, num = (fid[:1], fid[1:]) if fid[:1] in 'nwra' else ('n', '0')
        if kind == 'a':  # áreas de osmium: id par = vía, impar = relación
            n = int(num or 0)
            kind, num = ('w', n // 2) if n % 2 == 0 else ('r', n // 2)
        key = (kind, int(num or 0))
        if key in seen:
            continue
        seen.add(key)
        try:
            g = shape(f['geometry'])
            p = g if g.geom_type == 'Point' else g.representative_point()
        except Exception:
            continue
        name = t.get('name')
        tags = []
        for k, v in t.items():
            if not isinstance(v, str):
                continue
            if k in KEEP or (k.startswith('name:') and k[5:] in LANGS and v != name):
                tags.append(f'{k}={v}')
        place = (round(p.y * 1e5), round(p.x * 1e5), kind, key[1], tags, layers)
        (core if is_core(t) else services).append(place)

    write(core, dst + '.cerca.gz', None)
    write(core, dst + '.lejos.gz', FAR)
    write(services, dst + '.servicios.gz', None)


def write(places, dst, only):
    if only:
        places = [(*pl[:5], [l for l in pl[5] if l in only]) for pl in places]
        places = [pl for pl in places if pl[5]]
    freq = Counter(tag for pl in places for tag in pl[4])
    table = [s for s, c in freq.most_common() if c >= 3][:65000]
    code = {s: i + 1 for i, s in enumerate(table)}

    by = defaultdict(lambda: defaultdict(list))
    for i, pl in enumerate(places):
        cell = (pl[0] // 10000, pl[1] // 10000)  # cuadros de 0,1°
        for layer in pl[5]:
            by[layer][cell].append(i)

    blob = bytearray()
    enc = {}

    def record(i):
        if i in enc:
            return enc[i]
        lat, lon, kind, num, tags, _ = places[i]
        r = bytearray(struct.pack('<iiB', lat, lon, ord(kind)))
        uvarint(num, r)
        uvarint(len(tags), r)
        for tag in tags:
            c = code.get(tag, 0)
            uvarint(c, r)
            if not c:
                b = tag.encode()
                uvarint(len(b), r)
                r += b
        enc[i] = bytes(r)
        return enc[i]

    layer_bytes = bytearray(struct.pack('<I', len(by)))
    for layer in sorted(by):
        cells = by[layer]
        lb = layer.encode()
        layer_bytes += struct.pack('<H', len(lb)) + lb + struct.pack('<I', len(cells))
        for (clat, clon) in sorted(cells):
            ids = cells[(clat, clon)]
            start = len(blob)
            for i in ids:
                blob += record(i)
            layer_bytes += struct.pack('<hhII', clat, clon, start, len(ids))

    with gzip.open(dst, 'wb', compresslevel=9) as out:
        out.write(b'OTP1' + struct.pack('<I', len(table)))
        for s in table:
            b = s.encode()[:65535]
            out.write(struct.pack('<H', len(b)) + b)
        out.write(layer_bytes)
        out.write(struct.pack('<I', len(blob)))
        out.write(blob)
    print(dst, len(places), 'lugares,', len(table), 'textos,', len(blob) // 1024, 'KB sin comprimir')
    for layer in sorted(by, key=lambda l: -sum(len(v) for v in by[l].values())):
        print(f'  {sum(len(v) for v in by[layer].values()):8} {layer}')


if __name__ == '__main__':
    if sys.argv[1] == '--filtro':
        print(filtro())
    else:
        main(sys.argv[1:-1], sys.argv[-1])
