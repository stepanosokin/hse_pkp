# import shapely
from tkinter.constants import Y
from urllib.parse import parse_qsl
import geopandas as gpd
import pandas as pd
from pyproj import Geod, crs, Transformer
import shapely
import re
import json
import sqlalchemy
from zipfile import ZipFile
import os
import rasterio
import numpy as np
from tqdm import tqdm
import math
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.morphology import skeletonize
from centerline.geometry import Centerline

# Сложности https://github.com/astral-sh/uv/issues/11466
from osgeo import gdal
from osgeo import gdal_array
from osgeo_utils import gdal_calc
# from vgdb_general import smart_http_request
# import requests
import yadisk
import boto3


def drop_small_holes(poly, min_hole_area):
    if poly.geom_type != "Polygon":
        return poly
    # keep only holes with area >= threshold
    keep_holes = [r for r in poly.interiors if shapely.geometry.Polygon(r).area >= min_hole_area]
    return shapely.geometry.Polygon(poly.exterior, keep_holes)


def drop_small_holes_any(geom, min_hole_area):
    if geom.geom_type == "Polygon":
        return drop_small_holes(geom, min_hole_area)
    if geom.geom_type == "MultiPolygon":
        return shapely.geometry.MultiPolygon([drop_small_holes(p, min_hole_area) for p in geom.geoms])
    return geom


def get_y_token(yinfo='.secret/.yinfo'):
    """Получение токена доступа к Яндекс.Диску.

    Аргументы:
    - yinfo (str): путь к JSON-файлу с параметрами/учётными данными для доступа
      к Яндекс.Диску (например, '.secret/.yinfo').

    Структура файла .yinfo (JSON):
    - app_name (str): название приложения в Яндекс.OAuth. Используется для
      идентификации приложения, не участвует напрямую в аутентификации.
    - account_name (str): имя аккаунта Яндекса (например, e-mail), от имени
      которого зарегистрирован OAuth‑клиент и выдавались права.
    - client_id (str): идентификатор OAuth‑клиента, выданный при регистрации
      приложения в Яндекс.OAuth. Публичная часть пары для запроса токена.
    - client_secret (str): секрет OAuth‑клиента (приватная часть), используется
      совместно с client_id для получения/обновления токена. Должен храниться в
      секрете и не коммититься в репозиторий.
    """
    try:
        with open(yinfo, encoding='utf-8') as f:
            yi = json.load(f)
    except:
        raise
    if yi['client_id'] is None or yi['client_secret'] is None or yi['redirect_uri'] is None:
        raise ValueError("client_id or client_secret or redirect_uri is None")
    
    # https://yandex.ru/dev/id/doc/ru/codes/code-and-token
    # https://yadisk.readthedocs.io/ru/latest/intro.html
    with yadisk.Client(yi['client_id'], yi['client_secret']) as client:
        url = client.get_code_url()
        pass
    
    #####################################################
    # url = "https://oauth.yandex.ru/authorize"
    # headers = {
    #     "accept": "application/json",
    #     "accept-encoding": "gzip, deflate, br, zstd",
    # }
    # params = {
    #     "response_type": "code",
    #     "client_id": yi['client_id']
    # }
    # with requests.Session() as s:
    #     status, auth_code_response = smart_http_request(s, url=url, params=params, headers=headers)
    #     if status != 200:
    #         raise ValueError("auth_code_response is not 200")
    #     # auth_code_json = json.loads(auth_code_response.text)
    #     # auth_code_response_txt = auth_code_response.text
    #     # auth_code = auth_code_response.text.split('code=')[1].split('&')[0]
    #     # auth_code = auth_code_response.text.split('code=')[1].split('&')[0]
    #     response = requests.get(yi['redirect_uri'])
    #     pass

def get_lulc_from_y(link='', token=''):
    """Для создания токена доступа к Яндекс.Диску, надо зарегистрировать приложение в Яндексе и создать для него токен:
        перейти по адресу https://oauth.yandex.ru/client/new/api
        Создайте новое приложение:
        Заполните поля "Название" и "Иконка" для вашего приложения.
        Выберите тип платформы "Веб-сервисы".
        В поле Redirect URI укажите https://oauth.yandex.ru/verification_code.
        Добавьте необходимые права доступа:
        Нажмите "Добавить" в разделе "Название доступа" и выберите разрешение, например, "Запись в любом месте на Диске".
        Получите токен:
            Нажмите "Добавить приложение".
            Скопируйте сформированную ссылку из раздела "Для веб-сервисов", вставьте её в адресную строку браузера.
        На странице авторизации выберите нужные права для приложения и нажмите "Разрешить".
        Ваш токен будет указан на открывшейся странице, скопируйте его и используйте для доступа к Яндекс Диску. 
    Для данного проекта сгенерировано приложение pkp_bot и указана привилегия cloud_api:disk.read.    
    Использован Яндекс-профиль s.osokin@hse.ru с привязкой к аккаунту Госуслуг "Осокин Степан Артемович".
    """
    pass


def download_from_y_obj_storage(
    bucket='pkp-lulc-bucket',
    key='S2LULC_10m_LAEA_48_202507081046.tif',
    dest_file='lulc/S2LULC_10m_LAEA_48_202507081046.tif',
    y_static_key_file='.secret/pkp-bot-static-key.json'
    ):
    pass
    # https://yandex.cloud/ru/docs/iam/concepts/authorization/access-key
    # https://yandex.cloud/ru/docs/storage/s3/  
    # # это вроде не подходит для Object Storage https://yandex.cloud/ru/docs/iam/operations/iam-token/create-for-sa#via-jwt
    # https://yandex.cloud/ru/docs/storage/operations/objects/download
    # https://yandex.cloud/ru/docs/storage/s3/api-ref/object/get
    # https://yandex.cloud/en/docs/datasphere/operations/data/connect-to-s3

    try:
        with open(y_static_key_file, encoding='utf-8') as f:
            creds = json.load(f)
    except:
        raise
    if creds['key_id'] is None or creds['secret'] is None:
        raise ValueError("key_id or secret is None")
    session = boto3.session.Session()
    s3 = session.client(
        service_name='s3',
        endpoint_url='https://storage.yandexcloud.net',
        aws_access_key_id=creds['key_id'],
        aws_secret_access_key=creds['secret']
    )
    # Получить список объектов в бакете
    for k in s3.list_objects(Bucket=bucket)['Contents']:
        print(k['Key'])
        pass
    try:
        # s3.download_file(Bucket=bucket, Key=key, Filename=dest_file)
        s3.download_file(bucket, key, dest_file)
    except Exception as e:
        raise RuntimeError(f"Failed to download {key} from {bucket}: {e}")
        pass





def get_region_shortname(region):
    regions_dict = {
        "Алтайский край": "Altayskiy",
        "Амурская область": "Amurskaya",
        "Архангельская область": "Arkhangelskaya",
        "Астраханская область": "Astrakhanskaya",
        "Белгородская область": "Belgorodskaya",
        "Брянская область": "Bryanskaya",
        "Владимирская область": "Vladimirskaya",
        "Волгоградская область": "Volgogradskaya",
        "Вологодская область": "Vologodskaya",
        "Воронежская область": "Voronezhskaya",
        "г. Москва": "Moskva",
        "г. Санкт-Петербург": "Sankt-Peterburg",
        "г. Севастополь": "Sevastopol",
        "Еврейская автономная область": "Evreyskaya",
        "Забайкальский край": "Zabaykalskiy",
        "Ивановская область": "Ivanovskaya",
        "Иркутская область": "Irkutskaya",
        "Кабардино-Балкарская Республика": "Kabardino-Balkariya",
        "Калининградская область": "Kaliningradskaya",
        "Калужская область": "Kaluzhskaya",
        "Камчатский край": "Kamchatskiy",
        "Карачаево-Черкесская Республика": "Karachaevo-Cherkessiya",
        "Кировская область": "Kirovskaya",
        "Костромская область": "Kostromskaya",
        "Краснодарский край": "Krasnodarskiy",
        "Красноярский край": "Krasnoyarskiy",
        "Курганская область": "Kurganskaya",
        "Курская область": "Kurskaya",
        "Ленинградская область": "Leningradskaya",
        "Липецкая область": "Lipetskaya",
        "Магаданская область": "Magadanskaya",
        "Московская область": "Moskovskaya",
        "Мурманская область": "Murmanskaya",
        "Ненецкий автономный округ": "Nenetskiy",
        "Нижегородская область": "Nizhegorodskaya",
        "Новгородская область": "Novgorodskaya",
        "Новосибирская область": "Novosibirskaya",
        "Омская область": "Omskaya",
        "Оренбургская область": "Orenburgskaya",
        "Орловская область": "Orlovskaya",
        "Пензенская область": "Penzenskaya",
        "Пермский край": "Permskiy",
        "Приморский край": "Primorskiy",
        "Псковская область": "Pskovskaya",
        "Республика Алтай": "Altay",
        "Республика Башкортостан": "Bashkortostan",
        "Республика Бурятия": "Buryatiya",
        "Республика Дагестан": "Dagestan",
        "Республика Ингушетия": "Ingushetiya",
        "Республика Калмыкия": "Kalmykiya",
        "Республика Карелия": "Kareliya",
        "Республика Коми": "Komi",
        "Республика Крым": "Krym",
        "Республика Марий Эл": "Mariy-El",
        "Республика Мордовия": "Mordoviya",
        "Республика Саха (Якутия)": "Sakha",
        "Республика Северная Осетия": "Severnaya Osetiya",
        "Республика Тыва": "Tyva",
        "Республика Хакассия": "Khakassiya",
        "Ростовская область": "Rostovskaya",
        "Рязанская область": "Ryazanskaya",
        "Самарская область": "Samarskaya",
        "Саратовская область": "Saratovskaya",
        "Сахалинская область": "Sakhalinskaya",
        "Свердловская область": "Sverdlovskaya",
        "Смоленская область": "Smolenskaya",
        "Ставропольский край": "Stavropolskiy",
        "Тамбовская область": "Tambovskaya",
        "Тверская область": "Tverskaya",
        "Томская область": "Tomskaya",
        "Тульская область": "Tulskaya",
        "Тюменская область": "Tyumenskaya",
        "Удмуртская Республика": "Udmurtskaya",
        "Ульяновская область": "Ulyanovskaya",
        "Хабаровский край": "Khabarovskiy",
        "Челябинская область": "Chelyabinskaya",
        "Чеченская Республика": "Chechenskaya",
        "Чукотский автономный округ": "Chukotskiy",
        "Ямало-Ненецкий автономный округ": "Yamalo-Nenetskiy",
        "Ярославская область": "Yaroslavskaya",
        "Ханты-Мансийский автономный округ - Югра": "Ugra",
        "Кемеровская область - Кузбасс": "Kuzbass",
        "Республика Адыгея (Адыгея)": "Adygeya",
        "Республика Татарстан (Татарстан)": "Tatarstan",
        "Чувашская Республика - Чувашия": "Chuvashiya"
    }
    return regions_dict.get(region, None)


def calculate_geod_buffers(
    i_gdf: gpd.GeoDataFrame,
    buffer_crs: str,
    buffer_dist_source: str,
    buffer_distance,
    geom_field='geom'
):
    """_summary_

    Args:
        i_gdf (gpd.GeoDataFrame): входной набор данных (геометрия в WGS-1984)
        buffer_crs (str): проекция для расчета буфера: 'utm' или 'laea'
        buffer_dist_source (str): источник значения размера буфера: 'field' (столбец в датафрейме) или 'value' (фикс значение)
        buffer_distance: в зависимости от значения buffer_dist_source - название столбца (str) или фикс значение (float)

    Raises:
        ValueError: Если Некорректно задан параметр buffer_crs

    Returns:
        list: Список с геометриями буферов
    """
    # Расчет буферов вокруг линейных водных объектов
    buffer_geom = []
    for i, row in i_gdf.iterrows():   # итерация по линейным объектам
        # формирование проекции UTM с центральным меридианом в центроиде текущего объекта
        lon = row[geom_field].centroid.x
        lat = row[geom_field].centroid.y
        if buffer_crs == 'utm':
            buf_crs = crs.CRS.from_proj4(
                f"+proj=tmerc +lat_0=0 +lon_0={lon} " \
                f"+k=0.9996 +x_0=500000 +y_0=0 +ellps=WGS84 " \
                f"+units=m +no_defs +type=crs"
            )
        elif buffer_crs == 'laea':
            buf_crs = crs.CRS.from_proj4(
                f"+proj=laea +lat_0={lat} +lon_0={lon} " \
                f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
            )
        else:
            raise ValueError("Некорректно задан параметр buffer_crs")
        # модели пересчета туда и обратно
        transformer1 = Transformer.from_crs(4326, buf_crs, always_xy=True)
        transformer2 = Transformer.from_crs(buf_crs, 4326, always_xy=True)
        # вычисление буфера в текущей проекции UTM
        try:
            if buffer_dist_source == 'value':
                buffer = shapely.buffer(shapely.transform(row[geom_field], transformer1.transform, interleaved=False), float(buffer_distance))
            elif buffer_dist_source == 'field':
                buffer = shapely.buffer(shapely.transform(row[geom_field], transformer1.transform, interleaved=False), float(row[buffer_distance]))
            else:
                raise ValueError("Неверно задан параметр buffer_dist_source")
            # Пересчет буфера обратно в WGS-1984
            buffer = shapely.transform(buffer, transformer2.transform, interleaved=False)
            # Добавить результат в список
            buffer_geom.append(buffer)
        except Exception as e:
            print(f"Failed to calculate buffer: {e}")
    return buffer_geom


def prepare_water_limitations(
    postgres_info='.secret/.gdcdb',
    region='Липецкая область',
    # source_water_line='data/water_line.shp',
    water_line_table='osm.gis_osm_waterways_free',
    water_pol_table='osm.gis_osm_water_a_free',
    regions_table='admin.hse_russia_regions',
    region_buf_size=0,
    # source_water_pol='data/water_pol.shp',
    # result_gpkg='result/water_limitations.gpkg',
    buffer_distance = 5,
    buffer_crs = 'utm',
    geometry_field='geom'
):
    """Функция рассчитывает ограничения для проекта ПКП для линейных и площадных водных объектов по методике Жанны. 
    Документация дорабатывается.

    Args:
        source_water_line (str, optional): Линейные водные объекты, исходные данные. Defaults to 'data/water_line.shp'.
        source_water_pol (str, optional): Площадные водные объекты, исходные данные. Defaults to 'data/water_pol.shp'.
        result_gpkg (str, optional): относительный путь к Geopackage для сохранения результатов.
        buffer_distance (int, optional): расстояние первого буфера от линейных объектов в м. Будет рассчитан в UTM с осевым меридианом в центроиде объекта или в LAEA с центром в центроиде объекта. Defaults to 5.
        buffer_crs (str, optional): проекция для расчета буферов. 'utm' или 'laea'. Defaults to 'utm'.
    """
    # try:
    #     gdf_l = gpd.read_file(source_water_line)
    #     gdf_p = gpd.read_file(source_water_pol)
    # except:
    #     raise
    
    ###################################################################

    try:
        with open(postgres_info, encoding='utf-8') as f:
            pg = json.load(f)
    except:
        raise
    try:
        engine = sqlalchemy.create_engine(
            f"postgresql+psycopg2://{pg['user']}:{pg['password']}@{pg['host']}:{pg['port']}/{pg['database']}",
            connect_args={
                "sslmode": "verify-full",
                "target_session_attrs": "read-write"
                # "sslcert": "/path/to/client.crt",
                # "sslkey": "/path/to/client.key",
                # "sslrootcert": "/path/to/ca.crt",
            },
        )
        pass
    except:
        raise
    try:
        sql = f"select * from {regions_table} where lower(region) = '{region.lower()}';"
        region_gdf = gpd.read_postgis(sql, engine)
        if region_buf_size > 0:
            region_buffer = calculate_geod_buffers(region_gdf, 'laea', 'value', region_buf_size, geom_field='geom')
            region_gdf = region_gdf.set_geometry(region_buffer)
        sql = f"select * from {water_line_table} wl " \
            f"where ST_Intersects(" \
            f"(select ST_Buffer(geom::geography, {region_buf_size})::geometry from {regions_table} where lower(region) = '{region.lower()}' limit 1), " \
            f"wl.geom" \
            f");"
        gdf_l = gpd.read_postgis(sql, engine)
        sql = f"select * from {water_pol_table} wp " \
            f"where ST_Intersects(" \
            f"(select ST_Buffer(geom::geography, {region_buf_size})::geometry from {regions_table} where lower(region) = '{region.lower()}' limit 1), " \
            f"wp.geom" \
            f");"
        gdf_p = gpd.read_postgis(sql, engine)
    except:
        raise

    ######################################
    
    # сливаем геометрию по полям fclass и name
    gdf_l = gdf_l.dissolve(by=['fclass', 'name'], dropna=False)
    
    # Объединяем куски полилиний
    gdf_l[geometry_field] = gdf_l[geometry_field].line_merge()
    
    # пересчитать в WGS-84 для вычисления геодезических расстояний
    gdf_l = gdf_l.to_crs(4326)
    
    # рассчитать геодезические длины
    geod = Geod(ellps='WGS84')
    gdf_l['leng'] = [geod.geometry_length(row[geometry_field]) for i, row in gdf_l.iterrows()]    
    # gdf.to_file(result_gpkg, layer='water_line_dissolved')

    # разбить мульти полилинии на отдельные
    gdf_l = gdf_l.explode()
    
    # рассчитать геодезические длины после разбивки на отдельные линии
    gdf_l['leng'] = [geod.geometry_length(row[geometry_field]) for i, row in gdf_l.iterrows()]    
    # gdf.to_file(result_gpkg, layer='water_line_dissolved_exploded')
    
    # отфильтровать ручьи и каналы короче 10 км
    gdf_l = gdf_l.query('not (fclass in("stream", "canal") and leng < 10000)')
    # gdf_l.to_file(result_gpkg, layer='water_line_dissolved_exploded_filtered')
    
    # Расчет буферов вокруг линейных водных объектов
    buffer_geom = calculate_geod_buffers(gdf_l, buffer_crs, 'value', buffer_distance, geom_field='geom')
    
    # gdf_l['buffer_geometry'] = buffer_geom
    gdf_l_buffered = gdf_l.set_geometry(buffer_geom)
    my_columns = gdf_l_buffered.columns.tolist()
    # my_columns.remove('buffer_geometry')
    gdf_l_buffered = gdf_l_buffered[my_columns]
    # расчет значений размеров буферов
    gdf_l_buffered['buf'] = [50 if x <= 10000 else 100 if 10000 < x <= 50000 else 200 for x in gdf_l_buffered['leng']]
    gdf_l['buf'] = [50 if x <= 10 else 100 if 10000 < x <= 50000 else 200 for x in gdf_l['leng']]
    # gdf_l.to_file(result_gpkg, layer='water_lines_buf')
    water_lines_buf = gdf_l_buffered.copy()
    # вычислить геодезические буферы по полю buf  
    buffer_geom = calculate_geod_buffers(gdf_l_buffered, buffer_crs, 'field', 'buf', geom_field='geom')
    wp2 = gdf_l_buffered.set_geometry(buffer_geom)  
    # wp2.to_file(result_gpkg, layer='wp2')
    
    # ---------------------------------------
    # обработка полигональных водных объектов
    
    # перепроецировать в WGS-1984
    gdf_p = gdf_p.to_crs(4326)
    # убрать все wetland
    gdf_p = gdf_p.query('not fclass in("wetland")')
    water_poly = gdf_p.copy()
    
    # Вычисление "длин" площадных объектов
    # Сделать пространственное присоединение обработанных линейных объектов по пересечению
    gdf_l = gdf_l.clip(gdf_p)
    gdf_l['leng'] = [geod.geometry_length(row[geometry_field]) for i, row in gdf_l.iterrows()]
    # gdf_l.to_file(result_gpkg, layer='water_lines_clipped')
    gdf_p = gdf_p.sjoin(gdf_l, how='left')
    # Отсортировать результат по длине присоединенного объекта по убыванию
    gdf_p = gdf_p.sort_values(by='leng', ascending=False)
    # Убрать дубликаты, оставив только первый - с максимальным значением длины
    gdf_p = gdf_p.drop_duplicates(subset=['osm_id_left'], keep='first')
    # Удалить лишние столбцы и убрать суффиксы от присоединения
    columns_to_drop = gdf_p.filter(regex='_right').columns.tolist()
    gdf_p = gdf_p.drop(columns=columns_to_drop)
    gdf_p = gdf_p.drop(columns=['width'])
    new_column_names = {col: re.sub(r'_left', '', col.lower()) for col in gdf_p.columns}
    gdf_p = gdf_p.rename(columns=new_column_names)
    
    # Вычислить геодезические площади
    gdf_p['area_km'] = [abs(geod.geometry_area_perimeter(row[geometry_field])[0]) / 1000000 for i, row in gdf_p.iterrows()]
    
    # Там где не присоединилась длина линейного объекта и площадь больше 0.5, вставить buf=50
    gdf_p['buf'] = [50 if all([x >= 0.5, y]) else z for x, y, z in zip(gdf_p['area_km'], gdf_p['leng'].isna(), gdf_p['buf'])]
    # gdf_p.to_file(result_gpkg, layer='water_poly_leng')
    # убрать объекты с пустыми значениями buf
    gdf_p = gdf_p[gdf_p['buf'].notnull()]
    # вычислить геодезические буферы по полю buf  
    buffer_geom = calculate_geod_buffers(gdf_p, buffer_crs, 'field', 'buf')
    wp1 = gdf_p.set_geometry(buffer_geom)
    # wp1.to_file(result_gpkg, layer='wp1')
    
    # Объединение двух итоговых слоев
    wp1_wp2 = pd.concat([wp1, wp2])
    # wp1_wp2.to_file('result/water_line.gpkg', layer='wp1_wp2')
    water_prot_zone = wp1_wp2.union_all()
    water_prot_zone = gpd.GeoDataFrame(gpd.GeoSeries(water_prot_zone))
    water_prot_zone = water_prot_zone.rename(columns={0:geometry_field}).set_geometry(geometry_field)
    water_prot_zone = water_prot_zone.set_crs('epsg:4326')
    water_prot_zone = water_prot_zone.explode()

    region_shortname = get_region_shortname(region)
    
    # water_prot_zone.to_file(result_gpkg, layer='water_prot_zone')
    water_prot_zone.to_file(f"result/{region_shortname}_limitations.gpkg", layer='water_prot_zone')

    return (water_poly, water_lines_buf, water_prot_zone)


def prepare_slope_limitations(
    postgres_info='.secret/.gdcdb',
    region='', 
    slope_threshold = 12,
    regions_table='admin.hse_russia_regions', 
    region_buf_size=5000,
    fabdem_tiles_table='elevation.fabdem_v1_2_tiles',
    fabdem_zip_path=r"\\172.21.204.20\geodata\_PROJECTS\pkp\vm0047_prod\dem_fabdem",
    rescale=True,
    rescale_size=10
    ):
    """Формирует ограничения по уклонам рельефа для указанного региона на основе тайлов DEM FABDEM.

    Процедура выполняет следующие основные шаги:
    - извлекает границу региона из PostGIS и (опционально) расширяет её на заданное расстояние;
    - определяет список покрывающих регион тайлов FABDEM из служебной таблицы;
    - для каждого тайла: извлекает rasters из ZIP, (опционально) изменяет разрешение, вычисляет растр уклонов (в градусах),
      реклассифицирует по порогу `slope_threshold` и векторизует пиксели, удовлетворяющие условию (>= порога);
    - объединяет результаты по всем тайлам, обрезает по границе региона и сохраняет в GeoPackage слоя ограничений.

    Аргументы:
    - postgres_info (str): Путь к JSON-файлу с параметрами подключения к Postgres (`user`, `password`, `host`, `port`, `database`). По умолчанию '.secret/.gdcdb'.
    - region (str): Название региона в точном соответствии с атрибутом `region` в таблице `regions_table`.
    - slope_threshold (int | float): Порог уклона (в градусах). Пиксели с уклоном >= порога попадают в ограничения. По умолчанию 12.
    - regions_table (str): Полное имя таблицы PostGIS с границами регионов. По умолчанию 'admin.hse_russia_regions'.
    - region_buf_size (int): Радиус буфера (в метрах) вокруг региона для отбора тайлов DEM. 0 — без буфера. По умолчанию 5000.
    - fabdem_tiles_table (str): Таблица PostGIS со списком тайлов FABDEM и их геометриями/именами файлов.
    - fabdem_zip_path (str): Путь к каталогу с ZIP-архивами тайлов FABDEM.
    - rescale (bool): Признак изменения пространственного разрешения входных растров перед расчётом уклона. По умолчанию True.
    - rescale_size (int | float): Целевое размерение пикселя в метрах при рескейле. По умолчанию 10.

    Возвращает:
    - geopandas.GeoDataFrame | None: Векторный слой полигонов зон с уклонами >= `slope_threshold`,
      приведённый к CRS EPSG:4326 и обрезанный по региону. Если подходящих зон нет, возвращает None.

    Побочные эффекты:
    - Читает файл настроек подключения к БД, указанный в `postgres_info` (по умолчанию '.secret/.gdcdb').
    - Создаёт временную папку `fabdem/` в текущем рабочем каталоге и временные растровые файлы (которые затем удаляются).
    - Сохраняет итоговый слой в GeoPackage `result/{region_shortname}_limitations.gpkg` с именем слоя
      `{region_shortname}_{rescale_size}m_sl_more_{slope_threshold}_vector`.

    Примечания по реализации:
    - Расчёт уклона выполняется через `gdal.DEMProcessing` в градусах. Масштаб по осям рассчитывается с учётом широты тайла,
      чтобы корректно интерпретировать растры, заданные в градусах.
    - Если `rescale=True`, изменение разрешения делается через `gdal.Warp` с билинейной интерполяцией и компрессией LZW.
    - Все промежуточные растры по каждому тайлу удаляются по завершении обработки тайла.

    Исключения:
    - Любые ошибки чтения БД/файлов или обработки растров пробрасываются наверх, чтобы вызывающая сторона могла их обработать.
    """

    try:
        with open(postgres_info, encoding='utf-8') as f:
            pg = json.load(f)
    except:
        raise
    try:
        engine = sqlalchemy.create_engine(
            f"postgresql+psycopg2://{pg['user']}:{pg['password']}@{pg['host']}:{pg['port']}/{pg['database']}",
            connect_args={
                "sslmode": "verify-full",
                "target_session_attrs": "read-write"
                # "sslcert": "/path/to/client.crt",
                # "sslkey": "/path/to/client.key",
                # "sslrootcert": "/path/to/ca.crt",
            },
        )
        pass
    except:
        raise
    try:
        sql = f"select * from {regions_table} where lower(region) = '{region.lower()}';"
        region_gdf = gpd.read_postgis(sql, engine)
        region_buffer = calculate_geod_buffers(region_gdf, 'laea', 'value', region_buf_size, geom_field='geom')
        region_gdf = region_gdf.set_geometry(region_buffer)
        sql = f"select * from {fabdem_tiles_table} fbdm where ST_Intersects((select geom from {regions_table} where lower(region) = '{region.lower()}' limit 1), fbdm.geom);"
        tiles_gdf = gpd.read_postgis(sql, engine)
    except:
        raise
    pass

    current_dir = os.getcwd()
    os.environ["PROJ_LIB"] = os.path.join(current_dir, '.venv', 'Lib', 'site-packages', 'osgeo', 'data', 'proj')
    fabdemdir = os.path.join(current_dir, 'fabdem')
    if not os.path.isdir(fabdemdir):
        os.mkdir(fabdemdir)
    final_gdf = None

    region_centr_lat = region_gdf.iloc[0]['geom'].centroid.y
    region_centr_lon = region_gdf.iloc[0]['geom'].centroid.x

    for i, row in tqdm(tiles_gdf.iterrows(), desc='tiles loop', total=tiles_gdf.shape[0]):
        lon = row['geom'].centroid.x
        lat = row['geom'].centroid.y
        
        # extract current Fabdem tile
        zipfilename = row['zipfile_name']
        filename = row['file_name'].replace("N0", "N")
        zippath = os.path.join(fabdem_zip_path, zipfilename)
        try:
            with ZipFile(zippath, 'r') as zObject:
                zObject.extract(filename, path=fabdemdir)
            # print(f"Successfully extracted '{file_to_extract}' to '{destination_directory}'")
        except:
            raise
        input_file = os.path.join(fabdemdir, filename)
        input_dem = gdal.Open(input_file)
        
        # # reproject       # Перепроецирование в итоге убрали
        # if input_dem:
        #     # # UTM:
        #     # target_crs = f"+proj=tmerc +lat_0=0 +lon_0={lon} " \
        #     #     f"+k=0.9996 +x_0=500000 +y_0=0 +ellps=WGS84 " \
        #     #     f"+units=m +no_defs +type=crs"
        #     # LAEA:
        #     target_crs = f"+proj=laea +lat_0={region_centr_lat} +lon_0={region_centr_lon} " \
        #                  f"+x_0=4321000 +y_0=3210000 +ellps=WGS84 +units=m +no_defs"
        #     # # LAEA EPSG:3035
        #     # target_crs = f"+proj=laea +lat_0=52 +lon_0=10 " \
        #     #              f"+x_0=4321000 +y_0=3210000 +ellps=WGS84 +units=m +no_defs"
            
        #     output_reproj = os.path.join(fabdemdir, filename.replace('.tif', '_reproj.tif'))
        #     gdal.Warp(output_reproj, input_dem, dstSRS=target_crs)
            
        #     # temp_ds = temp_driver.Create('', 1, 1, 1, gdal.GDT_Byte)
        #     # gdal.Warp(temp_ds, input_dem, dstSRS=target_crs)
        #     input_dem = None
        #     input_dem = gdal.Open(output_reproj)
        #     pass
        
        # Rescaling with improved options
        output_rescale = None
        if rescale:
            rescale_options = gdal.WarpOptions(
                # xRes=rescale_size,
                # yRes=rescale_size,
                # xRes=rescale_size / (111320 * math.cos(math.radians(lat))),
                # yRes=rescale_size / (111320 * math.cos(math.radians(lat))),
                xRes=rescale_size / math.sqrt((111132.954 - (559.822 * math.cos(math.radians(2 * lat))) + 1.175 * math.cos(math.radians(4 * lat))) * (111132.954 * math.cos(math.radians(lat)))),
                yRes=rescale_size / math.sqrt((111132.954 - (559.822 * math.cos(math.radians(2 * lat))) + 1.175 * math.cos(math.radians(4 * lat))) * (111132.954 * math.cos(math.radians(lat)))),
                resampleAlg='bilinear',
                outputType=gdal.GDT_Float32,  # Specify output data type
                creationOptions=['COMPRESS=LZW', 'TILED=YES'],  # Compression and tiling
                multithread=True  # Enable multithreading for better performance
            )
            output_rescale = os.path.join(fabdemdir, filename.replace('.tif', '_reproj_rescale.tif'))
            # раскомментить если нужно перепроецировать
            # input_dem = None
            # input_dem = gdal.Open(output_reproj)
            if input_dem is None:
                # print(f"Error: Could not open {output_reproj}")
                pass
            else:
                gdal.Warp(output_rescale, input_dem, options=rescale_options)
                input_dem = None
                input_dem = gdal.Open(output_rescale)
                # gdal.Warp(output_rescale, temp_ds, options=rescale_options)        
        
        # Calculate slope
        # input_dem = None
        # input_dem = gdal.Open(output_rescale)        
        output_slope = os.path.join(fabdemdir, filename.replace('.tif', '_slope.tif'))
        if input_dem is None:
            print(f"Error: Could not open {os.path.join(fabdemdir, filename)}")
        else:
            # https://gdal.org/en/stable/api/python/utilities.html#osgeo.gdal.DEMProcessing
            # далее несколько вариантов расчета вертикального масштаба при вычислении slope, подсказанные GPT-5 (low reasoning)
            # scale = 1             # Это если в метрах
            # scale = 111320        # Это если в градусах и без учета широты
            # scale = 111320 * math.cos(math.radians(lat))      # Это если в градусах и по упрощенной формуле только с учетом масштаба по долготе в зависимости от широты
            # scale = 111132.954 * math.cos(math.radians(lat))  # Это если в градусах и по упрощенной формуле с учетом масштаба по долготе в зависимости от широты, с уточненным коэффициентом
            scale = math.sqrt((111132.954 - (559.822 * math.cos(math.radians(2 * lat))) + 1.175 * math.cos(math.radians(4 * lat))) * (111132.954 * math.cos(math.radians(lat))))  # Это если в градусах, по самой точной формуле как среднее геометрическое масштабов по долготе и широте в зависимости от широты
            gdal.DEMProcessing(
                output_slope, input_dem, "slope", 
                computeEdges=True, slopeFormat="degree", scale=scale   # Это если в градусах
                )
            # print(f"Slope calculated and saved to {output_slope}")
            pass
        
        # Reclass slope
        output_slope_reclass = os.path.join(fabdemdir, filename.replace('.tif', '_slope_reclass.tif'))
        calc_expression = f"(A>={slope_threshold})*1"
        ds = gdal_calc.Calc(calc_expression, A=output_slope, outfile=output_slope_reclass, overwrite=True)
        pass
        
        # Vectorize reclassed slope
        # https://www.google.com/search?q=python+rasterio+polygonize&sca_esv=7550878c098e0420&ei=AqK5aIbtPLqbwPAPxvCE6A8&ved=0ahUKEwiG9ujCqL-PAxW6DRAIHUY4Af0Q4dUDCBA&uact=5&oq=python+rasterio+polygonize&gs_lp=Egxnd3Mtd2l6LXNlcnAiGnB5dGhvbiByYXN0ZXJpbyBwb2x5Z29uaXplMgUQABjvBTIFEAAY7wUyBRAAGO8FSMUYUPQGWMIRcAF4AZABAJgBhwGgAbIEqgEDNy4xuAEDyAEA-AEBmAIIoALCA8ICChAAGLADGNYEGEfCAggQABgHGAgYHsICCBAAGIAEGKIEwgILEAAYgAQYhgMYigXCAgYQABgIGB6YAwCIBgGQBgiSBwE4oAfaFrIHATe4B7kDwgcFMC43LjHIBxM&sclient=gws-wiz-serp
        with rasterio.open(output_slope_reclass) as src:
            image = src.read(1)
            transform = src.transform
            crs = src.crs
            # mask = (image != 0 and image != src.nodata)
            mask = (image == 1) & (image != src.nodata)
            results = (
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v) in enumerate(
                    rasterio.features.shapes(image, mask=mask, transform=transform) # Replace mask=None with your mask if needed
                )
            )
            geoms = list(results)
            gdf = gpd.GeoDataFrame.from_features(geoms)
            gdf = gdf.set_crs(crs)
            gdf = gdf.to_crs('EPSG:4326')
            # add current vector data to final geodataframe
            if i == 0:
                final_gdf = gdf                
            else:
                final_gdf = pd.concat([final_gdf, gdf], ignore_index=True)
            # gdf.to_file('result/slope_limitations.gpkg', layer=filename.replace('.tif', '_slope'))
        ds = None
        input_dem = None
        # Delete intermediate rasters
        for fl in [
            output_slope, 
            output_slope_reclass, 
            # output_reproj, 
            input_file, 
            output_rescale
        ]:
            try:
                os.remove(fl)
            except Exception as err:
                print(err)
    if not final_gdf.empty:
        # Clip result by region and save to output
        final_gdf = final_gdf.clip(region_gdf)
        region_shortname = get_region_shortname(region)
        final_gdf.to_file(
            # 'result/slope_limitations.gpkg', 
            f"result/{region_shortname}_limitations.gpkg", 
            layer=f"{region_shortname}_{str(rescale_size)}m_sl_more_{str(slope_threshold)}_vector"
            )
        
        return final_gdf
    return None


def prepare_wetlands_limitations(
    postgres_info='.secret/.gdcdb',
    region='', 
    regions_table='admin.hse_russia_regions', 
    wetlands_table='osm.osm_wetlands_russia_final',
    region_buf_size=0,
):
    try:
        with open(postgres_info, encoding='utf-8') as f:
            pg = json.load(f)
    except:
        raise
    try:
        engine = sqlalchemy.create_engine(
            f"postgresql+psycopg2://{pg['user']}:{pg['password']}@{pg['host']}:{pg['port']}/{pg['database']}",
            connect_args={
                "sslmode": "verify-full",
                "target_session_attrs": "read-write"
            },
        )
    except:
        raise
    try:
        sql = f"select * from {regions_table} where lower(region) = '{region.lower()}';"
        region_gdf = gpd.read_postgis(sql, engine)
        if region_buf_size > 0:
            region_buffer = calculate_geod_buffers(region_gdf, 'laea', 'value', region_buf_size, geom_field='geom')
            region_gdf = region_gdf.set_geometry(region_buffer)
        sql = f"select * from {wetlands_table} wtlnd where ST_Intersects((select ST_Buffer(geom::geography, {region_buf_size})::geometry from {regions_table} where lower(region) = '{region.lower()}' limit 1), wtlnd.geom);"
        wetlands_gdf = gpd.read_postgis(sql, engine)
        pass
    except:
        raise
    region_shortname = get_region_shortname(region)
    wetlands_gdf = wetlands_gdf.clip(region_gdf)
    wetlands_gdf.to_file(
        # 'result/wetlands_limitations.gpkg', 
        f"result/{region_shortname}_limitations.gpkg", 
        layer=f"{region_shortname}_wetlands"
        )
    return wetlands_gdf


def prepare_soil_limitations(
    postgres_info='.secret/.gdcdb',
    region='', 
    regions_table='admin.hse_russia_regions', 
    soil_table='egrpr_esoil_ru.soil_map_m2_5_v',
    region_buf_size=0,
):
    try:
        with open(postgres_info, encoding='utf-8') as f:
            pg = json.load(f)
    except:
        raise
    try:
        engine = sqlalchemy.create_engine(
            f"postgresql+psycopg2://{pg['user']}:{pg['password']}@{pg['host']}:{pg['port']}/{pg['database']}",
            connect_args={
                "sslmode": "verify-full",
                "target_session_attrs": "read-write"
            },
        )
    except:
        raise
    try:
        sql = f"select * from {regions_table} where lower(region) = '{region.lower()}';"
        region_gdf = gpd.read_postgis(sql, engine)
        if region_buf_size > 0:
            region_buffer = calculate_geod_buffers(region_gdf, 'laea', 'value', region_buf_size, geom_field='geom')
            region_gdf = region_gdf.set_geometry(region_buffer)
        sql = f"select * from {soil_table} sl " \
              f"where ST_Intersects((select ST_Buffer(geom::geography, {region_buf_size})::geometry from {regions_table} where lower(region) = '{region.lower()}' limit 1), sl.geom) " \
              f"and sl.soil0 between 163 and 171;"
        soil_gdf = gpd.read_postgis(sql, engine)
        pass
    except:
        raise
    if not soil_gdf.empty:
        region_shortname = get_region_shortname(region)
        soil_gdf = soil_gdf.clip(region_gdf)
        soil_gdf.to_file(
            # 'result/soil_limitations.gpkg', 
            f"result/{region_shortname}_limitations.gpkg", 
            layer=f"{region_shortname}_soil"
        )
        return soil_gdf
    return None


def prepare_settlements_limitations(
    postgres_info='.secret/.gdcdb',
    region='', 
    regions_table='admin.hse_russia_regions',
    nspd_table='nspd.nspd_settlements_pol',
    osm_table='osm.gis_osm_places_a_free',
    region_buf_size=0,
):
    try:
        with open(postgres_info, encoding='utf-8') as f:
            pg = json.load(f)
    except:
        raise
    try:
        engine = sqlalchemy.create_engine(
            f"postgresql+psycopg2://{pg['user']}:{pg['password']}@{pg['host']}:{pg['port']}/{pg['database']}",
            connect_args={
                "sslmode": "verify-full",
                "target_session_attrs": "read-write"
            },
        )
    except:
        raise
    try:
        sql = f"select * from {regions_table} where lower(region) = '{region.lower()}';"
        region_gdf = gpd.read_postgis(sql, engine)
        if region_buf_size > 0:
            region_buffer = calculate_geod_buffers(region_gdf, 'laea', 'value', region_buf_size, geom_field='geom')
            region_gdf = region_gdf.set_geometry(region_buffer)
        sql = f"select * from {nspd_table} nspd " \
              f"where ST_Intersects(" \
              f"(select ST_Buffer(geom::geography, {region_buf_size})::geometry from {regions_table} where lower(region) = '{region.lower()}' limit 1), " \
              f"ST_Transform(nspd.geom, 4326)" \
              f");"
        nspd_gdf = gpd.read_postgis(sql, engine)
        nspd_gdf = nspd_gdf.to_crs('EPSG:4326')
        sql = f"select * from {osm_table} osm " \
              f"where ST_Intersects(" \
              f"(select ST_Buffer(geom::geography, {region_buf_size})::geometry from {regions_table} where lower(region) = '{region.lower()}' limit 1), " \
              f"osm.geom" \
              f") " \
              f"and fclass not in ('county', 'region', 'island');"
        osm_gdf = gpd.read_postgis(sql, engine)
    except:
        raise
    
    # выбрать только те объекты из OSM, которые не пересекают объекты из NSPD
    tmp = gpd.sjoin(osm_gdf, nspd_gdf[['geom']], how='left', predicate='intersects')
    osm_gdf = tmp[tmp.index_right.isna()].drop(columns=['index_right']).copy()

    # Unify column set
    all_cols = list(set(osm_gdf.columns) | set(nspd_gdf.columns))

    # Add missing columns with NA
    for g in (osm_gdf, nspd_gdf):
        missing = [c for c in all_cols if c not in g.columns]
        for c in missing:
            g[c] = pd.NA

    # Optional: keep track of source
    osm_gdf['source'] = 'osm'
    nspd_gdf['source'] = 'nspd'

    # Concatenate
    settlements_gdf = gpd.GeoDataFrame(
        pd.concat([osm_gdf[all_cols], nspd_gdf[all_cols]], ignore_index=True),
        geometry='geom',  # or osm_gdf.geometry.name
        crs=osm_gdf.crs
    )

    if not settlements_gdf.empty:
        region_shortname = get_region_shortname(region)
        settlements_gdf = settlements_gdf.clip(region_gdf)
        settlements_gdf.to_file(
            # 'result/poppol_limitations.gpkg', 
            f"result/{region_shortname}_limitations.gpkg", 
            layer=f"poppol_merge_{region_shortname}"
        )
        return settlements_gdf
    return None


def prepare_oopt_limitations(
    postgres_info='.secret/.gdcdb',
    region='', 
    regions_table='admin.hse_russia_regions', 
    oopt_table='ecology.pkp_oopt_russia_2024',
    region_buf_size=0,
):
    try:
        with open(postgres_info, encoding='utf-8') as f:
            pg = json.load(f)
    except:
        raise
    try:
        engine = sqlalchemy.create_engine(
            f"postgresql+psycopg2://{pg['user']}:{pg['password']}@{pg['host']}:{pg['port']}/{pg['database']}",
            connect_args={
                "sslmode": "verify-full",
                "target_session_attrs": "read-write"
            },
        )
    except:
        raise
    try:
        sql = f"select * from {regions_table} where lower(region) = '{region.lower()}';"
        region_gdf = gpd.read_postgis(sql, engine)
        if region_buf_size > 0:
            region_buffer = calculate_geod_buffers(region_gdf, 'laea', 'value', region_buf_size, geom_field='geom')
            region_gdf = region_gdf.set_geometry(region_buffer)
        sql = f"select * from {oopt_table} oopt " \
            f"where ST_Intersects(" \
            f"(select ST_Buffer(geom::geography, {region_buf_size})::geometry from {regions_table} where lower(region) = '{region.lower()}' limit 1), " \
            f"oopt.geom" \
            f") " \
            f"and oopt.name !~* 'охотнич' " \
            f"and oopt.actuality ~* 'действующ';"
        oopt_gdf = gpd.read_postgis(sql, engine)
        pass
    except:
        raise
    if not oopt_gdf.empty:
        region_shortname = get_region_shortname(region)
        oopt_gdf = oopt_gdf.clip(region_gdf)
        oopt_gdf.to_file(
            # 'result/oopt_limitations.gpkg', 
            f"result/{region_shortname}_limitations.gpkg", 
            layer=f"{region_shortname}_oopt"
        )
        return oopt_gdf
    return None


def prepare_forest_limitations(
    postgres_info='.secret/.gdcdb',
    region='', 
    regions_table='admin.hse_russia_regions', 
    forest_table='forest.pkp_forest_glf',
    region_buf_size=0,
):
    try:
        with open(postgres_info, encoding='utf-8') as f:
            pg = json.load(f)
    except:
        raise
    try:
        engine = sqlalchemy.create_engine(
            f"postgresql+psycopg2://{pg['user']}:{pg['password']}@{pg['host']}:{pg['port']}/{pg['database']}",
            connect_args={
                "sslmode": "verify-full",
                "target_session_attrs": "read-write"
            },
        )
    except:
        raise
    try:
        sql = f"select * from {regions_table} where lower(region) = '{region.lower()}';"
        region_gdf = gpd.read_postgis(sql, engine)
        if region_buf_size > 0:
            region_buffer = calculate_geod_buffers(region_gdf, 'laea', 'value', region_buf_size, geom_field='geom')
            region_gdf = region_gdf.set_geometry(region_buffer)
        sql = f"select gid, COALESCE(ST_MakeValid(geom), geom) as geom, vmr from {forest_table} forest " \
            f"where ST_Intersects(" \
            f"(select ST_Buffer(geom::geography, {region_buf_size})::geometry from {regions_table} where lower(region) = '{region.lower()}' limit 1), " \
            f"forest.geom" \
            f");"
        forest_gdf = gpd.read_postgis(sql, engine)
        pass
    except:
        raise
    if not forest_gdf.empty:
        region_shortname = get_region_shortname(region)
        forest_gdf = forest_gdf.set_geometry(forest_gdf.geometry.make_valid())
        forest_gdf = forest_gdf.clip(region_gdf)
        # forest_gdf = forest_gdf.dissolve(by=['vmr'], dropna=False)
        forest_gdf = forest_gdf.dissolve(dropna=False)
        forest_gdf = forest_gdf.explode()
        forest_gdf.to_file(
            # 'result/forest_limitations.gpkg', 
            f"result/{region_shortname}_limitations.gpkg", 
            layer=f"forest_{region_shortname}"
        )
        return forest_gdf
    return None


def belt_vectorize_lulc(
    region='Липецкая область',
    lulc_link='lulc/S2LULC_10m_LAEA_48_202507081046.tif'
):
    ######################LULC######################
    current_dir = os.getcwd()
    region_shortname = get_region_shortname(region)
    if region_shortname is None:
        region_shortname = "region"

    # Открыть растр LULC и векторизовать его в GeoDataFrame
    try:
        with rasterio.open(lulc_link) as src:
            band = src.read(1)
            # Маска валидных пикселей (исключаем NoData, если задано)
            if src.nodata is not None:
                mask = band != src.nodata
            else:
                mask = np.ones(band.shape, dtype=bool)

            shapes_iter = rasterio.features.shapes(band, mask=mask, transform=src.transform)
            records = []
            for geom, value in shapes_iter:
                if geom is None:
                    continue
                try:
                    shp = shapely.geometry.shape(geom)
                    if shp.is_empty:
                        continue
                    records.append({
                        'value': int(value) if value is not None else None,
                        'geometry': shp
                    })
                except Exception:
                    # Пропускать проблемные примитивы
                    continue

            lulc_gdf = gpd.GeoDataFrame(records, geometry='geometry', crs=src.crs)            
    except Exception as e:
        raise RuntimeError(f"Failed to open or vectorize raster '{lulc_link}': {e}")
    
    # Дополнительно: создать новый растр lulc_meadows.tif, где пиксели со значением 3 -> 1, остальные -> NoData
    try:
        with rasterio.open(lulc_link) as src_re:
            band = src_re.read(1)
            out_arr = np.where(band == 3, 1, 0).astype(np.uint8)
            out_mask = (out_arr == 0)
            out_ma = np.ma.array(out_arr, mask=out_mask)

            profile = src_re.profile.copy()
            profile.update({
                'dtype': rasterio.uint8,
                'count': 1,
                'nodata': 0,
                'compress': 'LZW'
            })

            out_raster_path = os.path.join(current_dir, 'lulc', 'lulc_meadows.tif')
            out_dir = os.path.dirname(out_raster_path)
            if out_dir and not os.path.isdir(out_dir):
                os.makedirs(out_dir, exist_ok=True)

            with rasterio.open(out_raster_path, 'w', **profile) as dst:
                dst.write(out_ma, 1)
    except Exception as e:
        raise RuntimeError(f"Failed to create 'lulc_meadows.tif' from '{lulc_link}': {e}")

    # Добавить в lulc_gdc поля area_ha (float, геодезическая площадь) и name (text 50),
    # заполнить name по value и сохранить в luls.gpkg
    try:
        # Геодезическая площадь в гектарах с использованием эллипсоида WGS84
        try:
            gdf_wgs = lulc_gdf.to_crs(4326)
        except Exception:
            gdf_wgs = lulc_gdf
        geod = Geod(ellps='WGS84')
        areas_ha = []
        for geom in gdf_wgs.geometry:
            if geom is None or geom.is_empty:
                areas_ha.append(0.0)
                continue
            try:
                area_m2, _ = geod.geometry_area_perimeter(geom)
                areas_ha.append(abs(area_m2) / 10000.0)
            except Exception:
                areas_ha.append(0.0)
        lulc_gdf['area_ha'] = pd.to_numeric(areas_ha, errors='coerce')

        # Классифицированное имя по значению
        value_to_name = {
            1: 'water',
            2: 'forest',
            3: 'meadow',
            4: 'arable',
            5: 'build',
        }
        lulc_gdf['name'] = lulc_gdf['value'].map(value_to_name)
        # Ограничить длину до 50 символов на всякий случай
        lulc_gdf['name'] = lulc_gdf['name'].astype('string').str.slice(0, 50)

        # Разбить на 5 GeoDataFrame по значению поля 'name'
        water_gdf = lulc_gdf[lulc_gdf['name'] == 'water'].copy()
        forest_gdf = lulc_gdf[lulc_gdf['name'] == 'forest'].copy()
        meadow_gdf = lulc_gdf[lulc_gdf['name'] == 'meadow'].copy()
        arable_gdf = lulc_gdf[lulc_gdf['name'] == 'arable'].copy()
        build_gdf = lulc_gdf[lulc_gdf['name'] == 'build'].copy()

        water_gdf = water_gdf.to_crs(4326)
        forest_gdf = forest_gdf.to_crs(4326)
        meadow_gdf = meadow_gdf.to_crs(4326)
        arable_gdf = arable_gdf.to_crs(4326)
        build_gdf = build_gdf.to_crs(4326)

        return {
            'water_gdf': water_gdf,
            'forest_gdf': forest_gdf,
            'meadow_gdf': meadow_gdf,
            'arable_gdf': arable_gdf,
            'build_gdf': build_gdf,
        }
    except Exception as e:
        raise RuntimeError(f"Failed to build/save classified LULC GeoDataFrame: {e}")


def belt_calculate_forest_buffer(
    region='Липецкая область',
    forest_gdf=None,
    buffer_distance=50
):
    region_shortname = get_region_shortname(region)
    if region_shortname is None:
        region_shortname = "region"
        # Построить геодезические буферы 50 м вокруг forest_gdf, объединить, взорвать и сохранить как forest_50m
    if not forest_gdf.empty:
        try:
            # Убрать небольшие полигоны
            forest_gdf = forest_gdf[forest_gdf['area_ha'] > 0.1].copy()
            # Буферы считаем в геодезическом режиме: сначала в WGS84, затем используем calculate_geod_buffers
            forest_wgs = forest_gdf.to_crs(4326)
            buffer_geom = calculate_geod_buffers(
                forest_wgs, 
                buffer_crs='laea', 
                buffer_dist_source='value', 
                buffer_distance=buffer_distance, 
                geom_field='geometry'
            )
            forest_buf = forest_wgs.set_geometry(buffer_geom)
            # Union all buffered parts
            unioned = forest_buf.union_all()
            forest_50m = gpd.GeoDataFrame(gpd.GeoSeries(unioned))
            forest_50m = forest_50m.rename(columns={0: 'geometry'}).set_geometry('geometry')
            forest_50m = forest_50m.set_crs('EPSG:4326')
            # Explode to individual polygons
            try:
                forest_50m = forest_50m.explode(index_parts=False)
            except TypeError:
                forest_50m = forest_50m.explode().reset_index(drop=True)
            # Save to GPKG as 'forest_50m'
            # region_shortname = get_region_shortname(region)

            forest_50m.to_file(
                f"result/{region_shortname}_limitations.gpkg", 
                layer=f'{region_shortname}_forest_{str(buffer_distance)}m', 
                driver='GPKG'
                )
            return forest_50m
        except Exception as e:
            raise RuntimeError(f"Failed to build/save forest_50m buffers: {e}")
    return False


def belt_merge_limitation_full(
    region='Липецкая область',
    limitations_all=None,   # geodataframe derived from prepare_limitations
    road_OSM_cover_buf=None,   # geodataframe derived from calculate_road_buffer
    forest_50m=None,   # geodataframe derived from calculate_forest_buffer
):
    ###################объединение ограничений########################
    if limitations_all is not None and road_OSM_cover_buf is not None and forest_50m is not None:
        ############################################
        # Объединяем ограничения
        # Merge all limitations into a single GeoDataFrame
        src_gdfs = [
            ("limitations_all", limitations_all),
            ("road_OSM_cover_buf", road_OSM_cover_buf),
            ("forest_50m", forest_50m)
        ]
        parts = []
        for name, gdf in src_gdfs:
            if gdf is None or gdf.empty:
                continue
            # Ensure CRS is EPSG:4326
            try:
                if gdf.crs is None or str(gdf.crs).lower() not in ("epsg:4326", "epsg:4326"):
                    gdf = gdf.to_crs("EPSG:4326")
            except Exception:
                # If CRS conversion fails, keep as is
                pass
            # Add source column to keep provenance
            gdf = gdf.copy()
            gdf["src"] = name
            parts.append(gdf)
        if parts:
            # Unify columns
            all_cols = set()
            for g in parts:
                all_cols |= set(g.columns)
            for g in parts:
                for c in all_cols:
                    if c not in g.columns:
                        g[c] = pd.NA
            # Concatenate
            limitation_full = gpd.GeoDataFrame(
                pd.concat([g[list(all_cols)] for g in parts], ignore_index=True),
                geometry='geometry',
                crs='EPSG:4326'
            )
            # Explode geometries so each part is a separate feature
            try:
                limitation_full = limitation_full.explode(index_parts=False)
            except TypeError:
                # Fallback for older GeoPandas versions without index_parts
                limitation_full = limitation_full.explode().reset_index(drop=True)
            # Reset index and regenerate sequential gid
            limitation_full = limitation_full.reset_index(drop=True)
            limitation_full['gid'] = range(1, len(limitation_full) + 1)
            # Keep only gid and geom
            limitation_full = limitation_full[['gid', 'geometry']]
            limitation_full = limitation_full.dissolve(dropna=False)
            limitation_full = limitation_full.explode()
            
            # Split by 6' grid (0.1 degree) to further subdivide geometries
            step_deg = 0.1  # 6 arc-minutes
            minx, miny, maxx, maxy = limitation_full.total_bounds
            # Snap bounds to grid
            start_x = math.floor(minx / step_deg) * step_deg
            end_x = math.ceil(maxx / step_deg) * step_deg
            start_y = math.floor(miny / step_deg) * step_deg
            end_y = math.ceil(maxy / step_deg) * step_deg
            xs = np.arange(start_x, end_x, step_deg)
            ys = np.arange(start_y, end_y, step_deg)
            cells = []
            cell_ids = []
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    cells.append(shapely.geometry.box(x, y, x + step_deg, y + step_deg))
                    cell_ids.append(f"c_{i}_{j}")
            grid_gdf = gpd.GeoDataFrame({'cell_id': cell_ids, 'geometry': cells}, geometry='geometry', crs='EPSG:4326')
            # Intersect merged limitations with grid to split them
            split_gdf = gpd.overlay(limitation_full, grid_gdf, how='intersection')
            try:
                split_gdf = split_gdf.explode(index_parts=False)
            except TypeError:
                split_gdf = split_gdf.explode().reset_index(drop=True)
            # Keep consistent columns and continue downstream with split geometries
            # split_gdf = split_gdf[['gid', 'geom', 'cell_id']]
            limitation_full = split_gdf
            
            # Save merged limitations
            region_shortname = get_region_shortname(region)
            if region_shortname is None:
                region_shortname = "region"
            limitation_full.to_file(
                f"result/{region_shortname}_limitations.gpkg",
                layer=f"{region_shortname}_limitation_full"
            )
            return limitation_full
    return False


def belt_calculate_arable_buffer(
    region='Липецкая область',
    arable_gdf=None,
    arable_area_ha_threshold=10,
    arable_buffer_distance=20,
):
    region_shortname = get_region_shortname(region)
    if region_shortname is None:
        region_shortname = "region"
    ###############arable###########################
    try:
        arable_gdf = arable_gdf[arable_gdf['area_ha'] > arable_area_ha_threshold].copy()
        # arable_gdf = arable_gdf.to_crs(4326)
        if not arable_gdf.empty:
            arable_buffers_geom = calculate_geod_buffers(
                i_gdf=arable_gdf,
                buffer_crs='utm',
                buffer_dist_source='value',
                buffer_distance=arable_buffer_distance,
                geom_field='geometry'
            )
            arable_buffer = arable_gdf.copy()
            arable_buffer = arable_buffer.set_geometry(arable_buffers_geom)
            arable_union = arable_gdf.geometry.unary_union
            arable_buffer = arable_buffer.set_geometry(
                arable_buffer.geometry.difference(arable_union)
            )
            arable_buffer = arable_buffer[~arable_buffer.geometry.is_empty].copy()
        else:
            arable_buffer = gpd.GeoDataFrame(columns=arable_gdf.columns, geometry=arable_gdf.geometry.name, crs=arable_gdf.crs)
        
        arable_buffer.to_file(
            f"result/{region_shortname}_limitations.gpkg",
            layer=f"{region_shortname}_arable_buffer"
        )
        return arable_buffer
    except Exception as e:
        raise RuntimeError(f"Failed to calculate arable buffer: {e}")
    return False


def belt_calculate_arable_buffer_eliminate(
    region='Липецкая область',
    arable_buffer=None,
    meadow_gdf=None,
    limitation_full=None,
    arabale_buffer_aggregate_distance_m=15,
    arable_buffer_aggregate_threshold_ha=0.1,   # минимальная площадь агрегированного участка
    arable_hole_area_threshold_m=5000   # минимальная площадь дыры в агрегированном участке
):
    region_shortname = get_region_shortname(region)
    if region_shortname is None:
        region_shortname = "region"
    ####################буферные зоны - начало######################
    # вырезать буферные зоны по слою meadow 
    arable_buffer_lim = gpd.clip(arable_buffer, meadow_gdf)
    # cтереть участки, пересекающиеся с limitation_full
    arable_buffer_lim = gpd.overlay(arable_buffer_lim, limitation_full, how='difference')
    # разбить составные на отдельные объекты
    arable_buffer_lim = arable_buffer_lim.explode()
    # сохраняем в arable_buffer_lim
    arable_buffer_lim.to_file(
        f"result/{region_shortname}_limitations.gpkg",
        layer=f"{region_shortname}_arable_buffer_lim"
    )
    minx, miny, maxx, maxy = arable_buffer_lim.total_bounds
    distance_crs = crs.CRS.from_proj4(
        f"+proj=laea +lat_0={(miny + maxy) / 2} +lon_0={(minx + maxx) / 2} " \
        f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    arable_buffer_aggregate = arable_buffer_lim.to_crs(distance_crs)
    
    arable_buffer_aggregate = arable_buffer_aggregate.reset_index(drop=True)
    
    pairs = arable_buffer_aggregate.sjoin(
        arable_buffer_aggregate,
        predicate="dwithin",
        distance=arabale_buffer_aggregate_distance_m,
        how="left"
    )
    pairs = pairs.dropna(subset=["index_right"])
    pairs["index_right"] = pairs["index_right"].astype(int)
    pairs = pairs[pairs.index != pairs["index_right"]]
    
    rows = pairs.index.to_numpy()
    cols = pairs["index_right"].to_numpy()
    graph = coo_matrix((np.ones_like(rows), (rows, cols)), shape=(len(arable_buffer_aggregate), len(arable_buffer_aggregate)))
    _, labels = connected_components(graph)
    arable_buffer_aggregate["cluster_id"] = labels
    arable_buffer_aggregate = arable_buffer_aggregate.dissolve(by="cluster_id")
    # arable_buffer_aggregate = arable_buffer_aggregate.explode()
    arable_buffer_aggregate = arable_buffer_aggregate.to_crs(4326)
    
    geod = Geod(ellps='WGS84')
    areas_ha = []
    for geom in arable_buffer_aggregate.geometry:
        if geom is None or geom.is_empty:
            areas_ha.append(0.0)
            continue
        try:
            area_m2, _ = geod.geometry_area_perimeter(geom)
            areas_ha.append(abs(area_m2) / 10000.0)
        except Exception:
            areas_ha.append(0.0)
    arable_buffer_aggregate['area_ha'] = pd.to_numeric(areas_ha, errors='coerce')

    # Удалить объекты площадью <= 0.1 га
    arable_buffer_aggregate = arable_buffer_aggregate[arable_buffer_aggregate['area_ha'] > arable_buffer_aggregate_threshold_ha].copy()

    arable_buffer_aggregate.to_file(
        f"result/{region_shortname}_limitations.gpkg",
        layer=f"{region_shortname}_arable_buffer_aggregate"
    )

    # PAEK-like smoothing: project to metric CRS, buffer-debuffer, project back
    if not arable_buffer_aggregate.empty:
        smooth_tolerance = 5  # meters
        g_proj = arable_buffer_aggregate.to_crs(distance_crs)
        
        g_proj["geometry"] = g_proj.buffer(smooth_tolerance).buffer(-smooth_tolerance)
        arable_buffer_smooth = g_proj.to_crs(4326)

        # Recompute area after smoothing
        geod = Geod(ellps='WGS84')
        areas_ha = []
        for geom in arable_buffer_smooth.geometry:
            if geom is None or geom.is_empty:
                areas_ha.append(0.0)
                continue
            try:
                area_m2, _ = geod.geometry_area_perimeter(geom)
                areas_ha.append(abs(area_m2) / 10000.0)
            except Exception:
                areas_ha.append(0.0)
        arable_buffer_smooth['area_ha'] = pd.to_numeric(areas_ha, errors='coerce')
        arable_buffer_smooth.to_file(
            f"result/{region_shortname}_limitations.gpkg",
            layer=f"{region_shortname}_arable_buffer_smooth"
        )

    # Удаляем пустоты внутри полигонов, критерий – площадь 0,5 га
    arable_buffer_eliminate = arable_buffer_smooth.to_crs(distance_crs)
    arable_buffer_eliminate["geometry"] = arable_buffer_eliminate.geometry.apply(lambda g: drop_small_holes_any(g, min_hole_area=arable_hole_area_threshold_m))  # CRS units
    arable_buffer_eliminate = arable_buffer_eliminate.to_crs(4326)
    arable_buffer_eliminate.to_file(
        f"result/{region_shortname}_limitations.gpkg",
        layer=f"{region_shortname}_arable_buffer_eliminate"
    )
    return arable_buffer_eliminate


def belt_calculate_skeletons(
    region='Липецкая область',
    polygons_gdf=None,
    pixel_size_m=1.0,
    min_branch_length_m=0.1
):
    # Raster-skeletonize polygons and return centerlines as a GeoDataFrame.
    # Steps:
    # 1) Project to local metric CRS (LAEA centered on bbox)
    # 2) Rasterize polygons at fixed pixel size
    # 3) Skeletonize raster (thinning)
    # 4) Convert skeleton pixels to line segments (4-neighborhood), merge, prune
    # 5) Save to GeoPackage and return

    if polygons_gdf is None or polygons_gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=4326)

    region_shortname = get_region_shortname(region)
    if region_shortname is None:
        region_shortname = "region"

    # Choose local metric CRS (LAEA) centered at polygon extent
    minx, miny, maxx, maxy = polygons_gdf.total_bounds
    distance_crs = crs.CRS.from_proj4(
        f"+proj=laea +lat_0={(miny + maxy) / 2} +lon_0={(minx + maxx) / 2} "
        f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )

    # Parameters
    pixel_size_m = pixel_size_m  # meters per pixel
    min_branch_length_m = min_branch_length_m  # prune very short segments

    g_proj = polygons_gdf.to_crs(distance_crs)
    gxmin, gymin, gxmax, gymax = g_proj.total_bounds
    if not np.isfinite([gxmin, gymin, gxmax, gymax]).all():
        return gpd.GeoDataFrame(geometry=[], crs=4326)

    # Compute raster grid size
    width = max(1, int(math.ceil((gxmax - gxmin) / pixel_size_m)))
    height = max(1, int(math.ceil((gymax - gymin) / pixel_size_m)))

    # Build affine transform (origin at top-left)
    transform = rasterio.transform.from_origin(gxmin, gymax, pixel_size_m, pixel_size_m)

    # Rasterize polygons to a binary mask (1 inside polygon)
    shapes = [(geom, 1) for geom in g_proj.geometry if geom and not geom.is_empty]
    if not shapes:
        return gpd.GeoDataFrame(geometry=[], crs=4326)

    mask = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )

    # Skeletonize (boolean array where True is foreground)
    sk = skeletonize(mask.astype(bool))

    # Convert skeleton pixels to line segments by connecting 4-neighbors (right, down)
    segments = []
    def pix_center(col, row):
        x = gxmin + (col + 0.5) * pixel_size_m
        y = gymax - (row + 0.5) * pixel_size_m
        return (x, y)

    H, W = sk.shape
    for r in range(H):
        row = sk[r]
        for c in range(W):
            if not row[c]:
                continue
            # right neighbor
            if c + 1 < W and sk[r, c + 1]:
                p0 = pix_center(c, r)
                p1 = pix_center(c + 1, r)
                segments.append(shapely.geometry.LineString([p0, p1]))
            # down neighbor
            if r + 1 < H and sk[r + 1, c]:
                p0 = pix_center(c, r)
                p1 = pix_center(c, r + 1)
                segments.append(shapely.geometry.LineString([p0, p1]))

    if not segments:
        return gpd.GeoDataFrame(geometry=[], crs=4326)

    # Merge segments into lines
    seg_series = gpd.GeoSeries(segments, crs=distance_crs)
    merged = shapely.line_merge(seg_series.unary_union)

    # Normalize to iterable of lines
    lines = []
    if merged is None or merged.is_empty:
        return gpd.GeoDataFrame(geometry=[], crs=4326)
    if merged.geom_type == "LineString":
        lines = [merged]
    elif merged.geom_type == "MultiLineString":
        lines = list(merged.geoms)
    else:
        # Fallback: explode geometries we can iterate
        try:
            lines = list(merged.geoms)
        except Exception:
            lines = []

    # Prune short branches (by metric length)
    lines = [ln for ln in lines if ln.length >= min_branch_length_m]
    if not lines:
        return gpd.GeoDataFrame(geometry=[], crs=4326)

    skeleton_gdf = gpd.GeoDataFrame(geometry=lines, crs=distance_crs)
    # Clip to input polygons (to remove rasterization outside effects)
    try:
        skeleton_gdf = gpd.overlay(skeleton_gdf, g_proj[[g_proj.geometry.name]], how="intersection")
    except Exception:
        # If overlay fails (topology issues), fall back to clip
        try:
            skeleton_gdf = gpd.clip(skeleton_gdf, g_proj)
        except Exception:
            pass

    # Reproject back to WGS84 for output
    skeleton_out = skeleton_gdf.to_crs(4326)

    # Save
    try:
        out_path = os.path.join("result", f"{region_shortname}_limitations.gpkg")
        layer_name = f"{region_shortname}_arable_buffer_line"
        skeleton_out.to_file(out_path, layer=layer_name)
    except Exception as e:
        print(f"Warning: failed to save skeletons to GeoPackage: {e}")

    return skeleton_out


def belt_calculate_centerlines(
    region='Липецкая область',
    polygons_gdf=None,
    segmentize_maxlen_m = 1.0,   # densify polygon boundary before centerline
    min_branch_length_m = 0.1   # prune tiny spurs
):
    # Compute vector centerlines using the 'centerline' library.
    # 1) Project polygons to a local metric CRS (LAEA centered on bbox)
    # 2) For each Polygon in the GeoDataFrame (explode MultiPolygons), compute Centerline
    # 3) Collect LineStrings/MultiLineStrings, prune shorts, clip back and save

    if polygons_gdf is None or polygons_gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=4326)

    region_shortname = get_region_shortname(region)
    if region_shortname is None:
        region_shortname = "region"

    # Local metric CRS (LAEA centered on dataset bbox)
    minx, miny, maxx, maxy = polygons_gdf.total_bounds
    distance_crs = crs.CRS.from_proj4(
        f"+proj=laea +lat_0={(miny + maxy) / 2} +lon_0={(minx + maxx) / 2} "
        f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )

    g_proj = polygons_gdf.to_crs(distance_crs)
    if g_proj.empty:
        return gpd.GeoDataFrame(geometry=[], crs=4326)

    # Parameters
    segmentize_maxlen_m = segmentize_maxlen_m   # densify polygon boundary before centerline
    min_branch_length_m = min_branch_length_m  # prune tiny spurs

    # Ensure simple polygons only
    g_exploded = g_proj.explode(ignore_index=True)

    lines = []
    for geom in g_exploded.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type != "Polygon":
            # Skip non-polygon geometry
            continue
        try:
            cl = Centerline(geom, segmentize_maxlen=segmentize_maxlen_m)
            cl_geom = cl.geometry
        except Exception:
            continue
        if cl_geom is None or cl_geom.is_empty:
            continue
        if cl_geom.geom_type == "LineString":
            lines.append(cl_geom)
        elif cl_geom.geom_type == "MultiLineString":
            lines.extend(list(cl_geom.geoms))

    if not lines:
        return gpd.GeoDataFrame(geometry=[], crs=4326)

    # Build GeoDataFrame in metric CRS
    lines_gdf = gpd.GeoDataFrame(geometry=lines, crs=distance_crs)
    # Prune short segments
    lines_gdf = lines_gdf[lines_gdf.length >= min_branch_length_m].copy()
    if lines_gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=4326)

    # Clip to polygons to be safe
    try:
        lines_gdf = gpd.overlay(lines_gdf, g_proj[[g_proj.geometry.name]], how="intersection")
    except Exception:
        try:
            lines_gdf = gpd.clip(lines_gdf, g_proj)
        except Exception:
            pass

    # Back to WGS84
    out_gdf = lines_gdf.to_crs(4326)

    # Save result
    try:
        out_path = os.path.join("result", f"{region_shortname}_limitations.gpkg")
        layer_name = f"{region_shortname}_arable_buffer_line_centerline"
        out_gdf.to_file(out_path, layer=layer_name)
    except Exception as e:
        print(f"Warning: failed to save centerlines: {e}")

    return out_gdf



def calculate_forest_belt(
    region='Липецкая область',
    limitations_all=None,   # geodataframe derived from prepare_limitations
    lulc_link='lulc/S2LULC_10m_LAEA_48_202507081046.tif',
    postgres_info='.secret/.gdcdb',
    regions_table='admin.hse_russia_regions',
    region_buf_size=5000,
    road_table='osm.gis_osm_roads_free',
    road_buf_size_rule={
        "fclass  in  ('primary', 'primary_link')": 80,
        "fclass in ('motorway' , 'motorway_link')": 80,
        "fclass in ('secondary', 'secondary_link')": 70,
        "fclass in ('tertiary', 'tertiary_link')": 70,
        "fclass in ('trunk', 'trunk_link')": 70,
        "fclass in ('unclassified')": 70,
    },
):
    region_shortname = get_region_shortname(region)
    if region_shortname is None:
        region_shortname = "region"
    current_dir = os.getcwd()
    os.environ["PROJ_LIB"] = os.path.join(current_dir, '.venv', 'Lib', 'site-packages', 'osgeo', 'data', 'proj')
    lulcdir = os.path.join(current_dir, 'lulc')
    if not os.path.isdir(lulcdir):
        os.mkdir(lulcdir)
    
    ######################LULC######################
    try:
        lulc_gdfs = belt_vectorize_lulc(
            region=region,
            lulc_link=lulc_link
            )
        
        water_gdf = lulc_gdfs.get('water_gdf')
        forest_gdf = lulc_gdfs.get('forest_gdf')
        meadow_gdf = lulc_gdfs.get('meadow_gdf')
        arable_gdf = lulc_gdfs.get('arable_gdf')
        build_gdf = lulc_gdfs.get('build_gdf')

        for name, lulc_gdf in lulc_gdfs.items():
            lulc_gdf.to_file(
                os.path.join(current_dir, 'result', f'{region_shortname}_lulc.gpkg'), 
                layer=f"{region_shortname}_lulc_{name.replace('_gdf', '')}"
                )
        pass
    except Exception as e:
        raise RuntimeError(f"Failed to build/save classified LULC GeoDataFrame: {e}")
        
        

    ######################forest######################
    forest_50m = belt_calculate_forest_buffer(
        region=region,
        forest_gdf=forest_gdf
        )

    ######################road_OSM_cover_buf######################
    road_OSM_cover_buf = prepare_road_limitations(
        postgres_info=postgres_info,
        region=region, 
        regions_table=regions_table,
        region_buf_size=region_buf_size,
        road_table=road_table,
        road_buf_size_rule=road_buf_size_rule,
        road_buffer_crs='utm',
    )

    ###################объединение ограничений########################
    limitation_full = belt_merge_limitation_full(
        region=region,
        limitations_all=limitations_all,   # geodataframe derived from prepare_limitations
        road_OSM_cover_buf=road_OSM_cover_buf,   # geodataframe derived from calculate_road_buffer
        forest_50m=forest_50m   # geodataframe derived from calculate_forest_buffer
    )

    ###############arable###########################
    arable_buffer = belt_calculate_arable_buffer(
        region=region,
        arable_gdf=arable_gdf,
        arable_area_ha_threshold=10,
        arable_buffer_distance=20
    )

    ####################буферные зоны######################
    arable_buffer_eliminate = belt_calculate_arable_buffer_eliminate(
        region=region,
        arable_buffer=arable_buffer,
        meadow_gdf=meadow_gdf,
        limitation_full=limitation_full
    )
    

def prepare_road_limitations(
    postgres_info='.secret/.gdcdb',
    region='Липецкая область', 
    regions_table='admin.hse_russia_regions',
    region_buf_size=5000,
    road_table='osm.gis_osm_roads_free',
    road_buf_size_rule={
        "fclass  in  ('primary', 'primary_link')": 80,
        "fclass in ('motorway' , 'motorway_link')": 80,
        "fclass in ('secondary', 'secondary_link')": 70,
        "fclass in ('tertiary', 'tertiary_link')": 70,
        "fclass in ('trunk', 'trunk_link')": 70,
        "fclass in ('unclassified')": 70,
    },
    road_buffer_crs='utm',
):
    try:
        with open(postgres_info, encoding='utf-8') as f:
            pg = json.load(f)
    except:
        raise
    try:
        engine = sqlalchemy.create_engine(
            f"postgresql+psycopg2://{pg['user']}:{pg['password']}@{pg['host']}:{pg['port']}/{pg['database']}",
            connect_args={
                "sslmode": "verify-full",
                "target_session_attrs": "read-write"
            },
        )
    except:
        raise
    try:
        sql = f"select * from {regions_table} where lower(region) = '{region.lower()}';"
        with engine.connect() as conn:
            region_gdf = gpd.read_postgis(sql, conn)
        if region_buf_size > 0:
            region_buffer = calculate_geod_buffers(region_gdf, 'laea', 'value', region_buf_size, geom_field='geom')
            region_gdf = region_gdf.set_geometry(region_buffer)
        
        sql = f"select * from {road_table} road " \
            f"where (ST_Intersects(" \
            f"(select ST_Buffer(geom::geography, {region_buf_size})::geometry from {regions_table} where lower(region) = '{region.lower()}' limit 1), " \
            f"road.geom" \
            f")) " \
            f"and ((road.fclass in ('primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'trunk', 'trunk_link', 'motorway', 'motorway_link')) or (road.fclass = 'unclassified' and road.ref !~ ' .*'));"

        with engine.connect() as conn:
            road_gdf = gpd.read_postgis(sql, conn)

        # Calculate buffer size per road based on provided rules
        # Prefer 'fname' if available, otherwise use 'fclass'
        match_field = 'fclass'
        if match_field not in road_gdf.columns:
            raise RuntimeError(
                f"'fclass' not exists in {road_table}; available columns: {list(road_gdf.columns)}"
            )

        # Initialize column
        road_gdf['buf_size'] = np.nan

        # Parse rule keys like "fclass in ('primary','secondary')" or "fclass = 'unclassified'"
        token_re = re.compile(r"'([^']+)'")
        for rule, size in road_buf_size_rule.items():
            # Extract all quoted tokens
            tokens = token_re.findall(rule)
            if tokens:
                mask = road_gdf[match_field].isin(tokens)
                road_gdf.loc[mask, 'buf_size'] = size
            else:
                # Fallback: if rule text directly equals a single value (no quotes), try exact match
                value = rule.strip()
                if value:
                    road_gdf.loc[road_gdf[match_field] == value, 'buf_size'] = size

        # Convert to nullable integer type if possible
        try:
            road_gdf['buf_size'] = road_gdf['buf_size'].astype('Int64')
        except Exception:
            pass
        # Build per-feature buffers using buf_size field
        road_buffers = calculate_geod_buffers(
            i_gdf=road_gdf,
            buffer_crs=road_buffer_crs,
            buffer_dist_source='field',
            buffer_distance='buf_size',
            geom_field='geom'
        )
        road_OSM_cover_buf = road_gdf.copy()
        road_OSM_cover_buf = road_OSM_cover_buf.set_geometry(road_buffers)
        
        # # Union all buffer geometries and explode to individual parts
        # unioned = road_OSM_cover_buf.union_all()
        # parts = gpd.GeoSeries(unioned, crs=road_OSM_cover_buf.crs).explode(index_parts=False)
        # road_OSM_cover_buf = gpd.GeoDataFrame(geometry=parts, crs=road_OSM_cover_buf.crs).reset_index(drop=True)
    except:
        raise
    pass

    if not road_OSM_cover_buf.empty:
        region_shortname = get_region_shortname(region)
        road_OSM_cover_buf = road_OSM_cover_buf.set_geometry(road_OSM_cover_buf.geometry.make_valid())
        road_OSM_cover_buf = road_OSM_cover_buf.clip(region_gdf)
        # forest_gdf = forest_gdf.dissolve(by=['vmr'], dropna=False)
        road_OSM_cover_buf = road_OSM_cover_buf.dissolve(dropna=False)
        road_OSM_cover_buf = road_OSM_cover_buf.explode()
        road_OSM_cover_buf.to_file(
            # 'result/forest_limitations.gpkg', 
            f"result/{region_shortname}_limitations.gpkg", 
            layer=f"{region_shortname}_road_OSM_cover_buf"
        )
        return road_OSM_cover_buf
    return None


def prepare_limitations(
    postgres_info='.secret/.gdcdb',
    region='Липецкая область', 
    regions_table='admin.hse_russia_regions',
    region_buf_size=5000,
    water_line_table='osm.gis_osm_waterways_free',
    water_pol_table='osm.gis_osm_water_a_free',
    hydro_buffer_distance_m = 5,
    hydro_buffer_crs='utm',
    wetlands_table='osm.osm_wetlands_russia_final',
    soil_table='egrpr_esoil_ru.soil_map_m2_5_v',
    fabdem_tiles_table='elevation.fabdem_v1_2_tiles',
    slope_threshold=12,
    fabdem_zip_path=r"\\172.21.204.20\geodata\_PROJECTS\pkp\vm0047_prod\dem_fabdem",
    rescale_slope_raster=True,
    slope_raster_rescale_size_m=10,
    nspd_settlements_table='nspd.nspd_settlements_pol',
    osm_settlements_table='osm.gis_osm_places_a_free',
    oopt_table='ecology.pkp_oopt_russia_2024',
    forest_table='forest.pkp_forest_glf'
):
    
    water_poly, water_lines_buf, water_prot_zone = prepare_water_limitations(
        postgres_info=postgres_info,
        region=region,
        water_line_table=water_line_table,
        water_pol_table=water_pol_table,
        regions_table=regions_table,
        region_buf_size=region_buf_size,
        buffer_distance = hydro_buffer_distance_m,
        buffer_crs = hydro_buffer_crs
        )

    wetland = prepare_wetlands_limitations(
        postgres_info=postgres_info,
        region=region,
        regions_table=regions_table,
        wetlands_table=wetlands_table,
        region_buf_size=region_buf_size,
    )

    soil = prepare_soil_limitations(
        postgres_info=postgres_info,
        region=region,
        regions_table=regions_table,
        soil_table=soil_table,
        region_buf_size=region_buf_size,
    )

    slope_more_12 = prepare_slope_limitations(
        postgres_info=postgres_info,
        region=region,
        slope_threshold=slope_threshold,
        regions_table=regions_table,
        region_buf_size=region_buf_size,
        fabdem_tiles_table=fabdem_tiles_table,
        fabdem_zip_path=fabdem_zip_path,
        rescale=rescale_slope_raster,
        rescale_size=slope_raster_rescale_size_m
    )

    poppol_merge = prepare_settlements_limitations(
        postgres_info=postgres_info,
        region=region,
        regions_table=regions_table,
        nspd_table=nspd_settlements_table,
        osm_table=osm_settlements_table,
        region_buf_size=region_buf_size
    )

    oopt = prepare_oopt_limitations(
        postgres_info=postgres_info,
        region=region,
        regions_table=regions_table,
        oopt_table=oopt_table,
        region_buf_size=region_buf_size
    )

    forest = prepare_forest_limitations(
        postgres_info=postgres_info,
        region=region,
        regions_table=regions_table,
        forest_table=forest_table,
        region_buf_size=region_buf_size
    )

    # Merge all limitations into a single GeoDataFrame
    src_gdfs = [
        ("water_poly", water_poly),
        ("water_lines_buf", water_lines_buf),
        ("water_prot_zone", water_prot_zone),
        ("wetland", wetland),
        ("soil", soil),
        ("slope_more_12", slope_more_12),
        ("poppol_merge", poppol_merge),
        ("oopt", oopt),
        ("forest", forest),
    ]
    parts = []
    for name, gdf in src_gdfs:
        if gdf is None or gdf.empty:
            continue
        # Ensure CRS is EPSG:4326
        try:
            if gdf.crs is None or str(gdf.crs).lower() not in ("epsg:4326", "epsg:4326"):
                gdf = gdf.to_crs("EPSG:4326")
        except Exception:
            # If CRS conversion fails, keep as is
            pass
        # Add source column to keep provenance
        gdf = gdf.copy()
        gdf["src"] = name
        parts.append(gdf)
    if parts:
        # Unify columns
        all_cols = set()
        for g in parts:
            all_cols |= set(g.columns)
        for g in parts:
            for c in all_cols:
                if c not in g.columns:
                    g[c] = pd.NA
        # Concatenate
        merged_gdf = gpd.GeoDataFrame(
            pd.concat([g[list(all_cols)] for g in parts], ignore_index=True),
            geometry='geom',
            crs='EPSG:4326'
        )
        # Explode geometries so each part is a separate feature
        try:
            merged_gdf = merged_gdf.explode(index_parts=False)
        except TypeError:
            # Fallback for older GeoPandas versions without index_parts
            merged_gdf = merged_gdf.explode().reset_index(drop=True)
        # Reset index and regenerate sequential gid
        merged_gdf = merged_gdf.reset_index(drop=True)
        merged_gdf['gid'] = range(1, len(merged_gdf) + 1)
        # Keep only gid and geom
        merged_gdf = merged_gdf[['gid', 'geom']]
        merged_gdf = merged_gdf.dissolve(dropna=False)
        merged_gdf = merged_gdf.explode()
        
        # Split by 6' grid (0.1 degree) to further subdivide geometries
        step_deg = 0.1  # 6 arc-minutes
        minx, miny, maxx, maxy = merged_gdf.total_bounds
        # Snap bounds to grid
        start_x = math.floor(minx / step_deg) * step_deg
        end_x = math.ceil(maxx / step_deg) * step_deg
        start_y = math.floor(miny / step_deg) * step_deg
        end_y = math.ceil(maxy / step_deg) * step_deg
        xs = np.arange(start_x, end_x, step_deg)
        ys = np.arange(start_y, end_y, step_deg)
        cells = []
        cell_ids = []
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                cells.append(shapely.geometry.box(x, y, x + step_deg, y + step_deg))
                cell_ids.append(f"c_{i}_{j}")
        grid_gdf = gpd.GeoDataFrame({'cell_id': cell_ids, 'geom': cells}, geometry='geom', crs='EPSG:4326')
        # Intersect merged limitations with grid to split them
        split_gdf = gpd.overlay(merged_gdf, grid_gdf, how='intersection')
        try:
            split_gdf = split_gdf.explode(index_parts=False)
        except TypeError:
            split_gdf = split_gdf.explode().reset_index(drop=True)
        # Keep consistent columns and continue downstream with split geometries
        # split_gdf = split_gdf[['gid', 'geom', 'cell_id']]
        merged_gdf = split_gdf
        
        # Save merged limitations
        region_shortname = get_region_shortname(region)
        if region_shortname is None:
            region_shortname = "region"
        merged_gdf.to_file(
            f"result/{region_shortname}_limitations.gpkg",
            layer=f"{region_shortname}_all_limitations"
        )
        return merged_gdf
    return None

    
if __name__ == '__main__':
    # prepare_water_limitations(
    #     region='Липецкая область',
    #     source_water_line='data/Lipetsk_water_lines_3857/Lipetsk_water_lines_3857.shp',
    #     source_water_pol='data/Lipetsk_water_poly_3857/Lipetsk_water_poly_3857.shp',
    #     result_gpkg='result/water_limitations.gpkg',
    #     buffer_distance = 5,
    #     buffer_crs = 'utm'
    #     )
    
    # prepare_slope_limitations(
    #     region='Липецкая область',
    #     slope_threshold=12,
    #     fabdem_zip_path=r"\\172.21.204.20\geodata\_PROJECTS\pkp\vm0047_prod\dem_fabdem",
    #     rescale=True,
    #     rescale_size=10
    # )

    # prepare_wetlands_limitations(
    #     region='Липецкая область',
    #     region_buf_size=5000,
    # )

    # prepare_soil_limitations(
    #     region='Липецкая область',
    #     region_buf_size=5000,
    # )

    # prepare_settlements_limitations(
    #     region='Калужская область'
    # )

    # prepare_oopt_limitations(
    #     region='Липецкая область'
    # )

    # prepare_forest_limitations(
    #     region='Липецкая область'
    # )
        
    # limitations_all = prepare_limitations(
    #     region='Липецкая область',
    #     postgres_info='.secret/.gdcdb',
    #     regions_table='admin.hse_russia_regions',
    #     region_buf_size=5000,
    #     water_line_table='osm.gis_osm_waterways_free',
    #     water_pol_table='osm.gis_osm_water_a_free',
    #     hydro_buffer_distance_m=5,
    #     hydro_buffer_crs='utm',
    #     wetlands_table='osm.osm_wetlands_russia_final',
    #     soil_table='egrpr_esoil_ru.soil_map_m2_5_v',
    #     fabdem_tiles_table='elevation.fabdem_v1_2_tiles',
    #     slope_threshold=12,
    #     fabdem_zip_path=r"\\172.21.204.20\geodata\_PROJECTS\pkp\vm0047_prod\dem_fabdem",
    #     rescale_slope_raster=True,
    #     slope_raster_rescale_size_m=10,
    #     nspd_settlements_table='nspd.nspd_settlements_pol',
    #     osm_settlements_table='osm.gis_osm_places_a_free',
    #     oopt_table='ecology.pkp_oopt_russia_2024',
    #     forest_table='forest.pkp_forest_glf'
    # )

    limitations_all = gpd.read_file(
        'result/Lipetskaya_limitations.gpkg', 
        layer='Lipetskaya_all_limitations'
        )
    road_OSM_cover_buf = gpd.read_file(
        'result/Lipetskaya_limitations.gpkg', 
        layer='Lipetskaya_road_OSM_cover_buf'
        )
    forest_50m = gpd.read_file(
        'result/Lipetskaya_limitations.gpkg', 
        layer='Lipetskaya_forest_50m'
        )
    arable_gdf = gpd.read_file(
        'result/Lipetskaya_lulc.gpkg', 
        layer='Lipetskaya_lulc_arable'
        )
    arable_buffer = gpd.read_file(
        'result/Lipetskaya_Limitations.gpkg', 
        layer='Lipetskaya_arable_buffer'
        )
    meadow_gdf = gpd.read_file(
        'result/Lipetskaya_lulc.gpkg', 
        layer='Lipetskaya_lulc_meadow'
        )
    limitation_full = gpd.read_file(
        'result/Lipetskaya_Limitations.gpkg', 
        layer='Lipetskaya_limitation_full'
        )
    arable_buffer_eliminate = gpd.read_file(
        'result/Lipetskaya_Limitations.gpkg', 
        layer='Lipetskaya_arable_buffer_eliminate'
        )
    # calculate_forest_belt(
    #     region='Липецкая область',
    #     limitations_all=limitations_all,   # geodataframe derived from prepare_limitations
    #     lulc_link='lulc/S2LULC_10m_LAEA_48_202507081046_sample.tif',
    #     postgres_info='.secret/.gdcdb',
    #     regions_table='admin.hse_russia_regions',
    #     region_buf_size=5000,
    #     road_table='osm.gis_osm_roads_free',
    #     road_buf_size_rule={
    #         "fclass  in  ('primary', 'primary_link')": 80,
    #         "fclass in ('motorway' , 'motorway_link')": 80,
    #         "fclass in ('secondary', 'secondary_link')": 70,
    #         "fclass in ('tertiary', 'tertiary_link')": 70,
    #         "fclass in ('trunk', 'trunk_link')": 70,
    #         "fclass in ('unclassified')": 70,
    #     }
    # )

    # limitation_full = belt_merge_limitation_full(
    #     region='Липецкая область',
    #     limitations_all=limitations_all,   # geodataframe derived from prepare_limitations
    #     road_OSM_cover_buf=road_OSM_cover_buf,   # geodataframe derived from calculate_road_buffer
    #     forest_50m=forest_50m   # geodataframe derived from calculate_forest_buffer
    # )

    # arable_buffer = belt_calculate_arable_buffer(
    #     region='Липецкая область',
    #     arable_gdf=arable_gdf,
    #     arable_area_ha_threshold=10,
    #     arable_buffer_distance=20
    # )

    # arable_buffer_eliminate = belt_calculate_arable_buffer_eliminate(
    #     region='Липецкая область',
    #     arable_buffer=arable_buffer,
    #     meadow_gdf=meadow_gdf,
    #     limitation_full=limitation_full
    # )

    # arable_buffer_line = belt_calculate_skeletons(
    #     region='Липецкая область',
    #     polygons_gdf=arable_buffer_eliminate,
    #     pixel_size_m=1,
    #     min_branch_length_m=0.1
    # )

    arable_buffer_line = belt_calculate_centerlines(
        region='Липецкая область',
        polygons_gdf=arable_buffer_eliminate,
        segmentize_maxlen_m = 1.0,   # densify polygon boundary before centerline
        min_branch_length_m = 0.1   # prune tiny spurs
    )

    pass
    # prepare_road_limitations(
    #     region='Липецкая область'
    # )