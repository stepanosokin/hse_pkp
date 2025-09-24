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
import rasterio.features
import numpy as np
from tqdm import tqdm
import math

# Сложности https://github.com/astral-sh/uv/issues/11466
from osgeo import gdal
from osgeo import gdal_array
from osgeo_utils import gdal_calc
from vgdb_general import smart_http_request
import requests
import yadisk


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


def download_from_y_obj_storage():
    pass
    # https://yandex.cloud/ru/docs/storage/s3/  
    # https://yandex.cloud/ru/docs/iam/operations/iam-token/create-for-sa#via-jwt
    # https://yandex.cloud/ru/docs/storage/operations/objects/download
    # https://yandex.cloud/ru/docs/storage/s3/
    # https://yandex.cloud/ru/docs/storage/s3/api-ref/object/get
    # https://yandex.cloud/en/docs/datasphere/operations/data/connect-to-s3

def calculate_forest_belt(
    region='Липецкая область',
    lulc_link=''
):
    pass
    # https://yandex.cloud/ru/docs/storage/s3/  
    # https://yandex.cloud/ru/docs/iam/operations/iam-token/create-for-sa#via-jwt
    # https://yandex.cloud/ru/docs/storage/operations/objects/download
    
    


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
        i_gdf (gpd.GeoDataFrame): входной набор данных
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
        
    # prepare_limitations(
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

    get_y_token()