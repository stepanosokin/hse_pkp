# import shapely
import geopandas as gpd
import pandas as pd
from pyproj import Geod, crs, Transformer
import shapely
import re


def calculate_geod_buffers(
    i_gdf: gpd.GeoDataFrame,
    buffer_crs: str,
    buffer_dist_source: str,
    buffer_distance
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
        lon = row['geometry'].centroid.x
        lat = row['geometry'].centroid.y
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
            buffer = shapely.buffer(shapely.transform(row['geometry'], transformer1.transform, interleaved=False), float(buffer_distance))
        elif buffer_dist_source == 'field':
            buffer = shapely.buffer(shapely.transform(row['geometry'], transformer1.transform, interleaved=False), float(row[buffer_distance]))
        else:
            raise ValueError("Неверно задан параметр buffer_dist_source")
        # Пересчет буфера обратно в WGS-1984
        buffer = shapely.transform(buffer, transformer2.transform, interleaved=False)
        # Добавить результат в список
        buffer_geom.append(buffer)
    return buffer_geom


def prepare_water_limitations(
    source_water_line='data/water_line.shp',
    source_water_pol='data/water_pol.shp',
    result_gpkg='result/water_limitations.gpkg',
    buffer_distance = 5,
    buffer_crs = 'utm'
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
    try:
        gdf_l = gpd.read_file(source_water_line)
        gdf_p = gpd.read_file(source_water_pol)
    except:
        raise
    
    # сливаем геометрию по полям fclass и name
    gdf_l = gdf_l.dissolve(by=['fclass', 'name'], dropna=False)
    
    # Объединяем куски полилиний
    gdf_l['geometry'] = gdf_l['geometry'].line_merge()
    
    # пересчитать в WGS-84 для вычисления геодезических расстояний
    gdf_l = gdf_l.to_crs(4326)
    
    # рассчитать геодезические длины
    geod = Geod(ellps='WGS84')
    gdf_l['leng'] = [geod.geometry_length(row['geometry']) for i, row in gdf_l.iterrows()]    
    # gdf.to_file(result_gpkg, layer='water_line_dissolved')

    # разбить мульти полилинии на отдельные
    gdf_l = gdf_l.explode()
    
    # рассчитать геодезические длины после разбивки на отдельные линии
    gdf_l['leng'] = [geod.geometry_length(row['geometry']) for i, row in gdf_l.iterrows()]    
    # gdf.to_file(result_gpkg, layer='water_line_dissolved_exploded')
    
    # отфильтровать ручьи и каналы короче 10 км
    gdf_l = gdf_l.query('not (fclass in("stream", "canal") and leng < 10000)')
    # gdf_l.to_file(result_gpkg, layer='water_line_dissolved_exploded_filtered')
    
    # Расчет буферов вокруг линейных водных объектов
    buffer_geom = calculate_geod_buffers(gdf_l, buffer_crs, 'value', buffer_distance)
    
    # gdf_l['buffer_geometry'] = buffer_geom
    gdf_l_buffered = gdf_l.set_geometry(buffer_geom)
    my_columns = gdf_l_buffered.columns.tolist()
    # my_columns.remove('buffer_geometry')
    gdf_l_buffered = gdf_l_buffered[my_columns]
    # расчет значений размеров буферов
    gdf_l_buffered['buf'] = [50 if x <= 10000 else 100 if 10000 < x <= 50000 else 200 for x in gdf_l_buffered['leng']]
    gdf_l['buf'] = [50 if x <= 10 else 100 if 10000 < x <= 50000 else 200 for x in gdf_l['leng']]
    # gdf_l.to_file(result_gpkg, layer='water_lines_buf')
    # вычислить геодезические буферы по полю buf  
    buffer_geom = calculate_geod_buffers(gdf_l_buffered, buffer_crs, 'field', 'buf')
    wp2 = gdf_l_buffered.set_geometry(buffer_geom)
    # wp2.to_file(result_gpkg, layer='wp2')
    
    # ---------------------------------------
    # обработка полигональных водных объектов
    
    # перепроецировать в WGS-1984
    gdf_p = gdf_p.to_crs(4326)
    # убрать все wetland
    gdf_p = gdf_p.query('not fclass in("wetland")')
    
    # Вычисление "длин" площадных объектов
    # Сделать пространственное присоединение обработанных линейных объектов по пересечению
    gdf_l = gdf_l.clip(gdf_p)
    gdf_l['leng'] = [geod.geometry_length(row['geometry']) for i, row in gdf_l.iterrows()]
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
    gdf_p['area_km'] = [abs(geod.geometry_area_perimeter(row['geometry'])[0]) / 1000000 for i, row in gdf_p.iterrows()]
    
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
    water_prot_zone = water_prot_zone.rename(columns={0:'geometry'}).set_geometry('geometry')
    water_prot_zone = water_prot_zone.set_crs('epsg:4326')
    water_prot_zone = water_prot_zone.explode()
    
    water_prot_zone.to_file(result_gpkg, layer='water_prot_zone')


if __name__ == '__main__':
    prepare_water_limitations(
        source_water_line='data/Lipetsk_water_lines_3857/Lipetsk_water_lines_3857.shp',
        source_water_pol='data/Lipetsk_water_poly_3857/Lipetsk_water_poly_3857.shp',
        result_gpkg='result/water_limitations.gpkg',
        buffer_distance = 5,
        buffer_crs = 'utm'
        )
    