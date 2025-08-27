# import shapely
import geopandas as gpd
from pyproj import Geod


def prepare_water_line_limitations(
    source='data/water_line.shp'
):
    """Функция рассчитывает ограничения для проекта ПКП для линейных водных объектов по методике Жанны. 
    Документация дорабатывается.

    Args:
        source (str, optional): Исходные данные. Defaults to 'data/water_line.shp'.
    """
    try:
        gdf = gpd.read_file(source)
    except:
        raise
    
    # сливаем геометрию по полям fclass и name
    gdf = gdf.dissolve(by=['fclass', 'name'], dropna=False)
    
    # Объединяем куски полилиний
    gdf['geometry'] = gdf['geometry'].line_merge()
    
    # пересчитать в WGS-84 для вычисления геодезических расстояний
    gdf = gdf.to_crs(4326)
    
    # рассчитать геодезические длины
    geod = Geod(ellps='WGS84')
    gdf['length_geod'] = [geod.geometry_length(row['geometry']) for i, row in gdf.iterrows()]    
    # gdf.to_file('result/water_line.gpkg', layer='water_line_dissolved')

    # разбить мульти полилинии на отдельные
    gdf = gdf.explode()
    
    # рассчитать геодезические длины после разбивки на отдельные линии
    gdf['length_geod'] = [geod.geometry_length(row['geometry']) for i, row in gdf.iterrows()]    
    # gdf.to_file('result/water_line.gpkg', layer='water_line_dissolved_exploded')
    
    # отфильтровать ручьи и каналы короче 10 км
    gdf = gdf.query('not (fclass in("stream", "canal") and length_geod < 10000)')
    gdf.to_file('result/water_line.gpkg', layer='water_line_dissolved_exploded_filtered')
    
    pass


if __name__ == '__main__':
    prepare_water_line_limitations(source='data/Lipetsk_water_lines_3857/Lipetsk_water_lines_3857.shp')